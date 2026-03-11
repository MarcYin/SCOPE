from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch

from ..canopy.reflectance import CanopyReflectanceModel, CanopyReflectanceResult
from ..canopy.foursail import FourSAILModel
from ..spectral.fluspect import FluspectModel, LeafBioBatch
from ..spectral.loaders import SoilSpectraLibrary, load_fluspect_resources, load_soil_spectra
from ..spectral.soil import SoilBSMModel, SoilEmpiricalParams
from ..data import ScopeGridDataModule


class ScopeGridRunner:
    """Dispatch batched SCOPE simulations across ROI/time grids."""

    def __init__(
        self,
        fluspect: FluspectModel,
        sail: FourSAILModel,
        *,
        lidf: torch.Tensor,
        default_hotspot: float = 0.2,
        soil_spectra: Optional[SoilSpectraLibrary] = None,
        soil_bsm: Optional[SoilBSMModel] = None,
        soil_index_base: int = 1,
    ) -> None:
        self.fluspect = fluspect
        self.sail = sail
        self.lidf = lidf
        self.default_hotspot = default_hotspot
        self.soil_spectra = soil_spectra
        self.soil_bsm = soil_bsm
        self.soil_index_base = soil_index_base
        self.reflectance_model = CanopyReflectanceModel(
            fluspect,
            sail,
            lidf=lidf,
            default_hotspot=default_hotspot,
            soil_spectra=soil_spectra,
            soil_bsm=soil_bsm,
            soil_index_base=soil_index_base,
        )

    @classmethod
    def from_scope_assets(
        cls,
        *,
        lidf: torch.Tensor,
        sail: Optional[FourSAILModel] = None,
        fluspect_path: Optional[str] = None,
        soil_path: Optional[str] = None,
        scope_root_path: Optional[str] = None,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
        ndub: int = 15,
        doublings_step: int = 5,
        default_hotspot: float = 0.2,
        soil_index_base: int = 1,
        soil_empirical: SoilEmpiricalParams | None = None,
    ) -> "ScopeGridRunner":
        resources = load_fluspect_resources(
            fluspect_path,
            scope_root_path=scope_root_path,
            device=device,
            dtype=dtype,
        )
        fluspect = FluspectModel(
            resources.spectral,
            resources.optipar,
            ndub=ndub,
            doublings_step=doublings_step,
            device=torch.device(device) if device is not None else resources.optipar.Kw.device,
            dtype=dtype,
        )
        soil_spectra = load_soil_spectra(
            soil_path,
            scope_root_path=scope_root_path,
            device=fluspect.device,
            dtype=fluspect.dtype,
        )
        soil_bsm = SoilBSMModel.from_resources(resources, empirical=soil_empirical, device=fluspect.device, dtype=fluspect.dtype)
        sail_model = sail if sail is not None else FourSAILModel(lidf=lidf)
        return cls(
            fluspect,
            sail_model,
            lidf=lidf,
            default_hotspot=default_hotspot,
            soil_spectra=soil_spectra,
            soil_bsm=soil_bsm,
            soil_index_base=soil_index_base,
        )

    def run(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs: dict[str, list[torch.Tensor]] = {name: [] for name in CanopyReflectanceResult.__dataclass_fields__}
        for batch in data_module.iter_batches():
            leaf_kwargs = self._leafbio_kwargs(batch, varmap)
            leafbio = LeafBioBatch(**leaf_kwargs)
            lai = batch[varmap["LAI"]]
            tts = batch[varmap["tts"]]
            tto = batch[varmap["tto"]]
            psi = batch[varmap["psi"]]
            soil = self._soil_refl(batch, varmap)
            if hotspot_var and hotspot_var in batch:
                hotspot = batch[hotspot_var]
            else:
                hotspot = torch.full_like(lai, self.default_hotspot)

            result = self.reflectance_model(
                leafbio,
                soil,
                lai,
                tts,
                tto,
                psi,
                hotspot=hotspot,
            )
            for name in outputs:
                outputs[name].append(getattr(result, name))

        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}

    def _leafbio_kwargs(self, batch: Mapping[str, torch.Tensor], varmap: Mapping[str, str]) -> Dict[str, torch.Tensor]:
        kwargs: Dict[str, torch.Tensor] = {}
        for field in LeafBioBatch.__dataclass_fields__:
            if field in varmap and varmap[field] in batch:
                kwargs[field] = batch[varmap[field]]
        return kwargs

    def _soil_refl(self, batch: Mapping[str, torch.Tensor], varmap: Mapping[str, str]) -> torch.Tensor:
        if "soil_refl" in varmap and varmap["soil_refl"] in batch:
            return self.reflectance_model.soil_reflectance(soil_refl=batch[varmap["soil_refl"]])
        if "soil_spectrum" in varmap and varmap["soil_spectrum"] in batch:
            return self.reflectance_model.soil_reflectance(soil_spectrum=batch[varmap["soil_spectrum"]])

        bsm_fields = ("BSMBrightness", "BSMlat", "BSMlon", "SMC")
        if all(field in varmap and varmap[field] in batch for field in bsm_fields):
            return self.reflectance_model.soil_reflectance(
                BSMBrightness=batch[varmap["BSMBrightness"]],
                BSMlat=batch[varmap["BSMlat"]],
                BSMlon=batch[varmap["BSMlon"]],
                SMC=batch[varmap["SMC"]],
            )
        raise KeyError("varmap must provide either 'soil_refl', 'soil_spectrum', or all BSM soil parameter keys")
