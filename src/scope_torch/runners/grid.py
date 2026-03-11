from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch

from ..biochem import BiochemicalOptions, LeafBiochemistryInputs, LeafBiochemistryResult
from ..canopy.fluorescence import CanopyFluorescenceModel, CanopyFluorescenceResult
from ..canopy.reflectance import CanopyReflectanceModel, CanopyReflectanceResult
from ..canopy.thermal import CanopyThermalRadianceModel, CanopyThermalRadianceResult, ThermalOptics
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
        self.fluorescence_model = CanopyFluorescenceModel(self.reflectance_model)
        self.thermal_model = CanopyThermalRadianceModel(self.reflectance_model)

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

    def run_biochemical_fluorescence(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        biochem_options: Optional[BiochemicalOptions] = None,
        hotspot_var: Optional[str] = None,
        nlayers: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        physiology_fields = [name for name in LeafBiochemistryResult.__dataclass_fields__ if name != "fcount"]
        outputs: dict[str, list[torch.Tensor]] = {
            **{name: [] for name in CanopyFluorescenceResult.__dataclass_fields__},
            "Pnu_Cab": [],
            "Pnh_Cab": [],
            **{f"sunlit_{name}": [] for name in physiology_fields},
            **{f"shaded_{name}": [] for name in physiology_fields},
        }
        for batch in data_module.iter_batches():
            leaf_kwargs = self._leafbio_kwargs(batch, varmap)
            leafbio = LeafBioBatch(**leaf_kwargs)
            biochem = LeafBiochemistryInputs(**self._biochemistry_kwargs(batch, varmap))
            lai = batch[varmap["LAI"]]
            tts = batch[varmap["tts"]]
            tto = batch[varmap["tto"]]
            psi = batch[varmap["psi"]]
            Esun = self._spectral_input(batch, varmap, "Esun_")
            Esky = self._spectral_input(batch, varmap, "Esky_")
            soil = self._soil_refl(batch, varmap)
            if hotspot_var and hotspot_var in batch:
                hotspot = batch[hotspot_var]
            else:
                hotspot = torch.full_like(lai, self.default_hotspot)

            result = self.fluorescence_model.layered_biochemical(
                leafbio,
                biochem,
                soil,
                lai,
                tts,
                tto,
                psi,
                Esun,
                Esky,
                Csu=batch[varmap["Csu"]],
                Csh=batch[varmap["Csh"]],
                ebu=batch[varmap["ebu"]],
                ebh=batch[varmap["ebh"]],
                Tcu=batch[varmap["Tcu"]],
                Tch=batch[varmap["Tch"]],
                Oa=batch[varmap["Oa"]],
                p=batch[varmap["p"]],
                fV=batch[varmap["fV"]] if "fV" in varmap and varmap["fV"] in batch else 1.0,
                biochem_options=biochem_options,
                hotspot=hotspot,
                nlayers=self._layer_count(nlayers, etau=None, etah=None, Tcu=batch[varmap["Tcu"]], Tch=batch[varmap["Tch"]]),
            )
            for name in CanopyFluorescenceResult.__dataclass_fields__:
                outputs[name].append(getattr(result.fluorescence, name))
            outputs["Pnu_Cab"].append(result.Pnu_Cab)
            outputs["Pnh_Cab"].append(result.Pnh_Cab)
            for name in physiology_fields:
                outputs[f"sunlit_{name}"].append(getattr(result.sunlit, name))
                outputs[f"shaded_{name}"].append(getattr(result.shaded, name))

        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}

    def run_fluorescence(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs: dict[str, list[torch.Tensor]] = {name: [] for name in CanopyFluorescenceResult.__dataclass_fields__}
        for batch in data_module.iter_batches():
            leaf_kwargs = self._leafbio_kwargs(batch, varmap)
            leafbio = LeafBioBatch(**leaf_kwargs)
            lai = batch[varmap["LAI"]]
            tts = batch[varmap["tts"]]
            tto = batch[varmap["tto"]]
            psi = batch[varmap["psi"]]
            excitation = self._excitation(batch, varmap)
            soil = self._soil_refl(batch, varmap)
            if hotspot_var and hotspot_var in batch:
                hotspot = batch[hotspot_var]
            else:
                hotspot = torch.full_like(lai, self.default_hotspot)

            result = self.fluorescence_model(
                leafbio,
                soil,
                lai,
                tts,
                tto,
                psi,
                excitation,
                hotspot=hotspot,
            )
            for name in outputs:
                outputs[name].append(getattr(result, name))

        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}

    def run_layered_fluorescence(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: Optional[str] = None,
        nlayers: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs: dict[str, list[torch.Tensor]] = {name: [] for name in CanopyFluorescenceResult.__dataclass_fields__}
        for batch in data_module.iter_batches():
            leaf_kwargs = self._leafbio_kwargs(batch, varmap)
            leafbio = LeafBioBatch(**leaf_kwargs)
            lai = batch[varmap["LAI"]]
            tts = batch[varmap["tts"]]
            tto = batch[varmap["tto"]]
            psi = batch[varmap["psi"]]
            Esun = self._spectral_input(batch, varmap, "Esun_")
            Esky = self._spectral_input(batch, varmap, "Esky_")
            soil = self._soil_refl(batch, varmap)
            etau = batch[varmap["etau"]] if "etau" in varmap and varmap["etau"] in batch else None
            etah = batch[varmap["etah"]] if "etah" in varmap and varmap["etah"] in batch else None
            if hotspot_var and hotspot_var in batch:
                hotspot = batch[hotspot_var]
            else:
                hotspot = torch.full_like(lai, self.default_hotspot)

            result = self.fluorescence_model.layered(
                leafbio,
                soil,
                lai,
                tts,
                tto,
                psi,
                Esun,
                Esky,
                etau=etau,
                etah=etah,
                hotspot=hotspot,
                nlayers=self._layer_count(nlayers, etau=etau, etah=etah, Tcu=None, Tch=None),
            )
            for name in outputs:
                outputs[name].append(getattr(result, name))

        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}

    def run_thermal(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: Optional[str] = None,
        nlayers: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        outputs: dict[str, list[torch.Tensor]] = {name: [] for name in CanopyThermalRadianceResult.__dataclass_fields__}
        for batch in data_module.iter_batches():
            lai = batch[varmap["LAI"]]
            tts = batch[varmap["tts"]]
            tto = batch[varmap["tto"]]
            psi = batch[varmap["psi"]]
            Tcu = batch[varmap["Tcu"]]
            Tch = batch[varmap["Tch"]]
            Tsu = batch[varmap["Tsu"]]
            Tsh = batch[varmap["Tsh"]]
            if hotspot_var and hotspot_var in batch:
                hotspot = batch[hotspot_var]
            else:
                hotspot = torch.full_like(lai, self.default_hotspot)
            thermal_optics = ThermalOptics(
                rho_thermal=batch[varmap["rho_thermal"]] if "rho_thermal" in varmap and varmap["rho_thermal"] in batch else 0.01,
                tau_thermal=batch[varmap["tau_thermal"]] if "tau_thermal" in varmap and varmap["tau_thermal"] in batch else 0.01,
                rs_thermal=batch[varmap["rs_thermal"]] if "rs_thermal" in varmap and varmap["rs_thermal"] in batch else 0.06,
            )

            result = self.thermal_model(
                lai,
                tts,
                tto,
                psi,
                Tcu,
                Tch,
                Tsu,
                Tsh,
                thermal_optics=thermal_optics,
                hotspot=hotspot,
                nlayers=self._layer_count(nlayers, etau=None, etah=None, Tcu=Tcu, Tch=Tch),
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

    def _biochemistry_kwargs(self, batch: Mapping[str, torch.Tensor], varmap: Mapping[str, str]) -> Dict[str, torch.Tensor]:
        kwargs: Dict[str, torch.Tensor] = {}
        for field in LeafBiochemistryInputs.__dataclass_fields__:
            if field == "TDP":
                continue
            if field in varmap and varmap[field] in batch:
                kwargs[field] = batch[varmap[field]]
        return kwargs

    def _excitation(self, batch: Mapping[str, torch.Tensor], varmap: Mapping[str, str]) -> torch.Tensor:
        if "excitation" not in varmap or varmap["excitation"] not in batch:
            raise KeyError("varmap must provide 'excitation' for fluorescence runs")
        return batch[varmap["excitation"]]

    def _spectral_input(self, batch: Mapping[str, torch.Tensor], varmap: Mapping[str, str], key: str) -> torch.Tensor:
        if key not in varmap or varmap[key] not in batch:
            raise KeyError(f"varmap must provide '{key}'")
        return batch[varmap[key]]

    def _layer_count(
        self,
        nlayers: Optional[int],
        *,
        etau: Optional[torch.Tensor],
        etah: Optional[torch.Tensor],
        Tcu: Optional[torch.Tensor],
        Tch: Optional[torch.Tensor],
    ) -> Optional[int]:
        for value in (etau, etah, Tcu, Tch):
            if value is None:
                continue
            tensor = torch.as_tensor(value)
            if tensor.ndim >= 2:
                return int(tensor.shape[1])
        return nlayers

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
