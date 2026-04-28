from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import xarray as xr

logger = logging.getLogger(__name__)

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover
    _tqdm = None


def _progress(iterable, total=None, desc=None, disable=False):
    """Thin wrapper: use tqdm when available, else iterate silently."""
    if _tqdm is not None and not disable:
        return _tqdm(iterable, total=total, desc=desc)
    return iterable


from ..biochem import BiochemicalOptions, LeafBiochemistryInputs, LeafBiochemistryResult
from ..canopy.fluorescence import (
    CanopyFluorescenceModel,
    CanopyFluorescenceProfileResult,
    CanopyFluorescenceResult,
)
from ..canopy.foursail import FourSAILModel
from ..canopy.reflectance import (
    CanopyRadiationProfileResult,
    CanopyReflectanceModel,
    CanopyReflectanceResult,
)
from ..canopy.thermal import (
    CanopyThermalProfileResult,
    CanopyThermalRadianceModel,
    CanopyThermalRadianceResult,
    ThermalOptics,
    default_thermal_wavelengths,
)
from ..data import ScopeGridDataModule
from ..energy import (
    CanopyEnergyBalanceModel,
    CanopyEnergyBalanceResult,
    EnergyBalanceCanopy,
    EnergyBalanceMeteo,
    EnergyBalanceOptions,
    EnergyBalanceSoil,
)
from ..spectral.fluspect import FluspectModel, LeafBioBatch
from ..spectral.loaders import SoilSpectraLibrary, load_fluspect_resources, load_soil_spectra
from ..spectral.soil import SoilBSMModel, SoilEmpiricalParams
from ..variables import annotate_dataset


@dataclass(slots=True)
class _EnergyBatchInputs:
    leafbio: LeafBioBatch
    biochem: LeafBiochemistryInputs
    lai: torch.Tensor
    tts: torch.Tensor
    tto: torch.Tensor
    psi: torch.Tensor
    Esun_sw: torch.Tensor
    Esky_sw: torch.Tensor
    Esun_lw: torch.Tensor | None
    Esky_lw: torch.Tensor | None
    soil_refl: torch.Tensor
    hotspot: torch.Tensor
    meteo: EnergyBalanceMeteo
    canopy: EnergyBalanceCanopy
    soil: EnergyBalanceSoil
    nlayers: int | None
    Tcu0: torch.Tensor | None
    Tch0: torch.Tensor | None
    Tsu0: torch.Tensor | None
    Tsh0: torch.Tensor | None


class _TensorBatchDataModule:
    def __init__(self, batch: Mapping[str, torch.Tensor], attrs: Mapping[str, object] | None = None) -> None:
        self._batch = batch
        self.dataset = xr.Dataset(attrs=dict(attrs or {}))

    def iter_batches(self):
        yield self._batch


class ScopeGridRunner:
    """Dispatch batched SCOPE simulations across ROI/time grids."""

    def __init__(
        self,
        fluspect: FluspectModel,
        sail: FourSAILModel,
        *,
        lidf: torch.Tensor,
        default_hotspot: float = 0.2,
        soil_spectra: SoilSpectraLibrary | None = None,
        soil_bsm: SoilBSMModel | None = None,
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
        self.energy_balance_model = CanopyEnergyBalanceModel(self.reflectance_model)

    @classmethod
    def from_scope_assets(
        cls,
        *,
        lidf: torch.Tensor,
        sail: FourSAILModel | None = None,
        fluspect_path: str | None = None,
        soil_path: str | None = None,
        scope_root_path: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        ndub: int = 15,
        doublings_step: int = 5,
        default_hotspot: float = 0.2,
        soil_index_base: int = 1,
        soil_empirical: SoilEmpiricalParams | None = None,
    ) -> ScopeGridRunner:
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
        soil_bsm = SoilBSMModel.from_resources(
            resources, empirical=soil_empirical, device=fluspect.device, dtype=fluspect.dtype
        )
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
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, list[torch.Tensor]] = {name: [] for name in CanopyReflectanceResult.__dataclass_fields__}
        for batch in _progress(data_module.iter_batches(), desc="reflectance"):
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
                nlayers=self._layer_count(nlayers, etau=None, etah=None, Tcu=None, Tch=None),
            )
            for name in outputs:
                outputs[name].append(getattr(result, name))

        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}

    def run_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> xr.Dataset:
        outputs = self.run(
            data_module,
            varmap=varmap,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
        )
        return self._outputs_to_dataset(data_module, outputs, product="reflectance")

    def run_directional_reflectance(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        directional_tto: torch.Tensor | None = None,
        directional_psi: torch.Tensor | None = None,
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, list[torch.Tensor]] = {name: [] for name in ("refl_", "rso_")}
        tto_angles, psi_angles = self._directional_angles(
            data_module,
            varmap=varmap,
            directional_tto=directional_tto,
            directional_psi=directional_psi,
        )
        for batch in _progress(data_module.iter_batches(), desc="directional-reflectance"):
            leaf_kwargs = self._leafbio_kwargs(batch, varmap)
            leafbio = LeafBioBatch(**leaf_kwargs)
            lai = batch[varmap["LAI"]]
            tts = batch[varmap["tts"]]
            soil = self._soil_refl(batch, varmap)
            Esun = self._optical_directional_input(batch, varmap, "Esun")
            Esky = self._optical_directional_input(batch, varmap, "Esky")
            if hotspot_var and hotspot_var in batch:
                hotspot = batch[hotspot_var]
            else:
                hotspot = torch.full_like(lai, self.default_hotspot)

            result = self.reflectance_model.directional(
                leafbio,
                soil,
                lai,
                tts,
                tto_angles,
                psi_angles,
                Esun,
                Esky,
                hotspot=hotspot,
                nlayers=self._layer_count(nlayers, etau=None, etah=None, Tcu=None, Tch=None),
            )
            outputs["refl_"].append(result.refl_)
            outputs["rso_"].append(result.rso_)

        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}

    def run_directional_reflectance_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        directional_tto: torch.Tensor | None = None,
        directional_psi: torch.Tensor | None = None,
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> xr.Dataset:
        tto_angles, psi_angles = self._directional_angles(
            data_module,
            varmap=varmap,
            directional_tto=directional_tto,
            directional_psi=directional_psi,
        )
        outputs = self.run_directional_reflectance(
            data_module,
            varmap=varmap,
            directional_tto=tto_angles,
            directional_psi=psi_angles,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
        )
        return self._directional_outputs_to_dataset(
            data_module,
            outputs,
            product="directional_reflectance",
            directional_tto=tto_angles,
            directional_psi=psi_angles,
            variable_dims={"refl_": ("direction", "wavelength"), "rso_": ("direction", "wavelength")},
        )

    def run_reflectance_profiles(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, list[torch.Tensor]] = {
            name: [] for name in CanopyRadiationProfileResult.__dataclass_fields__
        }
        expected_layer_count: int | None = None
        for batch in _progress(data_module.iter_batches(), desc="reflectance-profiles"):
            leaf_kwargs = self._leafbio_kwargs(batch, varmap)
            leafbio = LeafBioBatch(**leaf_kwargs)
            lai = batch[varmap["LAI"]]
            tts = batch[varmap["tts"]]
            tto = batch[varmap["tto"]]
            psi = batch[varmap["psi"]]
            soil = self._soil_refl(batch, varmap)
            Esun = self._optical_directional_input(batch, varmap, "Esun")
            Esky = self._optical_directional_input(batch, varmap, "Esky")
            if hotspot_var and hotspot_var in batch:
                hotspot = batch[hotspot_var]
            else:
                hotspot = torch.full_like(lai, self.default_hotspot)

            result = self.reflectance_model.profiles(
                leafbio,
                soil,
                lai,
                tts,
                tto,
                psi,
                Esun,
                Esky,
                hotspot=hotspot,
                nlayers=self._layer_count(nlayers, etau=None, etah=None, Tcu=None, Tch=None),
            )
            expected_layer_count = self._accumulate_profile_layer_count(expected_layer_count, result.Ps)
            for name in outputs:
                outputs[name].append(getattr(result, name))

        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}

    def run_reflectance_profiles_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> xr.Dataset:
        outputs = self.run_reflectance_profiles(
            data_module,
            varmap=varmap,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
        )
        return self._profile_outputs_to_dataset(
            data_module,
            outputs,
            product="reflectance_profiles",
            layer_count=self._profile_layer_count_from_tensor(outputs["Ps"]),
            variable_dims={
                "Ps": ("layer_interface",),
                "Po": ("layer_interface",),
                "Pso": ("layer_interface",),
                "Es_direct_": ("layer_interface", "wavelength"),
                "Emin_direct_": ("layer_interface", "wavelength"),
                "Eplu_direct_": ("layer_interface", "wavelength"),
                "Es_diffuse_": ("layer_interface", "wavelength"),
                "Emin_diffuse_": ("layer_interface", "wavelength"),
                "Eplu_diffuse_": ("layer_interface", "wavelength"),
                "Es_": ("layer_interface", "wavelength"),
                "Emin_": ("layer_interface", "wavelength"),
                "Eplu_": ("layer_interface", "wavelength"),
            },
        )

    def run_biochemical_fluorescence(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        biochem_options: BiochemicalOptions | None = None,
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> dict[str, torch.Tensor]:
        physiology_fields = [name for name in LeafBiochemistryResult.__dataclass_fields__ if name != "fcount"]
        outputs: dict[str, list[torch.Tensor]] = {
            **{name: [] for name in CanopyFluorescenceResult.__dataclass_fields__},
            "Pnu_Cab": [],
            "Pnh_Cab": [],
            **{f"sunlit_{name}": [] for name in physiology_fields},
            **{f"shaded_{name}": [] for name in physiology_fields},
        }
        for batch in _progress(data_module.iter_batches(), desc="biochemical-fluorescence"):
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
                nlayers=self._layer_count(
                    nlayers, etau=None, etah=None, Tcu=batch[varmap["Tcu"]], Tch=batch[varmap["Tch"]]
                ),
            )
            for name in CanopyFluorescenceResult.__dataclass_fields__:
                outputs[name].append(getattr(result.fluorescence, name))
            outputs["Pnu_Cab"].append(result.Pnu_Cab)
            outputs["Pnh_Cab"].append(result.Pnh_Cab)
            for name in physiology_fields:
                outputs[f"sunlit_{name}"].append(getattr(result.sunlit, name))
                outputs[f"shaded_{name}"].append(getattr(result.shaded, name))

        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}

    def run_biochemical_fluorescence_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        biochem_options: BiochemicalOptions | None = None,
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> xr.Dataset:
        outputs = self.run_biochemical_fluorescence(
            data_module,
            varmap=varmap,
            biochem_options=biochem_options,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
        )
        return self._outputs_to_dataset(data_module, outputs, product="biochemical_fluorescence")

    def _energy_result_fields(self) -> tuple[list[str], list[str]]:
        physiology_fields = [name for name in LeafBiochemistryResult.__dataclass_fields__ if name != "fcount"]
        energy_fields = [
            name for name in CanopyEnergyBalanceResult.__dataclass_fields__ if name not in {"sunlit", "shaded", "Tsold"}
        ]
        return physiology_fields, energy_fields

    def _build_energy_batch_inputs(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        varmap: Mapping[str, str],
        hotspot_var: str | None,
        nlayers: int | None,
        soil_heat_method: int,
    ) -> _EnergyBatchInputs:
        get_optional = lambda key, default=None: self._optional_batch_input(batch, varmap, key, default)
        leafbio = LeafBioBatch(**self._leafbio_kwargs(batch, varmap))
        biochem = LeafBiochemistryInputs(**self._biochemistry_kwargs(batch, varmap))
        lai = batch[varmap["LAI"]]
        tts = batch[varmap["tts"]]
        tto = batch[varmap["tto"]]
        psi = batch[varmap["psi"]]
        Esun_sw = self._spectral_input(batch, varmap, "Esun_sw")
        Esky_sw = self._spectral_input(batch, varmap, "Esky_sw")
        Esun_lw = self._optional_spectral_input(batch, varmap, "Esun_lw")
        Esky_lw = self._optional_spectral_input(batch, varmap, "Esky_lw")
        soil_refl = self._soil_refl(batch, varmap)
        hotspot = (
            batch[hotspot_var] if hotspot_var and hotspot_var in batch else torch.full_like(lai, self.default_hotspot)
        )

        meteo = EnergyBalanceMeteo(
            Ta=batch[varmap["Ta"]],
            ea=batch[varmap["ea"]],
            Ca=batch[varmap["Ca"]],
            Oa=batch[varmap["Oa"]],
            p=batch[varmap["p"]],
            z=batch[varmap["z"]],
            u=batch[varmap["u"]],
            L=get_optional("L", -1e6),
        )
        canopy = EnergyBalanceCanopy(
            Cd=batch[varmap["Cd"]],
            rwc=batch[varmap["rwc"]],
            z0m=batch[varmap["z0m"]],
            d=batch[varmap["d"]],
            h=batch[varmap["h"]],
            kV=get_optional("kV", 0.0),
            fV=get_optional("fV"),
        )
        soil = EnergyBalanceSoil(
            rss=batch[varmap["rss"]],
            rbs=batch[varmap["rbs"]],
            thermal_optics=ThermalOptics(
                rho_thermal=get_optional("rho_thermal", 0.01),
                tau_thermal=get_optional("tau_thermal", 0.01),
                rs_thermal=get_optional("rs_thermal", 0.06),
            ),
            soil_heat_method=soil_heat_method,
            GAM=get_optional("GAM", 0.0),
            Tsold=get_optional("Tsold"),
            dt_seconds=get_optional("dt_seconds"),
        )
        Tcu0 = get_optional("Tcu0")
        Tch0 = get_optional("Tch0")
        Tsu0 = get_optional("Tsu0")
        Tsh0 = get_optional("Tsh0")
        resolved_nlayers = self._layer_count(
            nlayers,
            etau=canopy.fV if isinstance(canopy.fV, torch.Tensor) else None,
            etah=None,
            Tcu=Tcu0,
            Tch=Tch0,
        )
        return _EnergyBatchInputs(
            leafbio=leafbio,
            biochem=biochem,
            lai=lai,
            tts=tts,
            tto=tto,
            psi=psi,
            Esun_sw=Esun_sw,
            Esky_sw=Esky_sw,
            Esun_lw=Esun_lw,
            Esky_lw=Esky_lw,
            soil_refl=soil_refl,
            hotspot=hotspot,
            meteo=meteo,
            canopy=canopy,
            soil=soil,
            nlayers=resolved_nlayers,
            Tcu0=Tcu0,
            Tch0=Tch0,
            Tsu0=Tsu0,
            Tsh0=Tsh0,
        )

    def _solve_energy_batch(
        self,
        inputs: _EnergyBatchInputs,
        *,
        energy_options: EnergyBalanceOptions | None,
        biochem_options: BiochemicalOptions | None,
    ) -> CanopyEnergyBalanceResult:
        return self.energy_balance_model.solve(
            inputs.leafbio,
            inputs.biochem,
            inputs.soil_refl,
            inputs.lai,
            inputs.tts,
            inputs.tto,
            inputs.psi,
            inputs.Esun_sw,
            inputs.Esky_sw,
            Esun_lw=inputs.Esun_lw,
            Esky_lw=inputs.Esky_lw,
            meteo=inputs.meteo,
            canopy=inputs.canopy,
            soil=inputs.soil,
            options=energy_options,
            biochem_options=biochem_options,
            hotspot=inputs.hotspot,
            nlayers=inputs.nlayers,
            Tcu0=inputs.Tcu0,
            Tch0=inputs.Tch0,
            Tsu0=inputs.Tsu0,
            Tsh0=inputs.Tsh0,
        )

    def _append_energy_outputs(
        self,
        outputs: dict[str, list[torch.Tensor]],
        result: CanopyEnergyBalanceResult,
        *,
        energy_fields: Sequence[str],
        physiology_fields: Sequence[str],
    ) -> None:
        for name in energy_fields:
            outputs[name].append(getattr(result, name))
        for name in physiology_fields:
            outputs[f"sunlit_{name}"].append(getattr(result.sunlit, name))
            outputs[f"shaded_{name}"].append(getattr(result.shaded, name))

    def _energy_excitation_spectra(self, inputs: _EnergyBatchInputs) -> tuple[torch.Tensor, torch.Tensor]:
        wlP = self.fluspect.spectral.wlP
        wlE = self.fluspect.spectral.wlE
        if wlE is None:
            raise ValueError("Spectral grids must define excitation wavelengths")
        batch = inputs.lai.shape[0]
        Esun = self.fluorescence_model._prepare_spectrum(inputs.Esun_sw, batch, wlP.numel())
        Esky = self.fluorescence_model._prepare_spectrum(inputs.Esky_sw, batch, wlP.numel())
        return (
            self.fluorescence_model._sample_spectrum(Esun, wlP, wlE),
            self.fluorescence_model._sample_spectrum(Esky, wlP, wlE),
        )

    def run_energy_balance(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        biochem_options: BiochemicalOptions | None = None,
        energy_options: EnergyBalanceOptions | None = None,
        hotspot_var: str | None = None,
        nlayers: int | None = None,
        soil_heat_method: int = 2,
    ) -> dict[str, torch.Tensor]:
        physiology_fields, energy_fields = self._energy_result_fields()
        outputs: dict[str, list[torch.Tensor]] = {
            **{name: [] for name in energy_fields},
            **{f"sunlit_{name}": [] for name in physiology_fields},
            **{f"shaded_{name}": [] for name in physiology_fields},
        }
        for batch in data_module.iter_batches():
            inputs = self._build_energy_batch_inputs(
                batch,
                varmap=varmap,
                hotspot_var=hotspot_var,
                nlayers=nlayers,
                soil_heat_method=soil_heat_method,
            )
            result = self._solve_energy_batch(
                inputs,
                energy_options=energy_options,
                biochem_options=biochem_options,
            )
            self._append_energy_outputs(
                outputs,
                result,
                energy_fields=energy_fields,
                physiology_fields=physiology_fields,
            )
        return self._concat_outputs(outputs)

    def run_energy_balance_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        biochem_options: BiochemicalOptions | None = None,
        energy_options: EnergyBalanceOptions | None = None,
        hotspot_var: str | None = None,
        nlayers: int | None = None,
        soil_heat_method: int = 2,
    ) -> xr.Dataset:
        outputs = self.run_energy_balance(
            data_module,
            varmap=varmap,
            biochem_options=biochem_options,
            energy_options=energy_options,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
            soil_heat_method=soil_heat_method,
        )
        return self._outputs_to_dataset(data_module, outputs, product="energy_balance")

    def run_energy_balance_fluorescence(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        biochem_options: BiochemicalOptions | None = None,
        energy_options: EnergyBalanceOptions | None = None,
        hotspot_var: str | None = None,
        nlayers: int | None = None,
        soil_heat_method: int = 2,
    ) -> dict[str, torch.Tensor]:
        physiology_fields, energy_fields = self._energy_result_fields()
        outputs: dict[str, list[torch.Tensor]] = {
            **{name: [] for name in CanopyFluorescenceResult.__dataclass_fields__},
            **{name: [] for name in energy_fields},
            **{f"sunlit_{name}": [] for name in physiology_fields},
            **{f"shaded_{name}": [] for name in physiology_fields},
        }
        for batch in _progress(data_module.iter_batches(), desc="energy-balance-fluorescence"):
            inputs = self._build_energy_batch_inputs(
                batch,
                varmap=varmap,
                hotspot_var=hotspot_var,
                nlayers=nlayers,
                soil_heat_method=soil_heat_method,
            )
            result = self.energy_balance_model.solve_fluorescence(
                inputs.leafbio,
                inputs.biochem,
                inputs.soil_refl,
                inputs.lai,
                inputs.tts,
                inputs.tto,
                inputs.psi,
                inputs.Esun_sw,
                inputs.Esky_sw,
                Esun_lw=inputs.Esun_lw,
                Esky_lw=inputs.Esky_lw,
                meteo=inputs.meteo,
                canopy=inputs.canopy,
                soil=inputs.soil,
                options=energy_options,
                biochem_options=biochem_options,
                hotspot=inputs.hotspot,
                nlayers=inputs.nlayers,
                Tcu0=inputs.Tcu0,
                Tch0=inputs.Tch0,
                Tsu0=inputs.Tsu0,
                Tsh0=inputs.Tsh0,
            )
            for name in CanopyFluorescenceResult.__dataclass_fields__:
                outputs[name].append(getattr(result.fluorescence, name))
            self._append_energy_outputs(
                outputs,
                result.energy,
                energy_fields=energy_fields,
                physiology_fields=physiology_fields,
            )

        return self._concat_outputs(outputs)

    def run_energy_balance_fluorescence_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        biochem_options: BiochemicalOptions | None = None,
        energy_options: EnergyBalanceOptions | None = None,
        hotspot_var: str | None = None,
        nlayers: int | None = None,
        soil_heat_method: int = 2,
    ) -> xr.Dataset:
        outputs = self.run_energy_balance_fluorescence(
            data_module,
            varmap=varmap,
            biochem_options=biochem_options,
            energy_options=energy_options,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
            soil_heat_method=soil_heat_method,
        )
        return self._outputs_to_dataset(data_module, outputs, product="energy_balance_fluorescence")

    def run_energy_balance_thermal(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        biochem_options: BiochemicalOptions | None = None,
        energy_options: EnergyBalanceOptions | None = None,
        hotspot_var: str | None = None,
        nlayers: int | None = None,
        soil_heat_method: int = 2,
    ) -> dict[str, torch.Tensor]:
        physiology_fields, energy_fields = self._energy_result_fields()
        outputs: dict[str, list[torch.Tensor]] = {
            **{name: [] for name in CanopyThermalRadianceResult.__dataclass_fields__},
            **{name: [] for name in energy_fields},
            **{f"sunlit_{name}": [] for name in physiology_fields},
            **{f"shaded_{name}": [] for name in physiology_fields},
        }
        for batch in _progress(data_module.iter_batches(), desc="energy-balance-thermal"):
            inputs = self._build_energy_batch_inputs(
                batch,
                varmap=varmap,
                hotspot_var=hotspot_var,
                nlayers=nlayers,
                soil_heat_method=soil_heat_method,
            )
            result = self.energy_balance_model.solve_thermal(
                inputs.leafbio,
                inputs.biochem,
                inputs.soil_refl,
                inputs.lai,
                inputs.tts,
                inputs.tto,
                inputs.psi,
                inputs.Esun_sw,
                inputs.Esky_sw,
                Esun_lw=inputs.Esun_lw,
                Esky_lw=inputs.Esky_lw,
                meteo=inputs.meteo,
                canopy=inputs.canopy,
                soil=inputs.soil,
                options=energy_options,
                biochem_options=biochem_options,
                hotspot=inputs.hotspot,
                nlayers=inputs.nlayers,
                Tcu0=inputs.Tcu0,
                Tch0=inputs.Tch0,
                Tsu0=inputs.Tsu0,
                Tsh0=inputs.Tsh0,
            )
            for name in CanopyThermalRadianceResult.__dataclass_fields__:
                outputs[name].append(getattr(result.thermal, name))
            self._append_energy_outputs(
                outputs,
                result.energy,
                energy_fields=energy_fields,
                physiology_fields=physiology_fields,
            )

        return self._concat_outputs(outputs)

    def run_energy_balance_thermal_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        biochem_options: BiochemicalOptions | None = None,
        energy_options: EnergyBalanceOptions | None = None,
        hotspot_var: str | None = None,
        nlayers: int | None = None,
        soil_heat_method: int = 2,
    ) -> xr.Dataset:
        outputs = self.run_energy_balance_thermal(
            data_module,
            varmap=varmap,
            biochem_options=biochem_options,
            energy_options=energy_options,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
            soil_heat_method=soil_heat_method,
        )
        return self._outputs_to_dataset(data_module, outputs, product="energy_balance_thermal")

    def run_fluorescence(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: str | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, list[torch.Tensor]] = {name: [] for name in CanopyFluorescenceResult.__dataclass_fields__}
        for batch in _progress(data_module.iter_batches(), desc="fluorescence"):
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

    def run_fluorescence_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: str | None = None,
    ) -> xr.Dataset:
        outputs = self.run_fluorescence(
            data_module,
            varmap=varmap,
            hotspot_var=hotspot_var,
        )
        return self._outputs_to_dataset(data_module, outputs, product="fluorescence")

    def run_directional_fluorescence(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        directional_tto: torch.Tensor | None = None,
        directional_psi: torch.Tensor | None = None,
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, list[torch.Tensor]] = {name: [] for name in ("LoF_",)}
        tto_angles, psi_angles = self._directional_angles(
            data_module,
            varmap=varmap,
            directional_tto=directional_tto,
            directional_psi=directional_psi,
        )
        for batch in _progress(data_module.iter_batches(), desc="directional-fluorescence"):
            leaf_kwargs = self._leafbio_kwargs(batch, varmap)
            leafbio = LeafBioBatch(**leaf_kwargs)
            lai = batch[varmap["LAI"]]
            tts = batch[varmap["tts"]]
            soil = self._soil_refl(batch, varmap)
            Esun = self._spectral_input(batch, varmap, "Esun_")
            Esky = self._spectral_input(batch, varmap, "Esky_")
            etau = batch[varmap["etau"]] if "etau" in varmap and varmap["etau"] in batch else None
            etah = batch[varmap["etah"]] if "etah" in varmap and varmap["etah"] in batch else None
            if hotspot_var and hotspot_var in batch:
                hotspot = batch[hotspot_var]
            else:
                hotspot = torch.full_like(lai, self.default_hotspot)

            result = self.fluorescence_model.directional(
                leafbio,
                soil,
                lai,
                tts,
                tto_angles,
                psi_angles,
                Esun,
                Esky,
                etau=etau,
                etah=etah,
                hotspot=hotspot,
                nlayers=self._layer_count(nlayers, etau=etau, etah=etah, Tcu=None, Tch=None),
            )
            outputs["LoF_"].append(result.LoF_)

        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}

    def run_directional_fluorescence_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        directional_tto: torch.Tensor | None = None,
        directional_psi: torch.Tensor | None = None,
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> xr.Dataset:
        tto_angles, psi_angles = self._directional_angles(
            data_module,
            varmap=varmap,
            directional_tto=directional_tto,
            directional_psi=directional_psi,
        )
        outputs = self.run_directional_fluorescence(
            data_module,
            varmap=varmap,
            directional_tto=tto_angles,
            directional_psi=psi_angles,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
        )
        return self._directional_outputs_to_dataset(
            data_module,
            outputs,
            product="directional_fluorescence",
            directional_tto=tto_angles,
            directional_psi=psi_angles,
            variable_dims={"LoF_": ("direction", "fluorescence_wavelength")},
        )

    def run_fluorescence_profiles(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, list[torch.Tensor]] = {
            name: [] for name in CanopyFluorescenceProfileResult.__dataclass_fields__
        }
        expected_layer_count: int | None = None
        for batch in _progress(data_module.iter_batches(), desc="fluorescence-profiles"):
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

            result = self.fluorescence_model.profiles(
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
            expected_layer_count = self._accumulate_profile_layer_count(expected_layer_count, result.Ps)
            for name in outputs:
                outputs[name].append(getattr(result, name))

        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}

    def run_fluorescence_profiles_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> xr.Dataset:
        outputs = self.run_fluorescence_profiles(
            data_module,
            varmap=varmap,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
        )
        return self._profile_outputs_to_dataset(
            data_module,
            outputs,
            product="fluorescence_profiles",
            layer_count=self._profile_layer_count_from_tensor(outputs["Ps"]),
            variable_dims={
                "Ps": ("layer_interface",),
                "Po": ("layer_interface",),
                "Pso": ("layer_interface",),
                "Fmin_": ("layer_interface", "fluorescence_wavelength"),
                "Fplu_": ("layer_interface", "fluorescence_wavelength"),
                "layer_fluorescence": ("layer",),
            },
        )

    def run_layered_fluorescence(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, list[torch.Tensor]] = {name: [] for name in CanopyFluorescenceResult.__dataclass_fields__}
        for batch in _progress(data_module.iter_batches(), desc="layered-fluorescence"):
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

    def run_layered_fluorescence_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> xr.Dataset:
        outputs = self.run_layered_fluorescence(
            data_module,
            varmap=varmap,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
        )
        return self._outputs_to_dataset(data_module, outputs, product="layered_fluorescence")

    def run_thermal(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, list[torch.Tensor]] = {name: [] for name in CanopyThermalRadianceResult.__dataclass_fields__}
        for batch in _progress(data_module.iter_batches(), desc="thermal"):
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
                rho_thermal=batch[varmap["rho_thermal"]]
                if "rho_thermal" in varmap and varmap["rho_thermal"] in batch
                else 0.01,
                tau_thermal=batch[varmap["tau_thermal"]]
                if "tau_thermal" in varmap and varmap["tau_thermal"] in batch
                else 0.01,
                rs_thermal=batch[varmap["rs_thermal"]]
                if "rs_thermal" in varmap and varmap["rs_thermal"] in batch
                else 0.06,
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

    def run_thermal_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> xr.Dataset:
        outputs = self.run_thermal(
            data_module,
            varmap=varmap,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
        )
        return self._outputs_to_dataset(data_module, outputs, product="thermal")

    def run_directional_thermal(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        directional_tto: torch.Tensor | None = None,
        directional_psi: torch.Tensor | None = None,
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, list[torch.Tensor]] = {name: [] for name in ("Lot_", "BrightnessT")}
        tto_angles, psi_angles = self._directional_angles(
            data_module,
            varmap=varmap,
            directional_tto=directional_tto,
            directional_psi=directional_psi,
        )
        for batch in _progress(data_module.iter_batches(), desc="directional-thermal"):
            lai = batch[varmap["LAI"]]
            tts = batch[varmap["tts"]]
            Tcu = batch[varmap["Tcu"]]
            Tch = batch[varmap["Tch"]]
            Tsu = batch[varmap["Tsu"]]
            Tsh = batch[varmap["Tsh"]]
            if hotspot_var and hotspot_var in batch:
                hotspot = batch[hotspot_var]
            else:
                hotspot = torch.full_like(lai, self.default_hotspot)
            thermal_optics = ThermalOptics(
                rho_thermal=batch[varmap["rho_thermal"]]
                if "rho_thermal" in varmap and varmap["rho_thermal"] in batch
                else 0.01,
                tau_thermal=batch[varmap["tau_thermal"]]
                if "tau_thermal" in varmap and varmap["tau_thermal"] in batch
                else 0.01,
                rs_thermal=batch[varmap["rs_thermal"]]
                if "rs_thermal" in varmap and varmap["rs_thermal"] in batch
                else 0.06,
            )

            result = self.thermal_model.directional(
                lai,
                tts,
                tto_angles,
                psi_angles,
                Tcu,
                Tch,
                Tsu,
                Tsh,
                thermal_optics=thermal_optics,
                hotspot=hotspot,
                nlayers=self._layer_count(nlayers, etau=None, etah=None, Tcu=Tcu, Tch=Tch),
            )
            outputs["Lot_"].append(result.Lot_)
            outputs["BrightnessT"].append(result.BrightnessT)

        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}

    def run_directional_thermal_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        directional_tto: torch.Tensor | None = None,
        directional_psi: torch.Tensor | None = None,
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> xr.Dataset:
        tto_angles, psi_angles = self._directional_angles(
            data_module,
            varmap=varmap,
            directional_tto=directional_tto,
            directional_psi=directional_psi,
        )
        outputs = self.run_directional_thermal(
            data_module,
            varmap=varmap,
            directional_tto=tto_angles,
            directional_psi=psi_angles,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
        )
        return self._directional_outputs_to_dataset(
            data_module,
            outputs,
            product="directional_thermal",
            directional_tto=tto_angles,
            directional_psi=psi_angles,
            variable_dims={"Lot_": ("direction", "thermal_wavelength"), "BrightnessT": ("direction",)},
        )

    def run_thermal_profiles(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, list[torch.Tensor]] = {name: [] for name in CanopyThermalProfileResult.__dataclass_fields__}
        expected_layer_count: int | None = None
        for batch in _progress(data_module.iter_batches(), desc="thermal-profiles"):
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
                rho_thermal=batch[varmap["rho_thermal"]]
                if "rho_thermal" in varmap and varmap["rho_thermal"] in batch
                else 0.01,
                tau_thermal=batch[varmap["tau_thermal"]]
                if "tau_thermal" in varmap and varmap["tau_thermal"] in batch
                else 0.01,
                rs_thermal=batch[varmap["rs_thermal"]]
                if "rs_thermal" in varmap and varmap["rs_thermal"] in batch
                else 0.06,
            )

            result = self.thermal_model.profiles(
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
            expected_layer_count = self._accumulate_profile_layer_count(expected_layer_count, result.Ps)
            for name in outputs:
                outputs[name].append(getattr(result, name))

        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}

    def run_thermal_profiles_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> xr.Dataset:
        outputs = self.run_thermal_profiles(
            data_module,
            varmap=varmap,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
        )
        return self._profile_outputs_to_dataset(
            data_module,
            outputs,
            product="thermal_profiles",
            layer_count=self._profile_layer_count_from_tensor(outputs["Ps"]),
            variable_dims={
                "Ps": ("layer_interface",),
                "Po": ("layer_interface",),
                "Pso": ("layer_interface",),
                "Emint_": ("layer_interface", "thermal_wavelength"),
                "Eplut_": ("layer_interface", "thermal_wavelength"),
                "layer_thermal_upward": ("layer",),
            },
        )

    def run_scope_tensors(
        self,
        inputs: Mapping[str, torch.Tensor | float | int],
        *,
        varmap: Mapping[str, str] | None = None,
        workflow: str | None = None,
        scope_options: Mapping[str, object] | None = None,
        energy_options: EnergyBalanceOptions | None = None,
        biochem_options: BiochemicalOptions | None = None,
        directional_tto: torch.Tensor | None = None,
        directional_psi: torch.Tensor | None = None,
        hotspot_var: str | None = None,
        nlayers: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a SCOPE workflow directly on tensors without xarray assembly."""

        tensor_inputs = self._normalise_tensor_inputs(inputs)
        resolved_varmap = dict(varmap or {name: name for name in tensor_inputs})
        if directional_tto is None:
            directional_name = resolved_varmap.get("directional_tto")
            if directional_name in tensor_inputs:
                directional_tto = tensor_inputs[directional_name]
        if directional_psi is None:
            directional_name = resolved_varmap.get("directional_psi")
            if directional_name in tensor_inputs:
                directional_psi = tensor_inputs[directional_name]

        data_module = _TensorBatchDataModule(tensor_inputs, attrs=scope_options)
        return self._run_named_tensor_workflow(
            data_module,
            workflow="scope" if workflow is None else workflow,
            varmap=resolved_varmap,
            scope_options=scope_options,
            energy_options=energy_options,
            biochem_options=biochem_options,
            directional_tto=directional_tto,
            directional_psi=directional_psi,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
        )

    def run_scope_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        scope_options: Mapping[str, object] | None = None,
        energy_options: EnergyBalanceOptions | None = None,
        biochem_options: BiochemicalOptions | None = None,
        directional_tto: torch.Tensor | None = None,
        directional_psi: torch.Tensor | None = None,
        hotspot_var: str | None = None,
        nlayers: int | None = None,
        return_tensors: bool = False,
        detach: bool = False,
    ) -> xr.Dataset | dict[str, torch.Tensor]:
        if return_tensors:
            outputs = self._run_scope_tensor_workflow(
                data_module,
                varmap=varmap,
                scope_options=scope_options,
                energy_options=energy_options,
                biochem_options=biochem_options,
                directional_tto=directional_tto,
                directional_psi=directional_psi,
                hotspot_var=hotspot_var,
                nlayers=nlayers,
            )
            return self._detach_tensor_outputs(outputs) if detach else outputs

        calc_directional = self._scope_option_flag(data_module, scope_options, "calc_directional")
        calc_vert_profiles = self._scope_option_flag(data_module, scope_options, "calc_vert_profiles")
        calc_fluor = self._scope_option_flag(data_module, scope_options, "calc_fluor")
        calc_planck = self._scope_option_flag(data_module, scope_options, "calc_planck")
        calc_ebal = self._scope_option_flag(data_module, scope_options, "calc_ebal")
        reflectance_varmap = self._workflow_reflectance_varmap(varmap)

        datasets = [
            self.run_dataset(
                data_module,
                varmap=varmap,
                hotspot_var=hotspot_var,
                nlayers=nlayers,
            )
        ]
        components = ["reflectance"]

        if calc_directional:
            datasets.append(
                self._prefixed_dataset(
                    self.run_directional_reflectance_dataset(
                        data_module,
                        varmap=reflectance_varmap,
                        directional_tto=directional_tto,
                        directional_psi=directional_psi,
                        hotspot_var=hotspot_var,
                        nlayers=nlayers,
                    ),
                    "reflectance_directional",
                )
            )
            components.append("reflectance_directional")

        if calc_vert_profiles:
            datasets.append(
                self._prefixed_dataset(
                    self.run_reflectance_profiles_dataset(
                        data_module,
                        varmap=reflectance_varmap,
                        hotspot_var=hotspot_var,
                        nlayers=nlayers,
                    ),
                    "reflectance_profile",
                )
            )
            components.append("reflectance_profile")

        if calc_ebal:
            coupled_datasets, coupled_components = self._run_coupled_scope_datasets(
                data_module,
                varmap=varmap,
                calc_fluor=calc_fluor,
                calc_planck=calc_planck,
                calc_directional=calc_directional,
                calc_vert_profiles=calc_vert_profiles,
                directional_tto=directional_tto,
                directional_psi=directional_psi,
                hotspot_var=hotspot_var,
                nlayers=nlayers,
                energy_options=self._scope_energy_options(data_module, scope_options, energy_options),
                biochem_options=biochem_options,
                soil_heat_method=self._scope_option_int(data_module, scope_options, "soil_heat_method", default=2),
            )
            datasets.extend(coupled_datasets)
            components.extend(coupled_components)
        else:
            if calc_fluor:
                datasets.append(
                    self.run_layered_fluorescence_dataset(
                        data_module,
                        varmap=varmap,
                        hotspot_var=hotspot_var,
                        nlayers=nlayers,
                    )
                )
                components.append("fluorescence")

                if calc_directional:
                    datasets.append(
                        self._prefixed_dataset(
                            self.run_directional_fluorescence_dataset(
                                data_module,
                                varmap=varmap,
                                directional_tto=directional_tto,
                                directional_psi=directional_psi,
                                hotspot_var=hotspot_var,
                                nlayers=nlayers,
                            ),
                            "fluorescence_directional",
                        )
                    )
                    components.append("fluorescence_directional")

                if calc_vert_profiles:
                    datasets.append(
                        self._prefixed_dataset(
                            self.run_fluorescence_profiles_dataset(
                                data_module,
                                varmap=varmap,
                                hotspot_var=hotspot_var,
                                nlayers=nlayers,
                            ),
                            "fluorescence_profile",
                        )
                    )
                    components.append("fluorescence_profile")

            if calc_planck:
                datasets.append(
                    self.run_thermal_dataset(
                        data_module,
                        varmap=varmap,
                        hotspot_var=hotspot_var,
                        nlayers=nlayers,
                    )
                )
                components.append("thermal")

                if calc_directional:
                    datasets.append(
                        self._prefixed_dataset(
                            self.run_directional_thermal_dataset(
                                data_module,
                                varmap=varmap,
                                directional_tto=directional_tto,
                                directional_psi=directional_psi,
                                hotspot_var=hotspot_var,
                                nlayers=nlayers,
                            ),
                            "thermal_directional",
                        )
                    )
                    components.append("thermal_directional")

                if calc_vert_profiles:
                    datasets.append(
                        self._prefixed_dataset(
                            self.run_thermal_profiles_dataset(
                                data_module,
                                varmap=varmap,
                                hotspot_var=hotspot_var,
                                nlayers=nlayers,
                            ),
                            "thermal_profile",
                        )
                    )
                    components.append("thermal_profile")

        return self._merge_workflow_datasets(
            data_module,
            datasets,
            product="scope_workflow",
            components=components,
            scope_options=scope_options,
        )

    def _run_named_tensor_workflow(
        self,
        data_module,
        *,
        workflow: str,
        varmap: Mapping[str, str],
        scope_options: Mapping[str, object] | None,
        energy_options: EnergyBalanceOptions | None,
        biochem_options: BiochemicalOptions | None,
        directional_tto: torch.Tensor | None,
        directional_psi: torch.Tensor | None,
        hotspot_var: str | None,
        nlayers: int | None,
    ) -> dict[str, torch.Tensor]:
        workflow_key = workflow.strip().lower().replace("-", "_")
        soil_heat_method = self._scope_option_int(data_module, scope_options, "soil_heat_method", default=2)
        resolved_energy_options = self._scope_energy_options(data_module, scope_options, energy_options)
        if workflow_key in {"scope", "scope_workflow"}:
            return self._run_scope_tensor_workflow(
                data_module,
                varmap=varmap,
                scope_options=scope_options,
                energy_options=resolved_energy_options,
                biochem_options=biochem_options,
                directional_tto=directional_tto,
                directional_psi=directional_psi,
                hotspot_var=hotspot_var,
                nlayers=nlayers,
            )
        if workflow_key == "reflectance":
            return self.run(data_module, varmap=varmap, hotspot_var=hotspot_var, nlayers=nlayers)
        if workflow_key == "directional_reflectance":
            return self.run_directional_reflectance(
                data_module,
                varmap=varmap,
                directional_tto=directional_tto,
                directional_psi=directional_psi,
                hotspot_var=hotspot_var,
                nlayers=nlayers,
            )
        if workflow_key == "reflectance_profiles":
            return self.run_reflectance_profiles(data_module, varmap=varmap, hotspot_var=hotspot_var, nlayers=nlayers)
        if workflow_key == "fluorescence":
            return self.run_fluorescence(data_module, varmap=varmap, hotspot_var=hotspot_var)
        if workflow_key == "layered_fluorescence":
            return self.run_layered_fluorescence(data_module, varmap=varmap, hotspot_var=hotspot_var, nlayers=nlayers)
        if workflow_key == "directional_fluorescence":
            return self.run_directional_fluorescence(
                data_module,
                varmap=varmap,
                directional_tto=directional_tto,
                directional_psi=directional_psi,
                hotspot_var=hotspot_var,
                nlayers=nlayers,
            )
        if workflow_key == "fluorescence_profiles":
            return self.run_fluorescence_profiles(data_module, varmap=varmap, hotspot_var=hotspot_var, nlayers=nlayers)
        if workflow_key == "thermal":
            return self.run_thermal(data_module, varmap=varmap, hotspot_var=hotspot_var, nlayers=nlayers)
        if workflow_key == "directional_thermal":
            return self.run_directional_thermal(
                data_module,
                varmap=varmap,
                directional_tto=directional_tto,
                directional_psi=directional_psi,
                hotspot_var=hotspot_var,
                nlayers=nlayers,
            )
        if workflow_key == "thermal_profiles":
            return self.run_thermal_profiles(data_module, varmap=varmap, hotspot_var=hotspot_var, nlayers=nlayers)
        if workflow_key == "biochemical_fluorescence":
            return self.run_biochemical_fluorescence(
                data_module,
                varmap=varmap,
                biochem_options=biochem_options,
                hotspot_var=hotspot_var,
                nlayers=nlayers,
            )
        if workflow_key == "energy_balance":
            return self.run_energy_balance(
                data_module,
                varmap=varmap,
                biochem_options=biochem_options,
                energy_options=resolved_energy_options,
                hotspot_var=hotspot_var,
                nlayers=nlayers,
                soil_heat_method=soil_heat_method,
            )
        if workflow_key == "energy_balance_fluorescence":
            return self.run_energy_balance_fluorescence(
                data_module,
                varmap=varmap,
                biochem_options=biochem_options,
                energy_options=resolved_energy_options,
                hotspot_var=hotspot_var,
                nlayers=nlayers,
                soil_heat_method=soil_heat_method,
            )
        if workflow_key == "energy_balance_thermal":
            return self.run_energy_balance_thermal(
                data_module,
                varmap=varmap,
                biochem_options=biochem_options,
                energy_options=resolved_energy_options,
                hotspot_var=hotspot_var,
                nlayers=nlayers,
                soil_heat_method=soil_heat_method,
            )
        raise ValueError(f"Unsupported SCOPE tensor workflow '{workflow}'")

    def _run_scope_tensor_workflow(
        self,
        data_module,
        *,
        varmap: Mapping[str, str],
        scope_options: Mapping[str, object] | None,
        energy_options: EnergyBalanceOptions | None,
        biochem_options: BiochemicalOptions | None,
        directional_tto: torch.Tensor | None,
        directional_psi: torch.Tensor | None,
        hotspot_var: str | None,
        nlayers: int | None,
    ) -> dict[str, torch.Tensor]:
        calc_directional = self._scope_option_flag(data_module, scope_options, "calc_directional")
        calc_vert_profiles = self._scope_option_flag(data_module, scope_options, "calc_vert_profiles")
        calc_fluor = self._scope_option_flag(data_module, scope_options, "calc_fluor")
        calc_planck = self._scope_option_flag(data_module, scope_options, "calc_planck")
        calc_ebal = self._scope_option_flag(data_module, scope_options, "calc_ebal")
        reflectance_varmap = self._workflow_reflectance_varmap(varmap)
        resolved_energy_options = self._scope_energy_options(data_module, scope_options, energy_options)

        outputs: dict[str, torch.Tensor] = {}
        self._merge_tensor_component(
            outputs,
            "reflectance",
            self.run(data_module, varmap=varmap, hotspot_var=hotspot_var, nlayers=nlayers),
        )

        if calc_directional:
            self._merge_tensor_component(
                outputs,
                "reflectance_directional",
                self.run_directional_reflectance(
                    data_module,
                    varmap=reflectance_varmap,
                    directional_tto=directional_tto,
                    directional_psi=directional_psi,
                    hotspot_var=hotspot_var,
                    nlayers=nlayers,
                ),
                prefix="reflectance_directional",
            )

        if calc_vert_profiles:
            self._merge_tensor_component(
                outputs,
                "reflectance_profile",
                self.run_reflectance_profiles(
                    data_module,
                    varmap=reflectance_varmap,
                    hotspot_var=hotspot_var,
                    nlayers=nlayers,
                ),
                prefix="reflectance_profile",
            )

        if calc_ebal:
            soil_heat_method = self._scope_option_int(data_module, scope_options, "soil_heat_method", default=2)
            self._merge_tensor_component(
                outputs,
                "energy_balance",
                self.run_energy_balance(
                    data_module,
                    varmap=varmap,
                    biochem_options=biochem_options,
                    energy_options=resolved_energy_options,
                    hotspot_var=hotspot_var,
                    nlayers=nlayers,
                    soil_heat_method=soil_heat_method,
                ),
            )
            if calc_fluor:
                fluorescence = self.run_energy_balance_fluorescence(
                    data_module,
                    varmap=varmap,
                    biochem_options=biochem_options,
                    energy_options=resolved_energy_options,
                    hotspot_var=hotspot_var,
                    nlayers=nlayers,
                    soil_heat_method=soil_heat_method,
                )
                self._merge_tensor_component(
                    outputs,
                    "energy_balance_fluorescence",
                    {name: fluorescence[name] for name in CanopyFluorescenceResult.__dataclass_fields__},
                )
            if calc_planck:
                thermal = self.run_energy_balance_thermal(
                    data_module,
                    varmap=varmap,
                    biochem_options=biochem_options,
                    energy_options=resolved_energy_options,
                    hotspot_var=hotspot_var,
                    nlayers=nlayers,
                    soil_heat_method=soil_heat_method,
                )
                self._merge_tensor_component(
                    outputs,
                    "energy_balance_thermal",
                    {name: thermal[name] for name in CanopyThermalRadianceResult.__dataclass_fields__},
                )
            return outputs

        if calc_fluor:
            self._merge_tensor_component(
                outputs,
                "fluorescence",
                self.run_layered_fluorescence(
                    data_module,
                    varmap=varmap,
                    hotspot_var=hotspot_var,
                    nlayers=nlayers,
                ),
            )
            if calc_directional:
                self._merge_tensor_component(
                    outputs,
                    "fluorescence_directional",
                    self.run_directional_fluorescence(
                        data_module,
                        varmap=varmap,
                        directional_tto=directional_tto,
                        directional_psi=directional_psi,
                        hotspot_var=hotspot_var,
                        nlayers=nlayers,
                    ),
                    prefix="fluorescence_directional",
                )
            if calc_vert_profiles:
                self._merge_tensor_component(
                    outputs,
                    "fluorescence_profile",
                    self.run_fluorescence_profiles(
                        data_module,
                        varmap=varmap,
                        hotspot_var=hotspot_var,
                        nlayers=nlayers,
                    ),
                    prefix="fluorescence_profile",
                )

        if calc_planck:
            self._merge_tensor_component(
                outputs,
                "thermal",
                self.run_thermal(data_module, varmap=varmap, hotspot_var=hotspot_var, nlayers=nlayers),
            )
            if calc_directional:
                self._merge_tensor_component(
                    outputs,
                    "thermal_directional",
                    self.run_directional_thermal(
                        data_module,
                        varmap=varmap,
                        directional_tto=directional_tto,
                        directional_psi=directional_psi,
                        hotspot_var=hotspot_var,
                        nlayers=nlayers,
                    ),
                    prefix="thermal_directional",
                )
            if calc_vert_profiles:
                self._merge_tensor_component(
                    outputs,
                    "thermal_profile",
                    self.run_thermal_profiles(data_module, varmap=varmap, hotspot_var=hotspot_var, nlayers=nlayers),
                    prefix="thermal_profile",
                )

        return outputs

    def _outputs_to_dataset(
        self,
        data_module: ScopeGridDataModule,
        outputs: Mapping[str, torch.Tensor],
        *,
        product: str,
        layer_count: int | None = None,
    ) -> xr.Dataset:
        dataset_outputs = {name: self._dataset_tensor(value) for name, value in outputs.items()}
        resolved_layer_count = layer_count
        if resolved_layer_count is None:
            resolved_layer_count = self._output_layer_count(data_module, dataset_outputs)
        variable_dims = {
            name: self._infer_output_dims(name, tensor, layer_count=resolved_layer_count)
            for name, tensor in dataset_outputs.items()
        }
        variable_coords = self._output_coords(resolved_layer_count)
        return data_module.assemble_dataset(
            dataset_outputs,
            variable_dims=variable_dims,
            variable_coords=variable_coords,
            attrs={
                "scope_product": product,
            },
        )

    def _merge_workflow_datasets(
        self,
        data_module: ScopeGridDataModule,
        datasets: Sequence[xr.Dataset],
        *,
        product: str,
        components: Sequence[str],
        scope_options: Mapping[str, object] | None,
    ) -> xr.Dataset:
        if len(datasets) != len(components):
            raise ValueError("datasets and components must have matching lengths")
        merged = datasets[0]
        for component, dataset in zip(components[1:], datasets[1:]):
            conflicting = set(dataset.data_vars).intersection(merged.data_vars)
            if conflicting:
                dataset = dataset.rename({name: f"{component}_{name}" for name in conflicting})
            merged = xr.merge([merged, dataset], compat="no_conflicts", combine_attrs="drop_conflicts")
        attrs = dict(data_module.dataset.attrs)
        if scope_options:
            attrs.update(scope_options)
        attrs["scope_product"] = product
        attrs["scope_components"] = ",".join(components)
        merged.attrs = attrs
        return annotate_dataset(merged)

    def _prefixed_dataset(self, dataset: xr.Dataset, prefix: str) -> xr.Dataset:
        rename_map = {name: f"{prefix}_{name}" for name in dataset.data_vars}
        return annotate_dataset(dataset.rename(rename_map))

    def _normalise_tensor_inputs(
        self,
        inputs: Mapping[str, torch.Tensor | float | int],
    ) -> dict[str, torch.Tensor]:
        tensors: dict[str, torch.Tensor] = {}
        for name, value in inputs.items():
            tensor = torch.as_tensor(value)
            if tensor.ndim == 0:
                tensor = tensor.reshape(1)
            tensors[name] = tensor
        return tensors

    def _merge_tensor_component(
        self,
        merged: dict[str, torch.Tensor],
        component: str,
        outputs: Mapping[str, torch.Tensor],
        *,
        prefix: str | None = None,
    ) -> None:
        for name, value in outputs.items():
            output_name = f"{prefix}_{name}" if prefix is not None else name
            if output_name in merged:
                output_name = f"{component}_{output_name}"
            merged[output_name] = value

    def _detach_tensor_outputs(self, outputs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: value.detach() for name, value in outputs.items()}

    def _run_coupled_scope_datasets(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        calc_fluor: bool,
        calc_planck: bool,
        calc_directional: bool,
        calc_vert_profiles: bool,
        directional_tto: torch.Tensor | None,
        directional_psi: torch.Tensor | None,
        hotspot_var: str | None,
        nlayers: int | None,
        energy_options: EnergyBalanceOptions | None,
        biochem_options: BiochemicalOptions | None,
        soil_heat_method: int,
    ) -> tuple[list[xr.Dataset], list[str]]:
        physiology_fields, energy_fields = self._energy_result_fields()
        coupled_layer_count: int | None = None
        energy_outputs: dict[str, list[torch.Tensor]] = {
            **{name: [] for name in energy_fields},
            **{f"sunlit_{name}": [] for name in physiology_fields},
            **{f"shaded_{name}": [] for name in physiology_fields},
        }

        def _init(names: Sequence[str]) -> dict[str, list[torch.Tensor]]:
            return {name: [] for name in names}

        fluorescence_outputs = _init(CanopyFluorescenceResult.__dataclass_fields__) if calc_fluor else None
        thermal_outputs = _init(CanopyThermalRadianceResult.__dataclass_fields__) if calc_planck else None
        fluorescence_directional_outputs = _init(("LoF_",)) if calc_fluor and calc_directional else None
        thermal_directional_outputs = _init(("Lot_", "BrightnessT")) if calc_planck and calc_directional else None
        fluorescence_profile_outputs = (
            _init(CanopyFluorescenceProfileResult.__dataclass_fields__) if calc_fluor and calc_vert_profiles else None
        )
        thermal_profile_outputs = (
            _init(CanopyThermalProfileResult.__dataclass_fields__) if calc_planck and calc_vert_profiles else None
        )
        need_directional_angles = (
            fluorescence_directional_outputs is not None or thermal_directional_outputs is not None
        )
        if need_directional_angles:
            tto_angles, psi_angles = self._directional_angles(
                data_module,
                varmap=varmap,
                directional_tto=directional_tto,
                directional_psi=directional_psi,
            )
        else:
            tto_angles = psi_angles = None

        for batch in data_module.iter_batches():
            inputs = self._build_energy_batch_inputs(
                batch,
                varmap=varmap,
                hotspot_var=hotspot_var,
                nlayers=nlayers,
                soil_heat_method=soil_heat_method,
            )
            energy = self._solve_energy_batch(
                inputs,
                energy_options=energy_options,
                biochem_options=biochem_options,
            )
            current_layer_count = self._layer_count(
                inputs.nlayers,
                etau=energy.sunlit.eta,
                etah=energy.shaded.eta,
                Tcu=energy.Tcu,
                Tch=energy.Tch,
            )
            if current_layer_count is not None:
                if coupled_layer_count is not None and coupled_layer_count != current_layer_count:
                    raise ValueError("Coupled scope workflows require a uniform layer count across all batches")
                coupled_layer_count = current_layer_count
            self._append_energy_outputs(
                energy_outputs,
                energy,
                energy_fields=energy_fields,
                physiology_fields=physiology_fields,
            )

            excitation_spectra: tuple[torch.Tensor, torch.Tensor] | None = None
            if fluorescence_directional_outputs is not None or fluorescence_profile_outputs is not None:
                excitation_spectra = self._energy_excitation_spectra(inputs)

            if fluorescence_outputs is not None:
                fluorescence = self.energy_balance_model._fluorescence_from_energy(
                    energy=energy,
                    leafbio=inputs.leafbio,
                    soil_refl=inputs.soil_refl,
                    lai=inputs.lai,
                    tts=inputs.tts,
                    tto=inputs.tto,
                    psi=inputs.psi,
                    Esun_sw=inputs.Esun_sw,
                    Esky_sw=inputs.Esky_sw,
                    hotspot=inputs.hotspot,
                    lidf=None,
                )
                for name in fluorescence_outputs:
                    fluorescence_outputs[name].append(getattr(fluorescence, name))

                if fluorescence_directional_outputs is not None:
                    assert tto_angles is not None and psi_angles is not None and excitation_spectra is not None
                    Esun_e, Esky_e = excitation_spectra
                    directional = self.fluorescence_model.directional(
                        inputs.leafbio,
                        inputs.soil_refl,
                        inputs.lai,
                        inputs.tts,
                        tto_angles,
                        psi_angles,
                        Esun_e,
                        Esky_e,
                        etau=energy.sunlit.eta,
                        etah=energy.shaded.eta,
                        hotspot=inputs.hotspot,
                        nlayers=inputs.nlayers,
                    )
                    fluorescence_directional_outputs["LoF_"].append(directional.LoF_)

                if fluorescence_profile_outputs is not None:
                    assert excitation_spectra is not None
                    Esun_e, Esky_e = excitation_spectra
                    profiles = self.fluorescence_model.profiles(
                        inputs.leafbio,
                        inputs.soil_refl,
                        inputs.lai,
                        inputs.tts,
                        inputs.tto,
                        inputs.psi,
                        Esun_e,
                        Esky_e,
                        etau=energy.sunlit.eta,
                        etah=energy.shaded.eta,
                        hotspot=inputs.hotspot,
                        nlayers=inputs.nlayers,
                    )
                    for name in fluorescence_profile_outputs:
                        fluorescence_profile_outputs[name].append(getattr(profiles, name))

            if thermal_outputs is not None:
                thermal = self.energy_balance_model._thermal_from_energy(
                    energy=energy,
                    lai=inputs.lai,
                    tts=inputs.tts,
                    tto=inputs.tto,
                    psi=inputs.psi,
                    thermal_optics=inputs.soil.thermal_optics,
                    hotspot=inputs.hotspot,
                    lidf=None,
                    wlT=None,
                )
                for name in thermal_outputs:
                    thermal_outputs[name].append(getattr(thermal, name))

                if thermal_directional_outputs is not None:
                    assert tto_angles is not None and psi_angles is not None
                    directional = self.thermal_model.directional(
                        inputs.lai,
                        inputs.tts,
                        tto_angles,
                        psi_angles,
                        energy.Tcu,
                        energy.Tch,
                        energy.Tsu,
                        energy.Tsh,
                        thermal_optics=inputs.soil.thermal_optics,
                        hotspot=inputs.hotspot,
                        nlayers=inputs.nlayers,
                    )
                    thermal_directional_outputs["Lot_"].append(directional.Lot_)
                    thermal_directional_outputs["BrightnessT"].append(directional.BrightnessT)

                if thermal_profile_outputs is not None:
                    profiles = self.thermal_model.profiles(
                        inputs.lai,
                        inputs.tts,
                        inputs.tto,
                        inputs.psi,
                        energy.Tcu,
                        energy.Tch,
                        energy.Tsu,
                        energy.Tsh,
                        thermal_optics=inputs.soil.thermal_optics,
                        hotspot=inputs.hotspot,
                        nlayers=inputs.nlayers,
                    )
                    for name in thermal_profile_outputs:
                        thermal_profile_outputs[name].append(getattr(profiles, name))

        datasets: list[xr.Dataset] = []
        components: list[str] = []

        def _add(dataset: xr.Dataset, component: str) -> None:
            datasets.append(dataset)
            components.append(component)

        _add(
            self._outputs_to_dataset(data_module, self._concat_outputs(energy_outputs), product="energy_balance"),
            "energy_balance",
        )

        if fluorescence_outputs is not None:
            _add(
                self._outputs_to_dataset(
                    data_module,
                    self._concat_outputs(fluorescence_outputs),
                    product="energy_balance_fluorescence",
                    layer_count=coupled_layer_count,
                ),
                "energy_balance_fluorescence",
            )

        if fluorescence_directional_outputs is not None:
            assert tto_angles is not None and psi_angles is not None
            _add(
                self._prefixed_dataset(
                    self._directional_outputs_to_dataset(
                        data_module,
                        self._concat_outputs(fluorescence_directional_outputs),
                        product="energy_balance_directional_fluorescence",
                        directional_tto=tto_angles,
                        directional_psi=psi_angles,
                        variable_dims={"LoF_": ("direction", "fluorescence_wavelength")},
                    ),
                    "energy_balance_fluorescence_directional",
                ),
                "energy_balance_fluorescence_directional",
            )

        if fluorescence_profile_outputs is not None:
            profile_dataset = self._profile_outputs_to_dataset(
                data_module,
                self._concat_outputs(fluorescence_profile_outputs),
                product="energy_balance_fluorescence_profiles",
                layer_count=self._profile_layer_count_from_tensor(fluorescence_profile_outputs["Ps"][0]),
                variable_dims={
                    "Ps": ("layer_interface",),
                    "Po": ("layer_interface",),
                    "Pso": ("layer_interface",),
                    "Fmin_": ("layer_interface", "fluorescence_wavelength"),
                    "Fplu_": ("layer_interface", "fluorescence_wavelength"),
                    "layer_fluorescence": ("layer",),
                },
            )
            _add(
                self._prefixed_dataset(profile_dataset, "energy_balance_fluorescence_profile"),
                "energy_balance_fluorescence_profile",
            )

        if thermal_outputs is not None:
            _add(
                self._outputs_to_dataset(
                    data_module,
                    self._concat_outputs(thermal_outputs),
                    product="energy_balance_thermal",
                    layer_count=coupled_layer_count,
                ),
                "energy_balance_thermal",
            )

        if thermal_directional_outputs is not None:
            assert tto_angles is not None and psi_angles is not None
            _add(
                self._prefixed_dataset(
                    self._directional_outputs_to_dataset(
                        data_module,
                        self._concat_outputs(thermal_directional_outputs),
                        product="energy_balance_directional_thermal",
                        directional_tto=tto_angles,
                        directional_psi=psi_angles,
                        variable_dims={"Lot_": ("direction", "thermal_wavelength"), "BrightnessT": ("direction",)},
                    ),
                    "energy_balance_thermal_directional",
                ),
                "energy_balance_thermal_directional",
            )

        if thermal_profile_outputs is not None:
            profile_dataset = self._profile_outputs_to_dataset(
                data_module,
                self._concat_outputs(thermal_profile_outputs),
                product="energy_balance_thermal_profiles",
                layer_count=self._profile_layer_count_from_tensor(thermal_profile_outputs["Ps"][0]),
                variable_dims={
                    "Ps": ("layer_interface",),
                    "Po": ("layer_interface",),
                    "Pso": ("layer_interface",),
                    "Emint_": ("layer_interface", "thermal_wavelength"),
                    "Eplut_": ("layer_interface", "thermal_wavelength"),
                    "layer_thermal_upward": ("layer",),
                },
            )
            _add(
                self._prefixed_dataset(profile_dataset, "energy_balance_thermal_profile"),
                "energy_balance_thermal_profile",
            )

        return datasets, components

    def _directional_outputs_to_dataset(
        self,
        data_module: ScopeGridDataModule,
        outputs: Mapping[str, torch.Tensor],
        *,
        product: str,
        directional_tto: torch.Tensor,
        directional_psi: torch.Tensor,
        variable_dims: Mapping[str, Sequence[str]],
    ) -> xr.Dataset:
        variable_coords = self._output_coords(layer_count=None)
        variable_coords["direction"] = torch.arange(
            directional_tto.numel(),
            device=self.fluspect.device,
            dtype=self.fluspect.dtype,
        )
        dataset = data_module.assemble_dataset(
            {name: torch.as_tensor(value) for name, value in outputs.items()},
            variable_dims=variable_dims,
            variable_coords=variable_coords,
            attrs={
                "scope_product": product,
            },
        )
        dataset = dataset.assign_coords(
            directional_tto=("direction", torch.as_tensor(directional_tto).detach().cpu().numpy()),
            directional_psi=("direction", torch.as_tensor(directional_psi).detach().cpu().numpy()),
        )
        return dataset

    def _profile_outputs_to_dataset(
        self,
        data_module: ScopeGridDataModule,
        outputs: Mapping[str, torch.Tensor],
        *,
        product: str,
        layer_count: int,
        variable_dims: Mapping[str, Sequence[str]],
    ) -> xr.Dataset:
        dataset_layer = data_module.dataset.coords.get("layer")
        if dataset_layer is not None and int(dataset_layer.size) != layer_count:
            raise ValueError(f"Dataset layer coordinate has size {dataset_layer.size}, expected {layer_count}")
        variable_coords = self._output_coords(layer_count)
        return data_module.assemble_dataset(
            {name: torch.as_tensor(value) for name, value in outputs.items()},
            variable_dims=variable_dims,
            variable_coords=variable_coords,
            attrs={
                "scope_product": product,
            },
        )

    def _dataset_tensor(self, value: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            spectral_sizes = self._spectral_sizes()
            if int(tensor.shape[2]) in spectral_sizes:
                return tensor.squeeze(1)
        return tensor

    def _output_coords(self, layer_count: int | None) -> dict[str, torch.Tensor]:
        coords: dict[str, torch.Tensor] = {
            "wavelength": self.fluspect.spectral.wlP,
            "thermal_wavelength": default_thermal_wavelengths(
                device=self.fluspect.device,
                dtype=self.fluspect.dtype,
            ),
        }
        if self.fluspect.spectral.wlF is not None:
            coords["fluorescence_wavelength"] = self.fluspect.spectral.wlF
        if self.fluspect.spectral.wlE is not None:
            coords["excitation_wavelength"] = self.fluspect.spectral.wlE
        if layer_count is not None:
            coords["layer_interface"] = torch.arange(
                layer_count + 1,
                device=self.fluspect.device,
                dtype=self.fluspect.dtype,
            )
        return coords

    def _output_layer_count(
        self,
        data_module: ScopeGridDataModule,
        outputs: Mapping[str, torch.Tensor],
    ) -> int | None:
        if "layer" in data_module.dataset.coords:
            return int(data_module.dataset.coords["layer"].size)

        spectral_sizes = self._spectral_sizes()
        layer_votes: Counter[int] = Counter()
        interface_votes: Counter[int] = Counter()

        for name, value in outputs.items():
            tensor = torch.as_tensor(value)
            if tensor.ndim == 2:
                candidate = int(tensor.shape[1])
                if candidate > 0 and candidate not in spectral_sizes:
                    layer_votes[candidate] += 1
                continue
            if tensor.ndim < 3:
                continue

            leading = int(tensor.shape[1])
            trailing = int(tensor.shape[-1])
            if leading > 0 and trailing in spectral_sizes:
                layer_votes[leading] += 1
            if name in {"Emint_", "Eplut_", "Fmin_", "Fplu_"} and leading > 1:
                interface_votes[leading - 1] += 1

        if layer_votes:
            return layer_votes.most_common(1)[0][0]
        if interface_votes:
            return interface_votes.most_common(1)[0][0]
        return None

    def _spectral_sizes(self) -> set[int]:
        spectral_sizes = {
            int(self.fluspect.spectral.wlP.numel()),
            int(default_thermal_wavelengths(device=self.fluspect.device, dtype=self.fluspect.dtype).numel()),
        }
        if self.fluspect.spectral.wlF is not None:
            spectral_sizes.add(int(self.fluspect.spectral.wlF.numel()))
        if self.fluspect.spectral.wlE is not None:
            spectral_sizes.add(int(self.fluspect.spectral.wlE.numel()))
        return spectral_sizes

    def _infer_output_dims(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        layer_count: int | None,
    ) -> tuple[str, ...]:
        trailing = tuple(int(size) for size in torch.as_tensor(tensor).shape[1:])
        if not trailing:
            return ()

        wlP = int(self.fluspect.spectral.wlP.numel())
        wlF = int(self.fluspect.spectral.wlF.numel()) if self.fluspect.spectral.wlF is not None else None
        wlE = int(self.fluspect.spectral.wlE.numel()) if self.fluspect.spectral.wlE is not None else None
        wlT = int(default_thermal_wavelengths(device=self.fluspect.device, dtype=self.fluspect.dtype).numel())

        if len(trailing) == 1:
            size = trailing[0]
            if size == wlP:
                return ("wavelength",)
            if wlF is not None and size == wlF:
                return ("fluorescence_wavelength",)
            if wlE is not None and size == wlE:
                return ("excitation_wavelength",)
            if size == wlT:
                return ("thermal_wavelength",)
            if layer_count is not None and size == layer_count:
                return ("layer",)
            return (f"{name}_dim_1",)

        if len(trailing) == 2:
            first, second = trailing
            if second == wlT:
                if layer_count is not None and first == layer_count + 1:
                    return ("layer_interface", "thermal_wavelength")
                if layer_count is not None and first == layer_count:
                    return ("layer", "thermal_wavelength")
                return (f"{name}_dim_1", "thermal_wavelength")
            if second == wlP:
                if layer_count is not None and first == layer_count + 1:
                    return ("layer_interface", "wavelength")
                if layer_count is not None and first == layer_count:
                    return ("layer", "wavelength")
                return (f"{name}_dim_1", "wavelength")
            if wlF is not None and second == wlF:
                if layer_count is not None and first == layer_count + 1:
                    return ("layer_interface", "fluorescence_wavelength")
                if layer_count is not None and first == layer_count:
                    return ("layer", "fluorescence_wavelength")
                return (f"{name}_dim_1", "fluorescence_wavelength")
            if wlE is not None and second == wlE:
                if layer_count is not None and first == layer_count + 1:
                    return ("layer_interface", "excitation_wavelength")
                if layer_count is not None and first == layer_count:
                    return ("layer", "excitation_wavelength")
                return (f"{name}_dim_1", "excitation_wavelength")
            if layer_count is not None and first == layer_count + 1:
                return ("layer_interface", f"{name}_dim_2")
            if layer_count is not None and first == layer_count:
                return ("layer", f"{name}_dim_2")
            return (f"{name}_dim_1", f"{name}_dim_2")

        return tuple(f"{name}_dim_{idx}" for idx in range(1, len(trailing) + 1))

    def _leafbio_kwargs(self, batch: Mapping[str, torch.Tensor], varmap: Mapping[str, str]) -> dict[str, torch.Tensor]:
        kwargs: dict[str, torch.Tensor] = {}
        for field in LeafBioBatch.__dataclass_fields__:
            if field in varmap and varmap[field] in batch:
                kwargs[field] = batch[varmap[field]]
        return kwargs

    def _biochemistry_kwargs(
        self, batch: Mapping[str, torch.Tensor], varmap: Mapping[str, str]
    ) -> dict[str, torch.Tensor]:
        kwargs: dict[str, torch.Tensor] = {}
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

    def _optional_spectral_input(
        self,
        batch: Mapping[str, torch.Tensor],
        varmap: Mapping[str, str],
        key: str,
    ) -> torch.Tensor | None:
        if key not in varmap or varmap[key] not in batch:
            return None
        return batch[varmap[key]]

    def _optional_batch_input(
        self,
        batch: Mapping[str, torch.Tensor],
        varmap: Mapping[str, str],
        key: str,
        default,
    ):
        if key in varmap and varmap[key] in batch:
            return batch[varmap[key]]
        return default

    def _concat_outputs(self, outputs: Mapping[str, Sequence[torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {name: torch.cat(list(chunks), dim=0) for name, chunks in outputs.items()}

    def _optical_directional_input(
        self,
        batch: Mapping[str, torch.Tensor],
        varmap: Mapping[str, str],
        kind: str,
    ) -> torch.Tensor:
        aliases = (f"{kind}_", f"{kind}_sw")
        for key in aliases:
            if key in varmap and varmap[key] in batch:
                return batch[varmap[key]]
        raise KeyError(f"varmap must provide one of {aliases!r} for directional reflectance runs")

    def _directional_angles(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        directional_tto: torch.Tensor | None,
        directional_psi: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if directional_tto is not None or directional_psi is not None:
            if directional_tto is None or directional_psi is None:
                raise ValueError("Provide both directional_tto and directional_psi")
            return self._directional_tensor(directional_tto), self._directional_tensor(directional_psi)

        tto_name = varmap.get("directional_tto", "directional_tto")
        psi_name = varmap.get("directional_psi", "directional_psi")
        tto = self._directional_source_array(data_module, tto_name)
        psi = self._directional_source_array(data_module, psi_name)
        return self._directional_tensor(tto), self._directional_tensor(psi)

    def _directional_source_array(self, data_module: ScopeGridDataModule, name: str):
        if name in data_module.dataset.coords:
            coord = data_module.dataset.coords[name]
            if coord.ndim != 1:
                raise ValueError(f"Directional coordinate '{name}' must be one-dimensional")
            return coord.values
        if name in data_module.dataset:
            data = data_module.dataset[name]
            if data.ndim != 1:
                raise ValueError(f"Directional variable '{name}' must be one-dimensional")
            return data.values
        raise KeyError(f"Directional angles '{name}' were not provided and are not present on the dataset")

    def _directional_tensor(self, value: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.fluspect.device, dtype=self.fluspect.dtype).reshape(-1)
        if tensor.numel() == 0:
            raise ValueError("Directional angle arrays must not be empty")
        return tensor

    def _workflow_reflectance_varmap(self, varmap: Mapping[str, str]) -> dict[str, str]:
        resolved = dict(varmap)
        if "Esun_sw" in varmap:
            resolved["Esun_"] = varmap["Esun_sw"]
        if "Esky_sw" in varmap:
            resolved["Esky_"] = varmap["Esky_sw"]
        return resolved

    def _scope_option_flag(
        self,
        data_module: ScopeGridDataModule,
        scope_options: Mapping[str, object] | None,
        name: str,
        *,
        default: bool = False,
    ) -> bool:
        if scope_options and name in scope_options:
            return self._as_bool(scope_options[name], default=default)
        return self._as_bool(data_module.dataset.attrs.get(name, default), default=default)

    def _scope_option_int(
        self,
        data_module: ScopeGridDataModule,
        scope_options: Mapping[str, object] | None,
        name: str,
        *,
        default: int,
    ) -> int:
        if scope_options and name in scope_options and scope_options[name] is not None:
            return int(scope_options[name])
        value = data_module.dataset.attrs.get(name, default)
        return default if value is None else int(value)

    def _scope_energy_options(
        self,
        data_module,
        scope_options: Mapping[str, object] | None,
        energy_options: EnergyBalanceOptions | None,
    ) -> EnergyBalanceOptions | None:
        if energy_options is not None:
            return energy_options

        attrs = dict(getattr(data_module.dataset, "attrs", {}))
        if scope_options:
            attrs.update(scope_options)

        aliases = {
            "max_iter": ("energy_max_iter", "max_iter"),
            "max_energy_error": ("energy_max_energy_error", "max_energy_error"),
            "fixed_iterations": ("energy_fixed_iterations", "fixed_iterations", "differentiable_energy_balance"),
            "monin_obukhov": ("energy_monin_obukhov", "monin_obukhov"),
            "initial_shaded_leaf_offset": (
                "energy_initial_shaded_leaf_offset",
                "initial_shaded_leaf_offset",
            ),
            "initial_sunlit_leaf_offset": (
                "energy_initial_sunlit_leaf_offset",
                "initial_sunlit_leaf_offset",
            ),
            "initial_soil_offset": ("energy_initial_soil_offset", "initial_soil_offset"),
        }
        values: dict[str, object] = {}
        for field_name, field_aliases in aliases.items():
            for alias in field_aliases:
                if alias in attrs and attrs[alias] is not None:
                    values[field_name] = attrs[alias]
                    break
        if not values:
            return None

        defaults = EnergyBalanceOptions()
        return EnergyBalanceOptions(
            max_iter=int(values.get("max_iter", defaults.max_iter)),
            max_energy_error=float(values.get("max_energy_error", defaults.max_energy_error)),
            fixed_iterations=self._as_bool(
                values.get("fixed_iterations", defaults.fixed_iterations),
                default=defaults.fixed_iterations,
            ),
            monin_obukhov=self._as_bool(
                values.get("monin_obukhov", defaults.monin_obukhov),
                default=defaults.monin_obukhov,
            ),
            initial_shaded_leaf_offset=float(
                values.get("initial_shaded_leaf_offset", defaults.initial_shaded_leaf_offset)
            ),
            initial_sunlit_leaf_offset=float(
                values.get("initial_sunlit_leaf_offset", defaults.initial_sunlit_leaf_offset)
            ),
            initial_soil_offset=float(values.get("initial_soil_offset", defaults.initial_soil_offset)),
        )

    def _as_bool(self, value: object, *, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            return default
        return bool(value)

    def _profile_layer_count_from_tensor(self, value: torch.Tensor) -> int:
        tensor = torch.as_tensor(value)
        if tensor.ndim < 2 or tensor.shape[1] < 1:
            raise ValueError("Profile outputs must include a layer-interface axis")
        return int(tensor.shape[1]) - 1

    def _accumulate_profile_layer_count(self, expected: int | None, value: torch.Tensor) -> int:
        current = self._profile_layer_count_from_tensor(value)
        if expected is not None and current != expected:
            raise ValueError("Profile workflows require a uniform layer count across all batches")
        return current

    def _layer_count(
        self,
        nlayers: int | None,
        *,
        etau: torch.Tensor | None,
        etah: torch.Tensor | None,
        Tcu: torch.Tensor | None,
        Tch: torch.Tensor | None,
    ) -> int | None:
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
