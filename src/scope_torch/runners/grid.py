from __future__ import annotations

from collections import Counter
from typing import Dict, Mapping, Optional, Sequence

import torch
import xarray as xr

from ..biochem import BiochemicalOptions, LeafBiochemistryInputs, LeafBiochemistryResult
from ..canopy.fluorescence import CanopyFluorescenceModel, CanopyFluorescenceResult
from ..canopy.reflectance import CanopyReflectanceModel, CanopyReflectanceResult
from ..canopy.thermal import CanopyThermalRadianceModel, CanopyThermalRadianceResult, ThermalOptics, default_thermal_wavelengths
from ..energy import (
    CanopyEnergyBalanceResult,
    CanopyEnergyBalanceModel,
    EnergyBalanceCanopy,
    EnergyBalanceMeteo,
    EnergyBalanceOptions,
    EnergyBalanceSoil,
)
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
        self.energy_balance_model = CanopyEnergyBalanceModel(self.reflectance_model)

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
        nlayers: Optional[int] = None,
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
        hotspot_var: Optional[str] = None,
        nlayers: Optional[int] = None,
    ) -> xr.Dataset:
        outputs = self.run(
            data_module,
            varmap=varmap,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
        )
        return self._outputs_to_dataset(data_module, outputs, product="reflectance")

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

    def run_biochemical_fluorescence_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        biochem_options: Optional[BiochemicalOptions] = None,
        hotspot_var: Optional[str] = None,
        nlayers: Optional[int] = None,
    ) -> xr.Dataset:
        outputs = self.run_biochemical_fluorescence(
            data_module,
            varmap=varmap,
            biochem_options=biochem_options,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
        )
        return self._outputs_to_dataset(data_module, outputs, product="biochemical_fluorescence")

    def run_energy_balance_fluorescence(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        biochem_options: Optional[BiochemicalOptions] = None,
        energy_options: Optional[EnergyBalanceOptions] = None,
        hotspot_var: Optional[str] = None,
        nlayers: Optional[int] = None,
        soil_heat_method: int = 2,
    ) -> Dict[str, torch.Tensor]:
        physiology_fields = [name for name in LeafBiochemistryResult.__dataclass_fields__ if name != "fcount"]
        energy_fields = [name for name in CanopyEnergyBalanceResult.__dataclass_fields__ if name not in {"sunlit", "shaded", "Tsold"}]
        outputs: dict[str, list[torch.Tensor]] = {
            **{name: [] for name in CanopyFluorescenceResult.__dataclass_fields__},
            **{name: [] for name in energy_fields},
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
            Esun_sw = self._spectral_input(batch, varmap, "Esun_sw")
            Esky_sw = self._spectral_input(batch, varmap, "Esky_sw")
            soil_refl = self._soil_refl(batch, varmap)
            if hotspot_var and hotspot_var in batch:
                hotspot = batch[hotspot_var]
            else:
                hotspot = torch.full_like(lai, self.default_hotspot)

            meteo = EnergyBalanceMeteo(
                Ta=batch[varmap["Ta"]],
                ea=batch[varmap["ea"]],
                Ca=batch[varmap["Ca"]],
                Oa=batch[varmap["Oa"]],
                p=batch[varmap["p"]],
                z=batch[varmap["z"]],
                u=batch[varmap["u"]],
                L=batch[varmap["L"]] if "L" in varmap and varmap["L"] in batch else -1e6,
            )
            canopy = EnergyBalanceCanopy(
                Cd=batch[varmap["Cd"]],
                rwc=batch[varmap["rwc"]],
                z0m=batch[varmap["z0m"]],
                d=batch[varmap["d"]],
                h=batch[varmap["h"]],
                kV=batch[varmap["kV"]] if "kV" in varmap and varmap["kV"] in batch else 0.0,
                fV=batch[varmap["fV"]] if "fV" in varmap and varmap["fV"] in batch else None,
            )
            soil = EnergyBalanceSoil(
                rss=batch[varmap["rss"]],
                rbs=batch[varmap["rbs"]],
                thermal_optics=ThermalOptics(
                    rho_thermal=batch[varmap["rho_thermal"]] if "rho_thermal" in varmap and varmap["rho_thermal"] in batch else 0.01,
                    tau_thermal=batch[varmap["tau_thermal"]] if "tau_thermal" in varmap and varmap["tau_thermal"] in batch else 0.01,
                    rs_thermal=batch[varmap["rs_thermal"]] if "rs_thermal" in varmap and varmap["rs_thermal"] in batch else 0.06,
                ),
                soil_heat_method=soil_heat_method,
                GAM=batch[varmap["GAM"]] if "GAM" in varmap and varmap["GAM"] in batch else 0.0,
                Tsold=batch[varmap["Tsold"]] if "Tsold" in varmap and varmap["Tsold"] in batch else None,
                dt_seconds=batch[varmap["dt_seconds"]] if "dt_seconds" in varmap and varmap["dt_seconds"] in batch else None,
            )

            result = self.energy_balance_model.solve_fluorescence(
                leafbio,
                biochem,
                soil_refl,
                lai,
                tts,
                tto,
                psi,
                Esun_sw,
                Esky_sw,
                meteo=meteo,
                canopy=canopy,
                soil=soil,
                options=energy_options,
                biochem_options=biochem_options,
                hotspot=hotspot,
                nlayers=self._layer_count(
                    nlayers,
                    etau=canopy.fV if isinstance(canopy.fV, torch.Tensor) else None,
                    etah=None,
                    Tcu=batch[varmap["Tcu0"]] if "Tcu0" in varmap and varmap["Tcu0"] in batch else None,
                    Tch=batch[varmap["Tch0"]] if "Tch0" in varmap and varmap["Tch0"] in batch else None,
                ),
                Tcu0=batch[varmap["Tcu0"]] if "Tcu0" in varmap and varmap["Tcu0"] in batch else None,
                Tch0=batch[varmap["Tch0"]] if "Tch0" in varmap and varmap["Tch0"] in batch else None,
                Tsu0=batch[varmap["Tsu0"]] if "Tsu0" in varmap and varmap["Tsu0"] in batch else None,
                Tsh0=batch[varmap["Tsh0"]] if "Tsh0" in varmap and varmap["Tsh0"] in batch else None,
            )
            for name in CanopyFluorescenceResult.__dataclass_fields__:
                outputs[name].append(getattr(result.fluorescence, name))
            for name in energy_fields:
                outputs[name].append(getattr(result.energy, name))
            for name in physiology_fields:
                outputs[f"sunlit_{name}"].append(getattr(result.energy.sunlit, name))
                outputs[f"shaded_{name}"].append(getattr(result.energy.shaded, name))

        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}

    def run_energy_balance_fluorescence_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        biochem_options: Optional[BiochemicalOptions] = None,
        energy_options: Optional[EnergyBalanceOptions] = None,
        hotspot_var: Optional[str] = None,
        nlayers: Optional[int] = None,
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
        biochem_options: Optional[BiochemicalOptions] = None,
        energy_options: Optional[EnergyBalanceOptions] = None,
        hotspot_var: Optional[str] = None,
        nlayers: Optional[int] = None,
        soil_heat_method: int = 2,
    ) -> Dict[str, torch.Tensor]:
        physiology_fields = [name for name in LeafBiochemistryResult.__dataclass_fields__ if name != "fcount"]
        energy_fields = [name for name in CanopyEnergyBalanceResult.__dataclass_fields__ if name not in {"sunlit", "shaded", "Tsold"}]
        outputs: dict[str, list[torch.Tensor]] = {
            **{name: [] for name in CanopyThermalRadianceResult.__dataclass_fields__},
            **{name: [] for name in energy_fields},
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
            Esun_sw = self._spectral_input(batch, varmap, "Esun_sw")
            Esky_sw = self._spectral_input(batch, varmap, "Esky_sw")
            soil_refl = self._soil_refl(batch, varmap)
            if hotspot_var and hotspot_var in batch:
                hotspot = batch[hotspot_var]
            else:
                hotspot = torch.full_like(lai, self.default_hotspot)

            meteo = EnergyBalanceMeteo(
                Ta=batch[varmap["Ta"]],
                ea=batch[varmap["ea"]],
                Ca=batch[varmap["Ca"]],
                Oa=batch[varmap["Oa"]],
                p=batch[varmap["p"]],
                z=batch[varmap["z"]],
                u=batch[varmap["u"]],
                L=batch[varmap["L"]] if "L" in varmap and varmap["L"] in batch else -1e6,
            )
            canopy = EnergyBalanceCanopy(
                Cd=batch[varmap["Cd"]],
                rwc=batch[varmap["rwc"]],
                z0m=batch[varmap["z0m"]],
                d=batch[varmap["d"]],
                h=batch[varmap["h"]],
                kV=batch[varmap["kV"]] if "kV" in varmap and varmap["kV"] in batch else 0.0,
                fV=batch[varmap["fV"]] if "fV" in varmap and varmap["fV"] in batch else None,
            )
            soil = EnergyBalanceSoil(
                rss=batch[varmap["rss"]],
                rbs=batch[varmap["rbs"]],
                thermal_optics=ThermalOptics(
                    rho_thermal=batch[varmap["rho_thermal"]] if "rho_thermal" in varmap and varmap["rho_thermal"] in batch else 0.01,
                    tau_thermal=batch[varmap["tau_thermal"]] if "tau_thermal" in varmap and varmap["tau_thermal"] in batch else 0.01,
                    rs_thermal=batch[varmap["rs_thermal"]] if "rs_thermal" in varmap and varmap["rs_thermal"] in batch else 0.06,
                ),
                soil_heat_method=soil_heat_method,
                GAM=batch[varmap["GAM"]] if "GAM" in varmap and varmap["GAM"] in batch else 0.0,
                Tsold=batch[varmap["Tsold"]] if "Tsold" in varmap and varmap["Tsold"] in batch else None,
                dt_seconds=batch[varmap["dt_seconds"]] if "dt_seconds" in varmap and varmap["dt_seconds"] in batch else None,
            )

            result = self.energy_balance_model.solve_thermal(
                leafbio,
                biochem,
                soil_refl,
                lai,
                tts,
                tto,
                psi,
                Esun_sw,
                Esky_sw,
                meteo=meteo,
                canopy=canopy,
                soil=soil,
                options=energy_options,
                biochem_options=biochem_options,
                hotspot=hotspot,
                nlayers=self._layer_count(
                    nlayers,
                    etau=canopy.fV if isinstance(canopy.fV, torch.Tensor) else None,
                    etah=None,
                    Tcu=batch[varmap["Tcu0"]] if "Tcu0" in varmap and varmap["Tcu0"] in batch else None,
                    Tch=batch[varmap["Tch0"]] if "Tch0" in varmap and varmap["Tch0"] in batch else None,
                ),
                Tcu0=batch[varmap["Tcu0"]] if "Tcu0" in varmap and varmap["Tcu0"] in batch else None,
                Tch0=batch[varmap["Tch0"]] if "Tch0" in varmap and varmap["Tch0"] in batch else None,
                Tsu0=batch[varmap["Tsu0"]] if "Tsu0" in varmap and varmap["Tsu0"] in batch else None,
                Tsh0=batch[varmap["Tsh0"]] if "Tsh0" in varmap and varmap["Tsh0"] in batch else None,
            )
            for name in CanopyThermalRadianceResult.__dataclass_fields__:
                outputs[name].append(getattr(result.thermal, name))
            for name in energy_fields:
                outputs[name].append(getattr(result.energy, name))
            for name in physiology_fields:
                outputs[f"sunlit_{name}"].append(getattr(result.energy.sunlit, name))
                outputs[f"shaded_{name}"].append(getattr(result.energy.shaded, name))

        return {name: torch.cat(chunks, dim=0) for name, chunks in outputs.items()}

    def run_energy_balance_thermal_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        biochem_options: Optional[BiochemicalOptions] = None,
        energy_options: Optional[EnergyBalanceOptions] = None,
        hotspot_var: Optional[str] = None,
        nlayers: Optional[int] = None,
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

    def run_fluorescence_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: Optional[str] = None,
    ) -> xr.Dataset:
        outputs = self.run_fluorescence(
            data_module,
            varmap=varmap,
            hotspot_var=hotspot_var,
        )
        return self._outputs_to_dataset(data_module, outputs, product="fluorescence")

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

    def run_layered_fluorescence_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: Optional[str] = None,
        nlayers: Optional[int] = None,
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

    def run_thermal_dataset(
        self,
        data_module: ScopeGridDataModule,
        *,
        varmap: Mapping[str, str],
        hotspot_var: Optional[str] = None,
        nlayers: Optional[int] = None,
    ) -> xr.Dataset:
        outputs = self.run_thermal(
            data_module,
            varmap=varmap,
            hotspot_var=hotspot_var,
            nlayers=nlayers,
        )
        return self._outputs_to_dataset(data_module, outputs, product="thermal")

    def _outputs_to_dataset(
        self,
        data_module: ScopeGridDataModule,
        outputs: Mapping[str, torch.Tensor],
        *,
        product: str,
    ) -> xr.Dataset:
        dataset_outputs = {name: self._dataset_tensor(value) for name, value in outputs.items()}
        layer_count = self._output_layer_count(data_module, dataset_outputs)
        variable_dims = {
            name: self._infer_output_dims(name, tensor, layer_count=layer_count)
            for name, tensor in dataset_outputs.items()
        }
        variable_coords = self._output_coords(layer_count)
        return data_module.assemble_dataset(
            dataset_outputs,
            variable_dims=variable_dims,
            variable_coords=variable_coords,
            attrs={"scope_torch_product": product},
        )

    def _dataset_tensor(self, value: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value)
        if tensor.ndim == 3 and tensor.shape[1] == 1:
            spectral_sizes = self._spectral_sizes()
            if int(tensor.shape[2]) in spectral_sizes:
                return tensor.squeeze(1)
        return tensor

    def _output_coords(self, layer_count: Optional[int]) -> Dict[str, torch.Tensor]:
        coords: Dict[str, torch.Tensor] = {
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
    ) -> Optional[int]:
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
        layer_count: Optional[int],
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
