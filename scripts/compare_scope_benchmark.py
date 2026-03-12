from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.io import loadmat

from scope_torch.biochem import LeafBiochemistryInputs, LeafMeteo
from scope_torch.canopy.fluorescence import CanopyFluorescenceModel
from scope_torch.canopy.foursail import FourSAILModel
from scope_torch.canopy.reflectance import CanopyReflectanceModel
from scope_torch.canopy.thermal import CanopyThermalRadianceModel, ThermalOptics
from scope_torch.energy import (
    CanopyEnergyBalanceModel,
    EnergyBalanceCanopy,
    EnergyBalanceMeteo,
    EnergyBalanceOptions,
    EnergyBalanceSoil,
    ResistanceInputs,
    aerodynamic_resistances,
)
from scope_torch.spectral.fluspect import FluspectModel, LeafBioBatch


def _load_benchmark(path: Path) -> dict[str, Any]:
    raw = loadmat(path, simplify_cells=True)
    return {key: value for key, value in raw.items() if not key.startswith("__")}


def _as_string(value: Any) -> str:
    if isinstance(value, str):
        return value
    array = np.asarray(value)
    if array.ndim == 0:
        return str(array.item())
    if array.dtype.kind in {"U", "S"}:
        return "".join(array.reshape(-1).tolist())
    return str(value)


def _tensor(value: Any, *, device: torch.device, dtype: torch.dtype, atleast_1d: bool = False) -> torch.Tensor:
    array = np.asarray(value, dtype=np.float64)
    tensor = torch.as_tensor(array, device=device, dtype=dtype)
    if atleast_1d and tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    return tensor


def _scalar(value: Any) -> float:
    return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])


def _vector(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _tensor(value, device=device, dtype=dtype).reshape(-1)


def _batch_spectrum(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _vector(value, device=device, dtype=dtype).unsqueeze(0)


def _batch_profile(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return _vector(value, device=device, dtype=dtype).unsqueeze(0)


def _metrics(predicted: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    pred = predicted.detach().cpu().to(torch.float64).reshape(-1)
    ref = reference.detach().cpu().to(torch.float64).reshape(-1)
    if pred.shape != ref.shape:
        raise ValueError(f"Shape mismatch: predicted {tuple(pred.shape)} vs reference {tuple(ref.shape)}")
    finite = torch.isfinite(pred) & torch.isfinite(ref)
    if not bool(finite.any().item()):
        return {"max_abs": float("nan"), "mean_abs": float("nan"), "max_rel": float("nan"), "mean_rel": float("nan")}
    pred = pred[finite]
    ref = ref[finite]
    abs_err = torch.abs(pred - ref)
    scale = torch.clamp(torch.abs(ref), min=1e-9)
    rel_err = abs_err / scale
    return {
        "max_abs": float(torch.max(abs_err).item()),
        "mean_abs": float(torch.mean(abs_err).item()),
        "max_rel": float(torch.max(rel_err).item()),
        "mean_rel": float(torch.mean(rel_err).item()),
    }


def _scope_refl_from_rso_rdo(rso: torch.Tensor, rdo: torch.Tensor, esun: torch.Tensor, esky: torch.Tensor) -> torch.Tensor:
    total = esun + esky
    refl = (rso * esun + rdo * esky) / total.clamp(min=1e-12)
    refl = torch.where(esky < 1e-4, rso, refl)
    refl = torch.where(esky < (2e-4 * torch.max(esky)), rso, refl)
    return refl


def _record(report: dict[str, dict[str, dict[str, float]]], section: str, name: str, predicted: torch.Tensor, reference: torch.Tensor) -> None:
    report.setdefault(section, {})[name] = _metrics(predicted, reference)


def _print_report(report: dict[str, dict[str, dict[str, float]]]) -> None:
    for section, metrics in report.items():
        print(section)
        for name, values in metrics.items():
            print(
                f"  {name:<18} "
                f"max_abs={values['max_abs']:.6e} "
                f"max_rel={values['max_rel']:.6e} "
                f"mean_abs={values['mean_abs']:.6e}"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare scope_torch outputs against a MATLAB-exported SCOPE benchmark scene.")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path("tests/data/scope_case_001.mat"),
        help="Path to the MATLAB-exported benchmark MAT file.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Optional JSON file for the comparison summary.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device to run the comparison on.",
    )
    args = parser.parse_args()

    benchmark = _load_benchmark(args.benchmark)
    repo_root = Path(__file__).resolve().parents[1]
    scope_root = repo_root / "upstream" / "SCOPE"

    device = torch.device(args.device)
    dtype = torch.float64

    wlF = _vector(benchmark["wlF"], device=device, dtype=dtype)
    wlE = _vector(benchmark["wlE"], device=device, dtype=dtype)
    rtmf_wlF = _vector(benchmark["rtmf_wlF"], device=device, dtype=dtype)
    rtmf_wlE = _vector(benchmark["rtmf_wlE"], device=device, dtype=dtype)
    fluspect = FluspectModel.from_scope_assets(
        scope_root_path=str(scope_root),
        device=device,
        dtype=dtype,
        wlF=wlF,
        wlE=wlE,
    )
    fluspect_rtmf = FluspectModel.from_scope_assets(
        scope_root_path=str(scope_root),
        device=device,
        dtype=dtype,
        wlF=rtmf_wlF,
        wlE=rtmf_wlE,
    )

    lidf = _vector(benchmark["canopy_lidf"], device=device, dtype=dtype)
    hotspot = torch.tensor([_scalar(benchmark["canopy_hot"])], device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)
    reflectance_model = CanopyReflectanceModel(fluspect, sail, lidf=lidf, default_hotspot=float(hotspot.item()))
    fluorescence_model = CanopyFluorescenceModel(reflectance_model)
    thermal_model = CanopyThermalRadianceModel(reflectance_model)
    energy_model = CanopyEnergyBalanceModel(reflectance_model)
    reflectance_model_rtmf = CanopyReflectanceModel(fluspect_rtmf, sail, lidf=lidf, default_hotspot=float(hotspot.item()))
    fluorescence_model_rtmf = CanopyFluorescenceModel(reflectance_model_rtmf)

    leafbio = LeafBioBatch(
        Cab=torch.tensor([_scalar(benchmark["leaf_Cab"])], device=device, dtype=dtype),
        Cca=torch.tensor([_scalar(benchmark["leaf_Cca"])], device=device, dtype=dtype),
        Cw=torch.tensor([_scalar(benchmark["leaf_Cw"])], device=device, dtype=dtype),
        Cdm=torch.tensor([_scalar(benchmark["leaf_Cdm"])], device=device, dtype=dtype),
        Cs=torch.tensor([_scalar(benchmark["leaf_Cs"])], device=device, dtype=dtype),
        Cant=torch.tensor([_scalar(benchmark["leaf_Cant"])], device=device, dtype=dtype),
        Cbc=torch.tensor([_scalar(benchmark["leaf_Cbc"])], device=device, dtype=dtype),
        Cp=torch.tensor([_scalar(benchmark["leaf_Cp"])], device=device, dtype=dtype),
        N=torch.tensor([_scalar(benchmark["leaf_N"])], device=device, dtype=dtype),
        fqe=torch.tensor([_scalar(benchmark["leaf_fqe"])], device=device, dtype=dtype),
    )

    soil_refl = _batch_spectrum(benchmark["soil_refl"], device=device, dtype=dtype)
    lai = torch.tensor([_scalar(benchmark["canopy_LAI"])], device=device, dtype=dtype)
    tts = torch.tensor([_scalar(benchmark["tts"])], device=device, dtype=dtype)
    tto = torch.tensor([_scalar(benchmark["tto"])], device=device, dtype=dtype)
    psi = torch.tensor([_scalar(benchmark["psi"])], device=device, dtype=dtype)
    esun_wlp = _batch_spectrum(benchmark["Esun_wlP"], device=device, dtype=dtype)
    esky_wlp = _batch_spectrum(benchmark["Esky_wlP"], device=device, dtype=dtype)
    esun_wlt = _batch_spectrum(benchmark["Esun_wlT"], device=device, dtype=dtype)
    esky_wlt = _batch_spectrum(benchmark["Esky_wlT"], device=device, dtype=dtype)
    esun_wle = _batch_spectrum(benchmark["Esun_wlE"], device=device, dtype=dtype)
    esky_wle = _batch_spectrum(benchmark["Esky_wlE"], device=device, dtype=dtype)
    esun_rtmf = _batch_spectrum(benchmark["Esun_rtmf"], device=device, dtype=dtype)
    esky_rtmf = _batch_spectrum(benchmark["Esky_rtmf"], device=device, dtype=dtype)
    nlayers = int(_scalar(benchmark["nlayers"]))

    report: dict[str, dict[str, dict[str, float]]] = {}

    leafopt = fluspect(leafbio)
    leafopt_rtmf = fluspect_rtmf(leafbio)
    _record(report, "leaf", "refl", leafopt.refl[0], _vector(benchmark["leaf_refl"], device=device, dtype=dtype))
    _record(report, "leaf", "tran", leafopt.tran[0], _vector(benchmark["leaf_tran"], device=device, dtype=dtype))
    _record(report, "leaf", "Mb", leafopt_rtmf.Mb[0], _tensor(benchmark["leaf_Mb"], device=device, dtype=dtype))
    _record(report, "leaf", "Mf", leafopt_rtmf.Mf[0], _tensor(benchmark["leaf_Mf"], device=device, dtype=dtype))

    canopy = reflectance_model(leafbio, soil_refl, lai, tts, tto, psi, hotspot=hotspot, nlayers=nlayers)
    _record(report, "reflectance", "rsd", canopy.rsd[0], _vector(benchmark["canopy_rsd"], device=device, dtype=dtype))
    _record(report, "reflectance", "rdd", canopy.rdd[0], _vector(benchmark["canopy_rdd"], device=device, dtype=dtype))
    _record(report, "reflectance", "rdo", canopy.rdo[0], _vector(benchmark["canopy_rdo"], device=device, dtype=dtype))
    _record(report, "reflectance", "rso", canopy.rso[0], _vector(benchmark["canopy_rso"], device=device, dtype=dtype))
    canopy_refl = _scope_refl_from_rso_rdo(canopy.rso[0], canopy.rdo[0], esun_wlp[0], esky_wlp[0])
    _record(report, "reflectance", "refl", canopy_refl, _vector(benchmark["canopy_refl"], device=device, dtype=dtype))

    etau = _batch_profile(benchmark["energy_sunlit_eta"], device=device, dtype=dtype)
    etah = _batch_profile(benchmark["energy_shaded_eta"], device=device, dtype=dtype)
    diagnostics = fluorescence_model_rtmf._layered_diagnostics(
        leafopt=leafopt_rtmf,
        leafbio=leafbio,
        soil_refl=soil_refl,
        lai=lai,
        tts=tts,
        tto=tto,
        psi=psi,
        Esun_=esun_rtmf,
        Esky_=esky_rtmf,
        hotspot=hotspot,
        lidf=None,
        nlayers=nlayers,
        wlP=fluspect_rtmf.spectral.wlP,
        wlE=fluspect_rtmf.spectral.wlE,
        wlF=fluspect_rtmf.spectral.wlF,
        etau=etau,
        etah=etah,
    )
    _record(report, "fluorescence_source", "MpluEsun", diagnostics.MpluEsun[0], _tensor(benchmark["fluor_MpluEsun"], device=device, dtype=dtype)[:, 0])
    _record(report, "fluorescence_source", "MminEsun", diagnostics.MminEsun[0], _tensor(benchmark["fluor_MminEsun"], device=device, dtype=dtype)[:, 0])
    _record(report, "fluorescence_source", "piLs", diagnostics.piLs[0], _tensor(benchmark["fluor_piLs"], device=device, dtype=dtype).transpose(0, 1))
    _record(report, "fluorescence_source", "piLd", diagnostics.piLd[0], _tensor(benchmark["fluor_piLd"], device=device, dtype=dtype).transpose(0, 1))
    _record(report, "fluorescence_source", "Femmin", diagnostics.Femmin[0], _tensor(benchmark["fluor_Femmin"], device=device, dtype=dtype).transpose(0, 1))
    _record(report, "fluorescence_source", "Femplu", diagnostics.Femplu[0], _tensor(benchmark["fluor_Femplu"], device=device, dtype=dtype).transpose(0, 1))
    _record(report, "fluorescence_source", "Fmin_", diagnostics.Fmin_[0], _tensor(benchmark["fluor_Fmin"], device=device, dtype=dtype))
    _record(report, "fluorescence_source", "Fplu_", diagnostics.Fplu_[0], _tensor(benchmark["fluor_Fplu"], device=device, dtype=dtype))

    fluorescence = fluorescence_model.layered(
        leafbio,
        soil_refl,
        lai,
        tts,
        tto,
        psi,
        esun_wle,
        esky_wle,
        etau=etau,
        etah=etah,
        Pnu_Cab=_batch_profile(benchmark["energy_Pnu_Cab"], device=device, dtype=dtype),
        Pnh_Cab=_batch_profile(benchmark["energy_Pnh_Cab"], device=device, dtype=dtype),
        hotspot=hotspot,
        nlayers=nlayers,
    )
    _record(report, "fluorescence_transport", "PoutFrc", fluorescence_model._layered_source_poutfrc(
        leafopt=fluspect(leafbio),
        leafbio=leafbio,
        soil_refl=soil_refl,
        lai=lai,
        tts=tts,
        tto=tto,
        psi=psi,
        Esun=esun_wle,
        Esky=esky_wle,
        hotspot=hotspot,
        lidf=None,
        nlayers=nlayers,
        etau=etau,
        etah=etah,
        Pnu_Cab=_batch_profile(benchmark["energy_Pnu_Cab"], device=device, dtype=dtype),
        Pnh_Cab=_batch_profile(benchmark["energy_Pnh_Cab"], device=device, dtype=dtype),
        wlP=fluspect.spectral.wlP,
        wlE=fluspect.spectral.wlE,
    ), torch.tensor([_scalar(benchmark["fluor_PoutFrc"])], device=device, dtype=dtype))
    _record(report, "fluorescence_transport", "LoF_", fluorescence.LoF_[0], _vector(benchmark["fluor_LoF"], device=device, dtype=dtype))
    _record(report, "fluorescence_transport", "EoutF_", fluorescence.EoutF_[0], _vector(benchmark["fluor_EoutF"], device=device, dtype=dtype))
    _record(report, "fluorescence_transport", "EoutFrc_", fluorescence.EoutFrc_[0], _vector(benchmark["fluor_EoutFrc"], device=device, dtype=dtype))
    _record(report, "fluorescence_transport", "Femleaves_", fluorescence.Femleaves_[0], _vector(benchmark["fluor_Femleaves"], device=device, dtype=dtype))
    _record(report, "fluorescence_transport", "sigmaF", fluorescence.sigmaF[0], _vector(benchmark["fluor_sigmaF"], device=device, dtype=dtype))
    _record(report, "fluorescence_transport", "LoF_sunlit", fluorescence.LoF_sunlit[0], _vector(benchmark["fluor_LoF_sunlit"], device=device, dtype=dtype))
    _record(report, "fluorescence_transport", "LoF_shaded", fluorescence.LoF_shaded[0], _vector(benchmark["fluor_LoF_shaded"], device=device, dtype=dtype))
    _record(report, "fluorescence_transport", "LoF_scattered", fluorescence.LoF_scattered[0], _vector(benchmark["fluor_LoF_scattered"], device=device, dtype=dtype))
    _record(report, "fluorescence_transport", "LoF_soil", fluorescence.LoF_soil[0], _vector(benchmark["fluor_LoF_soil"], device=device, dtype=dtype))

    thermal_optics = ThermalOptics(
        rho_thermal=_scalar(benchmark["leaf_rho_thermal"]),
        tau_thermal=_scalar(benchmark["leaf_tau_thermal"]),
        rs_thermal=_scalar(benchmark["soil_rs_thermal"]),
    )
    thermal = thermal_model(
        lai=lai,
        tts=tts,
        tto=tto,
        psi=psi,
        Tcu=_batch_profile(benchmark["energy_Tcu"], device=device, dtype=dtype),
        Tch=_batch_profile(benchmark["energy_Tch"], device=device, dtype=dtype),
        Tsu=torch.tensor([_scalar(benchmark["energy_Tsu"])], device=device, dtype=dtype),
        Tsh=torch.tensor([_scalar(benchmark["energy_Tsh"])], device=device, dtype=dtype),
        thermal_optics=thermal_optics,
        hotspot=hotspot,
        nlayers=nlayers,
    )
    _record(report, "thermal_transport", "Lot_", thermal.Lot_[0], _vector(benchmark["thermal_Lot"], device=device, dtype=dtype))
    _record(report, "thermal_transport", "Eoutte_", thermal.Eoutte_[0], _vector(benchmark["thermal_Eoutte"], device=device, dtype=dtype))
    _record(report, "thermal_transport", "Loutt", thermal.Loutt, torch.tensor([_scalar(benchmark["thermal_Loutt"])], device=device, dtype=dtype))
    _record(report, "thermal_transport", "Eoutt", thermal.Eoutt, torch.tensor([_scalar(benchmark["thermal_Eoutt"])], device=device, dtype=dtype))

    direct_resistances = aerodynamic_resistances(
        ResistanceInputs(
            LAI=lai,
            Cd=torch.tensor([_scalar(benchmark["canopy_Cd"])], device=device, dtype=dtype),
            rwc=torch.tensor([_scalar(benchmark["canopy_rwc"])], device=device, dtype=dtype),
            z0m=torch.tensor([_scalar(benchmark["canopy_zo"])], device=device, dtype=dtype),
            d=torch.tensor([_scalar(benchmark["canopy_d"])], device=device, dtype=dtype),
            h=torch.tensor([_scalar(benchmark["canopy_hc"])], device=device, dtype=dtype),
            z=torch.tensor([_scalar(benchmark["meteo_z"])], device=device, dtype=dtype),
            u=torch.tensor([_scalar(benchmark["meteo_u"])], device=device, dtype=dtype),
            L=torch.tensor([_scalar(benchmark.get("resistance_L", benchmark["meteo_L"]))], device=device, dtype=dtype),
            rbs=torch.tensor([_scalar(benchmark["soil_rbs"])], device=device, dtype=dtype),
        )
    )
    _record(report, "resistances_direct", "ustar", direct_resistances.ustar, torch.tensor([_scalar(benchmark["resistance_ustar"])], device=device, dtype=dtype))
    _record(report, "resistances_direct", "Kh", direct_resistances.Kh, torch.tensor([_scalar(benchmark["resistance_Kh"])], device=device, dtype=dtype))
    _record(report, "resistances_direct", "uz0", direct_resistances.uz0, torch.tensor([_scalar(benchmark["resistance_uz0"])], device=device, dtype=dtype))
    _record(report, "resistances_direct", "rai", direct_resistances.rai, torch.tensor([_scalar(benchmark["resistance_rai"])], device=device, dtype=dtype))
    _record(report, "resistances_direct", "rar", direct_resistances.rar, torch.tensor([_scalar(benchmark["resistance_rar"])], device=device, dtype=dtype))
    _record(report, "resistances_direct", "rac", direct_resistances.rac, torch.tensor([_scalar(benchmark["resistance_rac"])], device=device, dtype=dtype))
    _record(report, "resistances_direct", "rws", direct_resistances.rws, torch.tensor([_scalar(benchmark["resistance_rws"])], device=device, dtype=dtype))
    _record(report, "resistances_direct", "raa", direct_resistances.raa, torch.tensor([_scalar(benchmark["resistance_raa"])], device=device, dtype=dtype))
    _record(report, "resistances_direct", "rawc", direct_resistances.rawc, torch.tensor([_scalar(benchmark["resistance_rawc"])], device=device, dtype=dtype))
    _record(report, "resistances_direct", "raws", direct_resistances.raws, torch.tensor([_scalar(benchmark["resistance_raws"])], device=device, dtype=dtype))

    biochemistry = LeafBiochemistryInputs(
        Vcmax25=torch.tensor([_scalar(benchmark["biochem_Vcmax25"])], device=device, dtype=dtype),
        BallBerrySlope=torch.tensor([_scalar(benchmark["biochem_BallBerrySlope"])], device=device, dtype=dtype),
        Type=_as_string(benchmark["biochem_Type"]),
        BallBerry0=torch.tensor([_scalar(benchmark["biochem_BallBerry0"])], device=device, dtype=dtype),
        RdPerVcmax25=torch.tensor([_scalar(benchmark["biochem_RdPerVcmax25"])], device=device, dtype=dtype),
        Kn0=torch.tensor([_scalar(benchmark["biochem_Kn0"])], device=device, dtype=dtype),
        Knalpha=torch.tensor([_scalar(benchmark["biochem_Knalpha"])], device=device, dtype=dtype),
        Knbeta=torch.tensor([_scalar(benchmark["biochem_Knbeta"])], device=device, dtype=dtype),
        stressfactor=torch.tensor([_scalar(benchmark["biochem_stressfactor"])], device=device, dtype=dtype),
    )
    meteo = EnergyBalanceMeteo(
        Ta=torch.tensor([_scalar(benchmark["meteo_Ta"])], device=device, dtype=dtype),
        ea=torch.tensor([_scalar(benchmark["meteo_ea"])], device=device, dtype=dtype),
        Ca=torch.tensor([_scalar(benchmark["meteo_Ca"])], device=device, dtype=dtype),
        Oa=torch.tensor([_scalar(benchmark["meteo_Oa"])], device=device, dtype=dtype),
        p=torch.tensor([_scalar(benchmark["meteo_p"])], device=device, dtype=dtype),
        z=torch.tensor([_scalar(benchmark["meteo_z"])], device=device, dtype=dtype),
        u=torch.tensor([_scalar(benchmark["meteo_u"])], device=device, dtype=dtype),
        L=torch.tensor([-1e6], device=device, dtype=dtype),
    )
    canopy_state = EnergyBalanceCanopy(
        Cd=torch.tensor([_scalar(benchmark["canopy_Cd"])], device=device, dtype=dtype),
        rwc=torch.tensor([_scalar(benchmark["canopy_rwc"])], device=device, dtype=dtype),
        z0m=torch.tensor([_scalar(benchmark["canopy_zo"])], device=device, dtype=dtype),
        d=torch.tensor([_scalar(benchmark["canopy_d"])], device=device, dtype=dtype),
        h=torch.tensor([_scalar(benchmark["canopy_hc"])], device=device, dtype=dtype),
        kV=torch.tensor([_scalar(benchmark["canopy_kV"])], device=device, dtype=dtype),
    )
    soil_state = EnergyBalanceSoil(
        rss=torch.tensor([_scalar(benchmark["soil_rss"])], device=device, dtype=dtype),
        rbs=torch.tensor([_scalar(benchmark["soil_rbs"])], device=device, dtype=dtype),
        thermal_optics=thermal_optics,
        soil_heat_method=2,
    )
    energy_options = EnergyBalanceOptions(max_iter=100, max_energy_error=1.0, monin_obukhov=True)
    energy = energy_model.solve(
        leafbio,
        biochemistry,
        soil_refl,
        lai,
        tts,
        tto,
        psi,
        esun_wlp,
        esky_wlp,
        Esun_lw=esun_wlt,
        Esky_lw=esky_wlt,
        wlT=_vector(benchmark["wlT"], device=device, dtype=dtype),
        meteo=meteo,
        canopy=canopy_state,
        soil=soil_state,
        options=energy_options,
        hotspot=hotspot,
        nlayers=nlayers,
    )
    if "energy_iter_Csu" in benchmark:
        _record(report, "energy_iteration_input", "sunlit_Cs", energy.sunlit_Cs_input, _batch_profile(benchmark["energy_iter_Csu"], device=device, dtype=dtype))
        _record(report, "energy_iteration_input", "shaded_Cs", energy.shaded_Cs_input, _batch_profile(benchmark["energy_iter_Csh"], device=device, dtype=dtype))
        _record(report, "energy_iteration_input", "sunlit_eb", energy.sunlit_eb_input, _batch_profile(benchmark["energy_iter_ebu"], device=device, dtype=dtype))
        _record(report, "energy_iteration_input", "shaded_eb", energy.shaded_eb_input, _batch_profile(benchmark["energy_iter_ebh"], device=device, dtype=dtype))
        _record(report, "energy_iteration_input", "sunlit_T", energy.sunlit_T_input, _batch_profile(benchmark["energy_iter_Tcu"], device=device, dtype=dtype))
        _record(report, "energy_iteration_input", "shaded_T", energy.shaded_T_input, _batch_profile(benchmark["energy_iter_Tch"], device=device, dtype=dtype))

        xl = torch.cat(
            [
                torch.zeros(1, device=device, dtype=dtype),
                -torch.arange(1, nlayers + 1, device=device, dtype=dtype) / float(nlayers),
            ]
        )
        fV = torch.exp(torch.tensor([_scalar(benchmark["canopy_kV"])], device=device, dtype=dtype).unsqueeze(-1) * xl[:-1].unsqueeze(0)).reshape(-1)
        leaf_kernel = energy_model.fluorescence_model.leaf_biochemistry
        sunlit_iter = leaf_kernel(
            biochemistry,
            LeafMeteo(
                Q=_vector(benchmark["energy_iter_Pnu_Cab"], device=device, dtype=dtype),
                Cs=_vector(benchmark["energy_iter_Csu"], device=device, dtype=dtype),
                T=_vector(benchmark["energy_iter_Tcu"], device=device, dtype=dtype),
                eb=_vector(benchmark["energy_iter_ebu"], device=device, dtype=dtype),
                Oa=torch.full((nlayers,), _scalar(benchmark["meteo_Oa"]), device=device, dtype=dtype),
                p=torch.full((nlayers,), _scalar(benchmark["meteo_p"]), device=device, dtype=dtype),
            ),
            fV=fV,
        )
        shaded_iter = leaf_kernel(
            biochemistry,
            LeafMeteo(
                Q=_vector(benchmark["energy_iter_Pnh_Cab"], device=device, dtype=dtype),
                Cs=_vector(benchmark["energy_iter_Csh"], device=device, dtype=dtype),
                T=_vector(benchmark["energy_iter_Tch"], device=device, dtype=dtype),
                eb=_vector(benchmark["energy_iter_ebh"], device=device, dtype=dtype),
                Oa=torch.full((nlayers,), _scalar(benchmark["meteo_Oa"]), device=device, dtype=dtype),
                p=torch.full((nlayers,), _scalar(benchmark["meteo_p"]), device=device, dtype=dtype),
            ),
            fV=fV,
        )
        _record(report, "leaf_iteration", "sunlit_A", sunlit_iter.A, _vector(benchmark["energy_sunlit_A"], device=device, dtype=dtype))
        _record(report, "leaf_iteration", "shaded_A", shaded_iter.A, _vector(benchmark["energy_shaded_A"], device=device, dtype=dtype))
        _record(report, "leaf_iteration", "sunlit_Ci", sunlit_iter.Ci, _vector(benchmark["energy_sunlit_Ci"], device=device, dtype=dtype))
        _record(report, "leaf_iteration", "shaded_Ci", shaded_iter.Ci, _vector(benchmark["energy_shaded_Ci"], device=device, dtype=dtype))
        _record(report, "leaf_iteration", "sunlit_rcw", sunlit_iter.rcw, _vector(benchmark["energy_sunlit_rcw"], device=device, dtype=dtype))
        _record(report, "leaf_iteration", "shaded_rcw", shaded_iter.rcw, _vector(benchmark["energy_shaded_rcw"], device=device, dtype=dtype))
        _record(report, "leaf_iteration", "sunlit_eta", sunlit_iter.eta, _vector(benchmark["energy_sunlit_eta"], device=device, dtype=dtype))
        _record(report, "leaf_iteration", "shaded_eta", shaded_iter.eta, _vector(benchmark["energy_shaded_eta"], device=device, dtype=dtype))
    _record(report, "energy_balance", "sunlit_eta", energy.sunlit.eta, _batch_profile(benchmark["energy_sunlit_eta"], device=device, dtype=dtype))
    _record(report, "energy_balance", "shaded_eta", energy.shaded.eta, _batch_profile(benchmark["energy_shaded_eta"], device=device, dtype=dtype))
    _record(report, "energy_balance", "sunlit_A", energy.sunlit.A, _batch_profile(benchmark["energy_sunlit_A"], device=device, dtype=dtype))
    _record(report, "energy_balance", "shaded_A", energy.shaded.A, _batch_profile(benchmark["energy_shaded_A"], device=device, dtype=dtype))
    _record(report, "energy_balance", "sunlit_Ci", energy.sunlit.Ci, _batch_profile(benchmark["energy_sunlit_Ci"], device=device, dtype=dtype))
    _record(report, "energy_balance", "shaded_Ci", energy.shaded.Ci, _batch_profile(benchmark["energy_shaded_Ci"], device=device, dtype=dtype))
    _record(report, "energy_balance", "sunlit_rcw", energy.sunlit.rcw, _batch_profile(benchmark["energy_sunlit_rcw"], device=device, dtype=dtype))
    _record(report, "energy_balance", "shaded_rcw", energy.shaded.rcw, _batch_profile(benchmark["energy_shaded_rcw"], device=device, dtype=dtype))
    _record(report, "energy_balance", "Tcu", energy.Tcu, _batch_profile(benchmark["energy_Tcu"], device=device, dtype=dtype))
    _record(report, "energy_balance", "Tch", energy.Tch, _batch_profile(benchmark["energy_Tch"], device=device, dtype=dtype))
    _record(report, "energy_balance", "Tsu", energy.Tsu, torch.tensor([_scalar(benchmark["energy_Tsu"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "Tsh", energy.Tsh, torch.tensor([_scalar(benchmark["energy_Tsh"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "Rnuc_sw", energy.Rnuc_sw, _batch_profile(benchmark["energy_Rnuc_sw"], device=device, dtype=dtype))
    _record(report, "energy_balance", "Rnhc_sw", energy.Rnhc_sw, _batch_profile(benchmark["energy_Rnhc_sw"], device=device, dtype=dtype))
    _record(report, "energy_balance", "Rnus_sw", energy.Rnus_sw, torch.tensor([_scalar(benchmark["energy_Rnus_sw"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "Rnhs_sw", energy.Rnhs_sw, torch.tensor([_scalar(benchmark["energy_Rnhs_sw"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "Rnuct", energy.Rnuct, _batch_profile(benchmark["energy_Rnuct"], device=device, dtype=dtype))
    _record(report, "energy_balance", "Rnhct", energy.Rnhct, _batch_profile(benchmark["energy_Rnhct"], device=device, dtype=dtype))
    _record(report, "energy_balance", "Rnuc", energy.Rnuc, _batch_profile(benchmark["energy_Rnuc"], device=device, dtype=dtype))
    _record(report, "energy_balance", "Rnhc", energy.Rnhc, _batch_profile(benchmark["energy_Rnhc"], device=device, dtype=dtype))
    _record(report, "energy_balance", "Rnus", energy.Rnus, torch.tensor([_scalar(benchmark["energy_Rnus"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "Rnhs", energy.Rnhs, torch.tensor([_scalar(benchmark["energy_Rnhs"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "canopyemis", energy.canopyemis, torch.tensor([_scalar(benchmark["energy_canopyemis"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "L", energy.L, torch.tensor([_scalar(benchmark["meteo_L"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "counter", energy.counter.to(torch.float64), torch.tensor([_scalar(benchmark["energy_counter"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "Rnctot", energy.Rnctot, torch.tensor([_scalar(benchmark["flux_Rnctot"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "lEctot", energy.lEctot, torch.tensor([_scalar(benchmark["flux_lEctot"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "Hctot", energy.Hctot, torch.tensor([_scalar(benchmark["flux_Hctot"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "Actot", energy.Actot, torch.tensor([_scalar(benchmark["flux_Actot"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "Tcave", energy.Tcave, torch.tensor([_scalar(benchmark["flux_Tcave"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "Rnstot", energy.Rnstot, torch.tensor([_scalar(benchmark["flux_Rnstot"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "lEstot", energy.lEstot, torch.tensor([_scalar(benchmark["flux_lEstot"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "Hstot", energy.Hstot, torch.tensor([_scalar(benchmark["flux_Hstot"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "Gtot", energy.Gtot, torch.tensor([_scalar(benchmark["flux_Gtot"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "Tsave", energy.Tsave, torch.tensor([_scalar(benchmark["flux_Tsave"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "Rntot", energy.Rntot, torch.tensor([_scalar(benchmark["flux_Rntot"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "lEtot", energy.lEtot, torch.tensor([_scalar(benchmark["flux_lEtot"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "Htot", energy.Htot, torch.tensor([_scalar(benchmark["flux_Htot"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "raa", energy.raa, torch.tensor([_scalar(benchmark["resistance_raa"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "rawc", energy.rawc, torch.tensor([_scalar(benchmark["resistance_rawc"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "raws", energy.raws, torch.tensor([_scalar(benchmark["resistance_raws"])], device=device, dtype=dtype))
    _record(report, "energy_balance", "ustar", energy.ustar, torch.tensor([_scalar(benchmark["resistance_ustar"])], device=device, dtype=dtype))

    _print_report(report)

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote JSON report to {args.report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
