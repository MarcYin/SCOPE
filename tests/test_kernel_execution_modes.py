from __future__ import annotations

from dataclasses import fields, is_dataclass
from functools import lru_cache

import pytest
import torch

from scope_torch.biochem import LeafBiochemistryInputs, LeafBiochemistryModel, LeafMeteo
from scope_torch.canopy.fluorescence import CanopyFluorescenceModel
from scope_torch.canopy.foursail import FourSAILModel, campbell_lidf
from scope_torch.canopy.reflectance import CanopyReflectanceModel
from scope_torch.canopy.thermal import CanopyThermalRadianceModel
from scope_torch.spectral.fluspect import LeafBioBatch


def _index_value(value, index: int):
    if is_dataclass(value):
        kwargs = {field.name: _index_value(getattr(value, field.name), index) for field in fields(value)}
        return type(value)(**kwargs)
    if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] > index:
        return value[index : index + 1]
    return value


def _concat_output_maps(parts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = parts[0].keys()
    return {key: torch.cat([part[key] for part in parts], dim=0) for key in keys}


def _index_mapping(mapping: dict[str, torch.Tensor], index: int) -> dict[str, torch.Tensor]:
    return {key: _index_value(value, index) for key, value in mapping.items()}


def _assert_tensor_mapping_close(
    actual: dict[str, torch.Tensor],
    expected: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> None:
    assert set(actual) == set(expected)
    for key in actual:
        lhs = actual[key].detach().cpu()
        rhs = expected[key].detach().cpu()
        assert lhs.shape == rhs.shape, key
        assert torch.allclose(lhs.to(torch.float64), rhs.to(torch.float64), atol=atol, rtol=rtol), key


@lru_cache(maxsize=None)
def _build_models(device_type: str, dtype: torch.dtype):
    device = torch.device(device_type)
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)
    reflectance = CanopyReflectanceModel.from_scope_assets(lidf=lidf, sail=sail, device=device, dtype=dtype)
    fluorescence = CanopyFluorescenceModel(reflectance)
    thermal = CanopyThermalRadianceModel(reflectance)
    leaf_biochemistry = LeafBiochemistryModel(device=device, dtype=dtype)
    return {
        "reflectance": reflectance,
        "fluorescence": fluorescence,
        "thermal": thermal,
        "leaf_biochemistry": leaf_biochemistry,
    }


def _leafbio(device: torch.device, dtype: torch.dtype) -> LeafBioBatch:
    return LeafBioBatch(
        Cab=torch.tensor([45.0, 40.0], device=device, dtype=dtype),
        Cw=torch.tensor([0.010, 0.014], device=device, dtype=dtype),
        Cdm=torch.tensor([0.012, 0.017], device=device, dtype=dtype),
        fqe=torch.tensor([0.010, 0.015], device=device, dtype=dtype),
    )


def _canopy_geometry(device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    return {
        "lai": torch.tensor([2.8, 3.4], device=device, dtype=dtype),
        "tts": torch.tensor([30.0, 36.0], device=device, dtype=dtype),
        "tto": torch.tensor([20.0, 26.0], device=device, dtype=dtype),
        "psi": torch.tensor([10.0, 24.0], device=device, dtype=dtype),
    }


def _soil_spectrum(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor([1.0, 2.0], device=device, dtype=dtype)


def _run_fluspect_case(device_type: str, dtype: torch.dtype, index: int | None = None) -> dict[str, torch.Tensor]:
    models = _build_models(device_type, dtype)
    reflectance = models["reflectance"]
    leafbio = _leafbio(reflectance.fluspect.device, reflectance.fluspect.dtype)
    if index is not None:
        leafbio = _index_value(leafbio, index)
    leafopt = reflectance.fluspect(leafbio)
    assert leafopt.Mb is not None
    assert leafopt.Mf is not None
    return {
        "refl": leafopt.refl,
        "tran": leafopt.tran,
        "Mb": leafopt.Mb,
        "Mf": leafopt.Mf,
    }


def _run_reflectance_profile_case(device_type: str, dtype: torch.dtype, index: int | None = None) -> dict[str, torch.Tensor]:
    models = _build_models(device_type, dtype)
    reflectance = models["reflectance"]
    device = reflectance.fluspect.device
    geom = _canopy_geometry(device, reflectance.fluspect.dtype)
    soil = reflectance.soil_reflectance(soil_spectrum=_soil_spectrum(device, reflectance.fluspect.dtype))
    nwl = reflectance.fluspect.spectral.wlP.numel()
    Esun = torch.linspace(900.0, 1100.0, 2 * nwl, device=device, dtype=reflectance.fluspect.dtype).reshape(2, nwl)
    Esky = torch.linspace(120.0, 180.0, 2 * nwl, device=device, dtype=reflectance.fluspect.dtype).reshape(2, nwl)
    leafbio = _leafbio(device, reflectance.fluspect.dtype)
    if index is not None:
        leafbio = _index_value(leafbio, index)
        geom = _index_mapping(geom, index)
        soil = _index_value(soil, index)
        Esun = _index_value(Esun, index)
        Esky = _index_value(Esky, index)
    profiles = reflectance.profiles(
        leafbio,
        soil,
        geom["lai"],
        geom["tts"],
        geom["tto"],
        geom["psi"],
        Esun,
        Esky,
        nlayers=4,
    )
    return {
        "Ps": profiles.Ps,
        "Po": profiles.Po,
        "Pso": profiles.Pso,
        "Es_": profiles.Es_,
        "Emin_": profiles.Emin_,
        "Eplu_": profiles.Eplu_,
    }


def _run_fluorescence_profile_case(device_type: str, dtype: torch.dtype, index: int | None = None) -> dict[str, torch.Tensor]:
    models = _build_models(device_type, dtype)
    fluorescence = models["fluorescence"]
    reflectance = fluorescence.reflectance_model
    device = reflectance.fluspect.device
    geom = _canopy_geometry(device, reflectance.fluspect.dtype)
    soil = reflectance.soil_reflectance(soil_spectrum=_soil_spectrum(device, reflectance.fluspect.dtype))
    n_wle = fluorescence._rtmf_fluspect.spectral.wlE.numel()
    Esun = torch.linspace(1.0, 1.8, 2 * n_wle, device=device, dtype=reflectance.fluspect.dtype).reshape(2, n_wle)
    Esky = torch.linspace(0.2, 0.5, 2 * n_wle, device=device, dtype=reflectance.fluspect.dtype).reshape(2, n_wle)
    etau = torch.tensor([[0.010, 0.011, 0.012, 0.013], [0.014, 0.015, 0.016, 0.017]], device=device, dtype=reflectance.fluspect.dtype)
    etah = torch.tensor([[0.008, 0.009, 0.010, 0.011], [0.012, 0.013, 0.014, 0.015]], device=device, dtype=reflectance.fluspect.dtype)
    leafbio = _leafbio(device, reflectance.fluspect.dtype)
    if index is not None:
        leafbio = _index_value(leafbio, index)
        geom = _index_mapping(geom, index)
        soil = _index_value(soil, index)
        Esun = _index_value(Esun, index)
        Esky = _index_value(Esky, index)
        etau = _index_value(etau, index)
        etah = _index_value(etah, index)
    profiles = fluorescence.profiles(
        leafbio,
        soil,
        geom["lai"],
        geom["tts"],
        geom["tto"],
        geom["psi"],
        Esun,
        Esky,
        etau=etau,
        etah=etah,
        nlayers=4,
    )
    return {
        "Ps": profiles.Ps,
        "Po": profiles.Po,
        "Pso": profiles.Pso,
        "Fmin_": profiles.Fmin_,
        "Fplu_": profiles.Fplu_,
        "layer_fluorescence": profiles.layer_fluorescence,
    }


def _run_thermal_profile_case(device_type: str, dtype: torch.dtype, index: int | None = None) -> dict[str, torch.Tensor]:
    models = _build_models(device_type, dtype)
    thermal = models["thermal"]
    device = thermal.reflectance_model.fluspect.device
    geom = _canopy_geometry(device, thermal.reflectance_model.fluspect.dtype)
    Tcu = torch.tensor([[25.0, 25.4, 25.8, 26.2], [24.0, 24.4, 24.8, 25.2]], device=device, dtype=thermal.reflectance_model.fluspect.dtype)
    Tch = torch.tensor([[23.2, 23.5, 23.8, 24.1], [22.5, 22.8, 23.1, 23.4]], device=device, dtype=thermal.reflectance_model.fluspect.dtype)
    Tsu = torch.tensor([26.5, 27.0], device=device, dtype=thermal.reflectance_model.fluspect.dtype)
    Tsh = torch.tensor([22.0, 22.5], device=device, dtype=thermal.reflectance_model.fluspect.dtype)
    if index is not None:
        geom = _index_mapping(geom, index)
        Tcu = _index_value(Tcu, index)
        Tch = _index_value(Tch, index)
        Tsu = _index_value(Tsu, index)
        Tsh = _index_value(Tsh, index)
    profiles = thermal.profiles(
        geom["lai"],
        geom["tts"],
        geom["tto"],
        geom["psi"],
        Tcu,
        Tch,
        Tsu,
        Tsh,
        nlayers=4,
    )
    return {
        "Ps": profiles.Ps,
        "Po": profiles.Po,
        "Pso": profiles.Pso,
        "Emint_": profiles.Emint_,
        "Eplut_": profiles.Eplut_,
        "layer_thermal_upward": profiles.layer_thermal_upward,
    }


def _run_leaf_biochemistry_case(device_type: str, dtype: torch.dtype, index: int | None = None) -> dict[str, torch.Tensor]:
    models = _build_models(device_type, dtype)
    model = models["leaf_biochemistry"]
    fV = torch.tensor([1.0, 0.95], device=model.device, dtype=model.dtype)
    leafbio = LeafBiochemistryInputs(
        Vcmax25=torch.tensor([72.0, 60.0], device=model.device, dtype=model.dtype),
        BallBerrySlope=torch.tensor([9.0, 7.8], device=model.device, dtype=model.dtype),
        BallBerry0=torch.tensor([0.01, 0.01], device=model.device, dtype=model.dtype),
        g_m=torch.tensor([0.12, 0.10], device=model.device, dtype=model.dtype),
    )
    meteo = LeafMeteo(
        Q=torch.tensor([1200.0, 850.0], device=model.device, dtype=model.dtype),
        Cs=torch.tensor([390.0, 405.0], device=model.device, dtype=model.dtype),
        T=torch.tensor([25.0, 28.0], device=model.device, dtype=model.dtype),
        eb=torch.tensor([18.0, 21.0], device=model.device, dtype=model.dtype),
        Oa=torch.tensor([209.0, 209.0], device=model.device, dtype=model.dtype),
        p=torch.tensor([970.0, 965.0], device=model.device, dtype=model.dtype),
    )
    if index is not None:
        leafbio = _index_value(leafbio, index)
        meteo = _index_value(meteo, index)
        fV = _index_value(fV, index)
    result = model(leafbio, meteo, fV=fV)
    return {
        name: getattr(result, name)
        for name in result.__dataclass_fields__
        if name != "fcount"
    }


KERNEL_CASES = {
    "fluspect": (_run_fluspect_case, 1e-5, 5e-5),
    "reflectance_profiles": (_run_reflectance_profile_case, 4e-3, 1e-4),
    "fluorescence_profiles": (_run_fluorescence_profile_case, 3e-5, 1.5e-4),
    "thermal_profiles": (_run_thermal_profile_case, 3e-5, 1.5e-4),
    "leaf_biochemistry": (_run_leaf_biochemistry_case, 2e-5, 1e-4),
}


@pytest.mark.parametrize("kernel", list(KERNEL_CASES))
def test_kernel_batch_outputs_match_single_scene_solves(kernel: str):
    run_case, _, _ = KERNEL_CASES[kernel]
    batched = run_case("cpu", torch.float64)
    singles = [run_case("cpu", torch.float64, index) for index in range(2)]
    _assert_tensor_mapping_close(batched, _concat_output_maps(singles), atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("kernel", list(KERNEL_CASES))
def test_kernel_outputs_float32_match_float64(kernel: str):
    run_case, atol, rtol = KERNEL_CASES[kernel]
    float32_outputs = run_case("cpu", torch.float32)
    float64_outputs = run_case("cpu", torch.float64)
    _assert_tensor_mapping_close(float32_outputs, float64_outputs, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("kernel", list(KERNEL_CASES))
def test_kernel_outputs_cpu_match_cuda(kernel: str):
    run_case, atol, rtol = KERNEL_CASES[kernel]
    cpu_outputs = run_case("cpu", torch.float32)
    cuda_outputs = run_case("cuda", torch.float32)
    _assert_tensor_mapping_close(cpu_outputs, cuda_outputs, atol=atol, rtol=rtol)
