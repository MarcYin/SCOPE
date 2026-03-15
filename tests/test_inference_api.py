from __future__ import annotations

from dataclasses import dataclass

import torch

from scope import ScopeInferenceModel, select_inference_outputs
from scope.canopy.foursail import FourSAILModel, campbell_lidf
from scope.spectral.fluspect import FluspectModel, LeafBioBatch, OptiPar, SpectralGrids


def _spectral(device, dtype):
    wlP = torch.linspace(400.0, 700.0, 32, device=device, dtype=dtype)
    wlF = torch.linspace(640.0, 740.0, 16, device=device, dtype=dtype)
    wlE = torch.linspace(400.0, 700.0, 16, device=device, dtype=dtype)
    return SpectralGrids(wlP=wlP, wlF=wlF, wlE=wlE)


def _optipar(spectral):
    wl = spectral.wlP
    base = torch.linspace(0.0, 1.0, wl.numel(), dtype=wl.dtype, device=wl.device)
    return OptiPar(
        nr=1.4 + 0.05 * torch.sin(base),
        Kab=0.01 + 0.005 * torch.cos(base),
        Kca=0.008 + 0.003 * torch.sin(base * 2.0),
        KcaV=0.008 + 0.003 * torch.sin(base * 2.0) * 0.95,
        KcaZ=0.008 + 0.003 * torch.sin(base * 2.0) * 1.05,
        Kdm=0.005 + 0.002 * torch.cos(base * 3.0),
        Kw=0.002 + 0.001 * torch.sin(base * 4.0),
        Ks=0.001 + 0.0005 * torch.cos(base * 5.0),
        Kant=0.0002 + 0.0001 * torch.sin(base * 6.0),
        phi=torch.full_like(wl, 0.5),
    )


def _build_inference_model(*, device: str = "cpu", dtype: torch.dtype = torch.float64) -> ScopeInferenceModel:
    resolved_device = torch.device(device)
    spectral = _spectral(resolved_device, dtype)
    optipar = _optipar(spectral)
    fluspect = FluspectModel(spectral, optipar, dtype=dtype)
    lidf = campbell_lidf(57.0, device=resolved_device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)
    return ScopeInferenceModel(fluspect, sail, lidf=lidf)


def test_scope_inference_reflectance_returns_requested_outputs_only():
    inference = _build_inference_model()
    device = inference.fluspect.device
    dtype = inference.fluspect.dtype

    outputs = inference.reflectance(
        LeafBioBatch(
            Cab=torch.tensor([45.0, 40.0], device=device, dtype=dtype),
            Cw=torch.tensor([0.01, 0.012], device=device, dtype=dtype),
            Cdm=torch.tensor([0.012, 0.010], device=device, dtype=dtype),
        ),
        lai=torch.tensor([2.8, 3.4], device=device, dtype=dtype),
        tts=torch.tensor([30.0, 36.0], device=device, dtype=dtype),
        tto=torch.tensor([20.0, 26.0], device=device, dtype=dtype),
        psi=torch.tensor([10.0, 24.0], device=device, dtype=dtype),
        soil_refl=torch.full((2, inference.fluspect.spectral.wlP.numel()), 0.2, device=device, dtype=dtype),
        nlayers=4,
        outputs=("rsot", "rso"),
    )

    assert set(outputs) == {"rsot", "rso"}
    assert outputs["rsot"].shape == (2, inference.fluspect.spectral.wlP.numel())
    assert outputs["rso"].shape == outputs["rsot"].shape


def test_select_inference_outputs_supports_nested_paths():
    @dataclass
    class Inner:
        value: torch.Tensor

    @dataclass
    class Outer:
        energy: Inner
        thermal: Inner

    result = Outer(
        energy=Inner(value=torch.tensor([1.0])),
        thermal=Inner(value=torch.tensor([2.0])),
    )

    selected = select_inference_outputs(result, ("energy.value", "thermal.value"))

    assert set(selected) == {"energy.value", "thermal.value"}
    assert torch.equal(selected["energy.value"], torch.tensor([1.0]))
    assert torch.equal(selected["thermal.value"], torch.tensor([2.0]))


def test_energy_balance_thermal_forwards_forcing_and_output_wavelength_grids():
    inference = _build_inference_model()
    device = inference.fluspect.device
    dtype = inference.fluspect.dtype
    spectral = inference.fluspect.spectral

    soil_refl = torch.full((1, spectral.wlP.numel()), 0.2, device=device, dtype=dtype)
    inference.reflectance_model.soil_reflectance = lambda **_: soil_refl

    captured: dict[str, object] = {}
    sentinel = object()

    class DummyEnergyBalanceModel:
        def solve_thermal(self, *args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return sentinel

    inference.energy_balance_model = DummyEnergyBalanceModel()

    wlT_forcing = torch.linspace(9000.0, 12000.0, 4, device=device, dtype=dtype)
    wlT_output = torch.linspace(8000.0, 13000.0, 6, device=device, dtype=dtype)
    result = inference.energy_balance_thermal(
        LeafBioBatch(
            Cab=torch.tensor([45.0], device=device, dtype=dtype),
            Cw=torch.tensor([0.01], device=device, dtype=dtype),
            Cdm=torch.tensor([0.012], device=device, dtype=dtype),
        ),
        object(),
        lai=torch.tensor([2.8], device=device, dtype=dtype),
        tts=torch.tensor([30.0], device=device, dtype=dtype),
        tto=torch.tensor([20.0], device=device, dtype=dtype),
        psi=torch.tensor([10.0], device=device, dtype=dtype),
        Esun_sw=torch.ones((1, spectral.wlE.numel()), device=device, dtype=dtype),
        Esky_sw=torch.ones((1, spectral.wlE.numel()), device=device, dtype=dtype),
        meteo=object(),
        canopy=object(),
        soil=object(),
        soil_refl=soil_refl,
        wlT_forcing=wlT_forcing,
        wlT=wlT_output,
    )

    assert result is sentinel
    assert torch.equal(captured["kwargs"]["wlT_forcing"], wlT_forcing)
    assert torch.equal(captured["kwargs"]["wlT"], wlT_output)
