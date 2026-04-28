from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import warnings
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from scope.biochem import LeafBiochemistryInputs, LeafBiochemistryModel, LeafMeteo
from scope.canopy import (
    CanopyFluorescenceModel,
    CanopyReflectanceModel,
    CanopyThermalRadianceModel,
    FourSAILModel,
    campbell_lidf,
)
from scope.spectral.fluspect import FluspectModel, LeafBioBatch, OptiPar, SpectralGrids

DEFAULT_KERNELS = ("fluspect", "reflectance", "fluorescence", "thermal", "leaf_biochemistry")
DEFAULT_NLAYERS = 4


@dataclass(slots=True)
class BenchmarkContext:
    device: torch.device
    dtype: torch.dtype
    fluspect: FluspectModel
    reflectance: CanopyReflectanceModel
    fluorescence: CanopyFluorescenceModel
    thermal: CanopyThermalRadianceModel
    leaf_biochemistry: LeafBiochemistryModel
    leafbio: LeafBioBatch
    soil: torch.Tensor
    lai: torch.Tensor
    tts: torch.Tensor
    tto: torch.Tensor
    psi: torch.Tensor
    Esun_optical: torch.Tensor
    Esky_optical: torch.Tensor
    excitation_sun: torch.Tensor
    excitation_sky: torch.Tensor
    etau: torch.Tensor
    etah: torch.Tensor
    Tcu: torch.Tensor
    Tch: torch.Tensor
    Tsu: torch.Tensor
    Tsh: torch.Tensor
    leaf_inputs: LeafBiochemistryInputs
    leaf_meteo: LeafMeteo
    fV: torch.Tensor
    nlayers: int
    fixture: str


def _scope_root() -> Path | None:
    candidate = REPO_ROOT / "upstream" / "SCOPE"
    return candidate if candidate.exists() else None


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}") from exc


def _spectral(device: torch.device, dtype: torch.dtype) -> SpectralGrids:
    wlP = torch.linspace(400.0, 700.0, 256, device=device, dtype=dtype)
    wlF = torch.linspace(640.0, 850.0, 64, device=device, dtype=dtype)
    wlE = torch.linspace(400.0, 750.0, 64, device=device, dtype=dtype)
    return SpectralGrids(wlP=wlP, wlF=wlF, wlE=wlE)


def _optipar(spectral: SpectralGrids) -> OptiPar:
    wl = spectral.wlP
    base = torch.linspace(0.0, 1.0, wl.numel(), device=wl.device, dtype=wl.dtype)
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


def _sequence(start: float, stop: float, batch: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.linspace(start, stop, batch, device=device, dtype=dtype)


def _leafbio(batch: int, *, device: torch.device, dtype: torch.dtype) -> LeafBioBatch:
    return LeafBioBatch(
        Cab=_sequence(35.0, 60.0, batch, device=device, dtype=dtype),
        Cw=_sequence(0.009, 0.018, batch, device=device, dtype=dtype),
        Cdm=_sequence(0.010, 0.018, batch, device=device, dtype=dtype),
        fqe=_sequence(0.010, 0.018, batch, device=device, dtype=dtype),
    )


def _soil(batch: int, nwl: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    base = torch.linspace(0.10, 0.24, nwl, device=device, dtype=dtype)
    scale = _sequence(0.95, 1.05, batch, device=device, dtype=dtype).unsqueeze(-1)
    return (base.unsqueeze(0) * scale).clamp(max=0.95)


def _profile(batch: int, ncols: int, start: float, stop: float, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.linspace(start, stop, batch * ncols, device=device, dtype=dtype).reshape(batch, ncols)


def build_context(*, batch: int, device: torch.device, dtype: torch.dtype, fixture: str, nlayers: int) -> BenchmarkContext:
    lidf = campbell_lidf(57.0, device=device, dtype=dtype)
    sail = FourSAILModel(lidf=lidf)

    scope_root = _scope_root()
    resolved_fixture = fixture
    if fixture == "auto":
        resolved_fixture = "scope-assets" if scope_root is not None else "synthetic"

    if resolved_fixture == "scope-assets":
        if scope_root is None:
            raise FileNotFoundError(f"Could not find SCOPE root at {REPO_ROOT / 'upstream' / 'SCOPE'}")
        reflectance = CanopyReflectanceModel.from_scope_assets(
            lidf=lidf,
            sail=sail,
            scope_root_path=str(scope_root),
            device=device,
            dtype=dtype,
        )
        fluspect = reflectance.fluspect
        soil = reflectance.soil_reflectance(soil_spectrum=torch.ones(batch, device=device, dtype=dtype))
    elif resolved_fixture == "synthetic":
        spectral = _spectral(device, dtype)
        fluspect = FluspectModel(spectral, _optipar(spectral), device=device, dtype=dtype)
        reflectance = CanopyReflectanceModel(fluspect, sail, lidf=lidf)
        soil = _soil(batch, fluspect.spectral.wlP.numel(), device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported fixture {fixture!r}")

    fluorescence = CanopyFluorescenceModel(reflectance)
    thermal = CanopyThermalRadianceModel(reflectance)
    leaf_biochemistry = LeafBiochemistryModel(device=device, dtype=dtype)

    lai = _sequence(2.0, 5.0, batch, device=device, dtype=dtype)
    tts = _sequence(25.0, 45.0, batch, device=device, dtype=dtype)
    tto = _sequence(15.0, 35.0, batch, device=device, dtype=dtype)
    psi = _sequence(5.0, 65.0, batch, device=device, dtype=dtype)

    nwl_p = fluspect.spectral.wlP.numel()
    nwl_e = fluorescence._rtmf_fluspect.spectral.wlE.numel()

    Esun_optical = _profile(batch, nwl_p, 900.0, 1100.0, device=device, dtype=dtype)
    Esky_optical = _profile(batch, nwl_p, 120.0, 180.0, device=device, dtype=dtype)
    excitation_sun = _profile(batch, nwl_e, 1.0, 1.8, device=device, dtype=dtype)
    excitation_sky = _profile(batch, nwl_e, 0.2, 0.5, device=device, dtype=dtype)
    etau = _profile(batch, nlayers, 0.010, 0.018, device=device, dtype=dtype)
    etah = _profile(batch, nlayers, 0.008, 0.016, device=device, dtype=dtype)

    Tcu = _profile(batch, nlayers, 24.5, 27.5, device=device, dtype=dtype)
    Tch = _profile(batch, nlayers, 22.5, 25.0, device=device, dtype=dtype)
    Tsu = _sequence(26.0, 30.0, batch, device=device, dtype=dtype)
    Tsh = _sequence(21.0, 24.0, batch, device=device, dtype=dtype)

    leaf_inputs = LeafBiochemistryInputs(
        Vcmax25=_sequence(55.0, 80.0, batch, device=device, dtype=dtype),
        BallBerrySlope=_sequence(7.0, 10.0, batch, device=device, dtype=dtype),
        BallBerry0=torch.full((batch,), 0.01, device=device, dtype=dtype),
        g_m=_sequence(0.10, 0.14, batch, device=device, dtype=dtype),
    )
    leaf_meteo = LeafMeteo(
        Q=_sequence(900.0, 1500.0, batch, device=device, dtype=dtype),
        Cs=_sequence(385.0, 410.0, batch, device=device, dtype=dtype),
        T=_sequence(22.0, 30.0, batch, device=device, dtype=dtype),
        eb=_sequence(14.0, 24.0, batch, device=device, dtype=dtype),
        Oa=torch.full((batch,), 209.0, device=device, dtype=dtype),
        p=_sequence(960.0, 985.0, batch, device=device, dtype=dtype),
    )
    fV = _sequence(1.0, 0.9, batch, device=device, dtype=dtype)

    return BenchmarkContext(
        device=device,
        dtype=dtype,
        fluspect=fluspect,
        reflectance=reflectance,
        fluorescence=fluorescence,
        thermal=thermal,
        leaf_biochemistry=leaf_biochemistry,
        leafbio=_leafbio(batch, device=device, dtype=dtype),
        soil=soil,
        lai=lai,
        tts=tts,
        tto=tto,
        psi=psi,
        Esun_optical=Esun_optical,
        Esky_optical=Esky_optical,
        excitation_sun=excitation_sun,
        excitation_sky=excitation_sky,
        etau=etau,
        etah=etah,
        Tcu=Tcu,
        Tch=Tch,
        Tsu=Tsu,
        Tsh=Tsh,
        leaf_inputs=leaf_inputs,
        leaf_meteo=leaf_meteo,
        fV=fV,
        nlayers=nlayers,
        fixture=resolved_fixture,
    )


def make_kernel_map(context: BenchmarkContext) -> dict[str, Callable[[], torch.Tensor]]:
    return {
        "fluspect": lambda: context.fluspect(context.leafbio).refl,
        "reflectance": lambda: context.reflectance(
            context.leafbio,
            context.soil,
            context.lai,
            context.tts,
            context.tto,
            context.psi,
            nlayers=context.nlayers,
        ).rsot,
        "fluorescence": lambda: context.fluorescence.layered(
            context.leafbio,
            context.soil,
            context.lai,
            context.tts,
            context.tto,
            context.psi,
            context.excitation_sun,
            context.excitation_sky,
            etau=context.etau,
            etah=context.etah,
            nlayers=context.nlayers,
        ).LoF_,
        "thermal": lambda: context.thermal(
            context.lai,
            context.tts,
            context.tto,
            context.psi,
            context.Tcu,
            context.Tch,
            context.Tsu,
            context.Tsh,
            nlayers=context.nlayers,
        ).Lot_,
        "leaf_biochemistry": lambda: context.leaf_biochemistry(
            context.leaf_inputs,
            context.leaf_meteo,
            fV=context.fV,
        ).A,
    }


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _benchmark_callable(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iters: int,
    device: torch.device,
) -> dict[str, float]:
    for _ in range(warmup):
        _ = fn()
        _sync(device)

    samples: list[float] = []
    for _ in range(iters):
        started = time.perf_counter()
        _ = fn()
        _sync(device)
        samples.append(time.perf_counter() - started)

    return {
        "median_seconds": statistics.median(samples),
        "mean_seconds": statistics.fmean(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
        "samples": samples,
    }


def _compile_supported() -> bool:
    return hasattr(torch, "compile")


def benchmark_kernel(
    name: str,
    fn: Callable[[], torch.Tensor],
    *,
    mode: str,
    warmup: int,
    iters: int,
    device: torch.device,
    compile_mode: str,
) -> dict[str, object]:
    result: dict[str, object] = {
        "kernel": name,
        "mode": mode,
        "compile_supported": _compile_supported(),
    }
    eager = _benchmark_callable(fn, warmup=warmup, iters=iters, device=device)
    result["eager"] = eager
    if mode == "eager":
        return result

    if not _compile_supported():
        result["compiled"] = None
        result["compile_viable"] = False
        result["compile_reason"] = "torch.compile is not available in this environment"
        return result

    with suppress(Exception):
        torch._dynamo.reset()  # type: ignore[attr-defined]

    try:
        compiled_fn = torch.compile(fn, mode=compile_mode)
        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            started = time.perf_counter()
            _ = compiled_fn()
            _sync(device)
            first_call_seconds = time.perf_counter() - started

        compiled = _benchmark_callable(compiled_fn, warmup=warmup, iters=iters, device=device)
    except Exception as exc:
        result["compiled"] = None
        result["warnings"] = []
        result["compile_viable"] = False
        result["compile_reason"] = f"torch.compile failed: {type(exc).__name__}: {exc}"
        return result

    compiled["first_call_seconds"] = first_call_seconds
    compiled["compile_overhead_seconds"] = max(first_call_seconds - compiled["median_seconds"], 0.0)

    speedup = eager["median_seconds"] / compiled["median_seconds"] if compiled["median_seconds"] > 0 else None
    per_call_saving = eager["median_seconds"] - compiled["median_seconds"]
    if per_call_saving > 0:
        break_even_calls = compiled["compile_overhead_seconds"] / per_call_saving
    else:
        break_even_calls = None

    warning_messages = sorted({str(item.message) for item in recorded})
    result["compiled"] = compiled
    result["speedup"] = speedup
    result["break_even_calls"] = break_even_calls
    result["warnings"] = warning_messages
    result["compile_viable"] = bool(speedup is not None and speedup > 1.0 and break_even_calls is not None)
    if break_even_calls is not None:
        result["compile_reason"] = f"Steady-state speedup {speedup:.2f}x with ~{break_even_calls:.0f} calls to amortize compile overhead."
    elif speedup is not None and speedup <= 1.0:
        result["compile_reason"] = "Compiled execution was not faster than eager execution."
    else:
        result["compile_reason"] = "Unable to estimate compile break-even for this kernel."
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark core SCOPE kernels in eager or torch.compile mode.")
    parser.add_argument("--device", default="cpu", help="Execution device, e.g. cpu or cuda.")
    parser.add_argument("--dtype", default="float64", choices=("float32", "float64"))
    parser.add_argument("--batch", type=int, default=32, help="Batch size for synthetic benchmark inputs.")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations before timing.")
    parser.add_argument("--iters", type=int, default=10, help="Timed iterations.")
    parser.add_argument("--mode", choices=("eager", "compare"), default="compare", help="Whether to benchmark eager only or eager vs compiled.")
    parser.add_argument("--compile-mode", default="default", help="torch.compile mode, e.g. default or reduce-overhead.")
    parser.add_argument("--kernels", default=",".join(DEFAULT_KERNELS), help=f"Comma-separated kernels from: {', '.join(DEFAULT_KERNELS)}")
    parser.add_argument("--fixture", choices=("auto", "synthetic", "scope-assets"), default="auto", help="Model fixture source.")
    parser.add_argument("--nlayers", type=int, default=DEFAULT_NLAYERS)
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args(argv)


def run_benchmarks(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    dtype = _dtype_from_name(args.dtype)
    previous_grad_state = torch.is_grad_enabled()
    torch.set_grad_enabled(False)
    try:
        context = build_context(batch=args.batch, device=device, dtype=dtype, fixture=args.fixture, nlayers=args.nlayers)
        kernel_map = make_kernel_map(context)
        kernel_names = [name.strip() for name in args.kernels.split(",") if name.strip()]
        unknown = sorted(set(kernel_names) - set(kernel_map))
        if unknown:
            raise ValueError(f"Unknown kernels: {', '.join(unknown)}")

        results = {
            name: benchmark_kernel(
                name,
                kernel_map[name],
                mode=args.mode,
                warmup=args.warmup,
                iters=args.iters,
                device=device,
                compile_mode=args.compile_mode,
            )
            for name in kernel_names
        }
    finally:
        torch.set_grad_enabled(previous_grad_state)
    return {
        "fixture": context.fixture,
        "device": str(device),
        "dtype": args.dtype,
        "batch": args.batch,
        "warmup": args.warmup,
        "iters": args.iters,
        "mode": args.mode,
        "compile_mode": args.compile_mode,
        "nlayers": args.nlayers,
        "torch_version": torch.__version__,
        "results": results,
    }


def main(argv: list[str] | None = None) -> dict[str, object]:
    args = parse_args(argv)
    report = run_benchmarks(args)
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return report


if __name__ == "__main__":
    main()
