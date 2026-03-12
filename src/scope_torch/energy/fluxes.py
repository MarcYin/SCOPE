from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class ResistanceInputs:
    LAI: torch.Tensor | float
    Cd: torch.Tensor | float
    rwc: torch.Tensor | float
    z0m: torch.Tensor | float
    d: torch.Tensor | float
    h: torch.Tensor | float
    z: torch.Tensor | float
    u: torch.Tensor | float
    L: torch.Tensor | float
    rbs: torch.Tensor | float


@dataclass(slots=True)
class ResistanceResult:
    ustar: torch.Tensor
    Kh: torch.Tensor
    uz0: torch.Tensor
    rai: torch.Tensor
    rar: torch.Tensor
    rac: torch.Tensor
    rws: torch.Tensor
    raa: torch.Tensor
    rawc: torch.Tensor
    raws: torch.Tensor


@dataclass(slots=True)
class HeatFluxInputs:
    ra: torch.Tensor | float
    rs: torch.Tensor | float
    Tc: torch.Tensor | float
    ea: torch.Tensor | float
    Ta: torch.Tensor | float
    e_to_q: torch.Tensor | float
    Ca: torch.Tensor | float
    Ci: torch.Tensor | float


@dataclass(slots=True)
class HeatFluxResult:
    latent_heat: torch.Tensor
    sensible_heat: torch.Tensor
    ec: torch.Tensor
    Cc: torch.Tensor
    lambda_evap: torch.Tensor
    slope_satvap: torch.Tensor


def saturated_vapor_pressure(temperature_c: torch.Tensor | float) -> torch.Tensor:
    temp = torch.as_tensor(temperature_c)
    return 6.107 * torch.pow(torch.full_like(temp, 10.0), 7.5 * temp / (237.3 + temp))


def slope_saturated_vapor_pressure(es: torch.Tensor | float, temperature_c: torch.Tensor | float) -> torch.Tensor:
    es_tensor, temp = torch.broadcast_tensors(torch.as_tensor(es), torch.as_tensor(temperature_c))
    return es_tensor * 2.3026 * 7.5 * 237.3 / (237.3 + temp) ** 2


def aerodynamic_resistances(
    inputs: ResistanceInputs,
    *,
    kappa: float = 0.4,
) -> ResistanceResult:
    LAI, Cd, rwc, z0m, d, h, z, u, L, rbs = torch.broadcast_tensors(
        torch.as_tensor(inputs.LAI),
        torch.as_tensor(inputs.Cd),
        torch.as_tensor(inputs.rwc),
        torch.as_tensor(inputs.z0m),
        torch.as_tensor(inputs.d),
        torch.as_tensor(inputs.h),
        torch.as_tensor(inputs.z),
        torch.as_tensor(inputs.u),
        torch.as_tensor(inputs.L),
        torch.as_tensor(inputs.rbs),
    )
    dtype = torch.promote_types(LAI.dtype, torch.float32)
    device = LAI.device
    LAI = LAI.to(device=device, dtype=dtype)
    Cd = Cd.to(device=device, dtype=dtype)
    rwc = rwc.to(device=device, dtype=dtype)
    z0m = z0m.to(device=device, dtype=dtype)
    d = d.to(device=device, dtype=dtype)
    h = h.to(device=device, dtype=dtype)
    z = z.to(device=device, dtype=dtype)
    u = u.to(device=device, dtype=dtype)
    L = L.to(device=device, dtype=dtype)
    rbs = rbs.to(device=device, dtype=dtype)

    zr = 2.5 * h
    n = Cd * LAI / (2.0 * kappa**2)
    u = torch.maximum(u, torch.full_like(u, 0.3))

    unst = (L < 0.0) & (L > -500.0)
    st = (L > 0.0) & (L < 500.0)
    x = torch.where(
        unst,
        torch.pow((1.0 - 16.0 * z / L).clamp(min=1e-12), 0.25),
        torch.ones_like(L),
    )

    pm_z = _psim(z - d, L, unst, st, x)
    ph_z = _psih(z - d, L, unst, st, x)
    pm_h = _psim(h - d, L, unst, st, x)
    # Match upstream SCOPE `resistances.m`: the same unstable-stability `x`
    # derived from measurement height is reused in all correction terms.
    ph_zr = torch.where(z >= zr, _psih(zr - d, L, unst, st, x), ph_z)
    phs_zr = _phstar(zr, zr, d, L, st, unst, x)
    phs_h = _phstar(h, zr, d, L, st, unst, x)

    ustar = torch.maximum(
        torch.full_like(u, 0.001),
        kappa * u / (torch.log(((z - d) / z0m).clamp(min=1e-12)) - pm_z).clamp(min=1e-12),
    )
    Kh_base = kappa * ustar * (zr - d)
    Kh = torch.where(
        unst,
        Kh_base * torch.sqrt((1.0 - 16.0 * (h - d) / L).clamp(min=1e-12)),
        torch.where(st, Kh_base / (1.0 + 5.0 * (h - d) / L), Kh_base),
    )

    uh = torch.maximum(
        ustar / kappa * (torch.log(((h - d) / z0m).clamp(min=1e-12)) - pm_h),
        torch.full_like(ustar, 0.01),
    )
    uz0 = uh * torch.exp(n * (((z0m + d) / h) - 1.0))

    sinh_n = torch.sinh(n)
    denom = (n * Kh_base).clamp(min=1e-12)
    exp_top = torch.exp(n)
    exp_z0 = torch.exp(n * (z0m + d) / h)
    exp_soil = torch.exp(n * 0.01 / h)
    log_term_top = torch.log(((exp_top - 1.0) / (exp_top + 1.0)).clamp(min=1e-12))
    log_term_z0 = torch.log(((exp_z0 - 1.0) / (exp_z0 + 1.0)).clamp(min=1e-12))
    log_term_soil = torch.log(((exp_soil - 1.0) / (exp_soil + 1.0)).clamp(min=1e-12))

    rai = torch.where(
        z > zr,
        (torch.log(((z - d) / (zr - d)).clamp(min=1e-12)) - ph_z + ph_zr) / (kappa * ustar).clamp(min=1e-12),
        torch.zeros_like(ustar),
    )
    rar = ((zr - h) / (zr - d)) / (kappa * ustar).clamp(min=1e-12) - phs_zr + phs_h
    rac = h * sinh_n / denom * (log_term_top - log_term_z0)
    rws = h * sinh_n / denom * (log_term_z0 - log_term_soil)

    raa = rai + rar + rac
    rawc = rwc
    raws = rws + rbs
    return ResistanceResult(
        ustar=ustar,
        Kh=Kh,
        uz0=uz0,
        rai=rai,
        rar=rar,
        rac=rac,
        rws=rws,
        raa=raa,
        rawc=rawc,
        raws=raws,
    )


def heat_fluxes(
    inputs: HeatFluxInputs,
    *,
    rhoa: float = 1.2047,
    cp: float = 1004.0,
) -> HeatFluxResult:
    ra, rs, Tc, ea, Ta, e_to_q, Ca, Ci = torch.broadcast_tensors(
        torch.as_tensor(inputs.ra),
        torch.as_tensor(inputs.rs),
        torch.as_tensor(inputs.Tc),
        torch.as_tensor(inputs.ea),
        torch.as_tensor(inputs.Ta),
        torch.as_tensor(inputs.e_to_q),
        torch.as_tensor(inputs.Ca),
        torch.as_tensor(inputs.Ci),
    )
    dtype = torch.promote_types(ra.dtype, torch.float32)
    device = ra.device
    ra = ra.to(device=device, dtype=dtype)
    rs = rs.to(device=device, dtype=dtype)
    Tc = Tc.to(device=device, dtype=dtype)
    ea = ea.to(device=device, dtype=dtype)
    Ta = Ta.to(device=device, dtype=dtype)
    e_to_q = e_to_q.to(device=device, dtype=dtype)
    Ca = Ca.to(device=device, dtype=dtype)
    Ci = Ci.to(device=device, dtype=dtype)

    lambda_evap = (2.501 - 0.002361 * Tc) * 1e6
    ei = saturated_vapor_pressure(Tc).to(device=device, dtype=dtype)
    slope = slope_saturated_vapor_pressure(ei, Tc).to(device=device, dtype=dtype)
    qi = ei * e_to_q
    qa = ea * e_to_q
    total_resistance = (ra + rs).clamp(min=1e-12)

    latent = rhoa / total_resistance * lambda_evap * (qi - qa)
    sensible = (rhoa * cp) / ra.clamp(min=1e-12) * (Tc - Ta)
    ec = ea + (ei - ea) * ra / total_resistance
    Cc = Ca - (Ca - Ci) * ra / total_resistance
    return HeatFluxResult(
        latent_heat=latent,
        sensible_heat=sensible,
        ec=ec,
        Cc=Cc,
        lambda_evap=lambda_evap,
        slope_satvap=slope,
    )


def _psim(z: torch.Tensor, L: torch.Tensor, unst: torch.Tensor, st: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(z)
    out = torch.where(
        unst,
        2.0 * torch.log((1.0 + x) / 2.0) + torch.log((1.0 + x**2) / 2.0) - 2.0 * torch.atan(x) + torch.pi / 2.0,
        out,
    )
    out = torch.where(st, -5.0 * z / L, out)
    return out


def _psih(z: torch.Tensor, L: torch.Tensor, unst: torch.Tensor, st: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(z)
    out = torch.where(unst, 2.0 * torch.log((1.0 + x**2) / 2.0), out)
    out = torch.where(st, -5.0 * z / L, out)
    return out


def _phstar(z: torch.Tensor, zR: torch.Tensor, d: torch.Tensor, L: torch.Tensor, st: torch.Tensor, unst: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(z)
    out = torch.where(unst, (z - d) / (zR - d) * (x**2 - 1.0) / (x**2 + 1.0), out)
    out = torch.where(st, -5.0 * z / L, out)
    return out
