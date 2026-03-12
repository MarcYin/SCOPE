import torch

from scope_torch.energy import (
    HeatFluxInputs,
    ResistanceInputs,
    aerodynamic_resistances,
    heat_fluxes,
    saturated_vapor_pressure,
    slope_saturated_vapor_pressure,
)


def _reference_resistances(inputs: ResistanceInputs) -> dict[str, torch.Tensor]:
    kappa = 0.4
    LAI = torch.as_tensor(inputs.LAI, dtype=torch.float64)
    Cd = torch.as_tensor(inputs.Cd, dtype=torch.float64)
    rwc = torch.as_tensor(inputs.rwc, dtype=torch.float64)
    z0m = torch.as_tensor(inputs.z0m, dtype=torch.float64)
    d = torch.as_tensor(inputs.d, dtype=torch.float64)
    h = torch.as_tensor(inputs.h, dtype=torch.float64)
    z = torch.as_tensor(inputs.z, dtype=torch.float64)
    u = torch.maximum(torch.as_tensor(inputs.u, dtype=torch.float64), torch.tensor(0.3, dtype=torch.float64))
    L = torch.as_tensor(inputs.L, dtype=torch.float64)
    rbs = torch.as_tensor(inputs.rbs, dtype=torch.float64)

    zr = 2.5 * h
    n = Cd * LAI / (2 * kappa**2)
    unst = (L < 0) & (L > -500)
    st = (L > 0) & (L < 500)
    x = torch.ones_like(L)
    if bool(unst.item()):
        x = (1 - 16 * z / L) ** 0.25

    def psim(zz):
        if bool(unst.item()):
            return 2 * torch.log((1 + x) / 2) + torch.log((1 + x**2) / 2) - 2 * torch.atan(x) + torch.pi / 2
        if bool(st.item()):
            return -5 * zz / L
        return torch.tensor(0.0, dtype=torch.float64)

    def psih(zz, xx):
        if bool(unst.item()):
            return 2 * torch.log((1 + xx**2) / 2)
        if bool(st.item()):
            return -5 * zz / L
        return torch.tensor(0.0, dtype=torch.float64)

    def phstar(zz, xx):
        if bool(unst.item()):
            return (zz - d) / (zr - d) * (xx**2 - 1) / (xx**2 + 1)
        if bool(st.item()):
            return -5 * zz / L
        return torch.tensor(0.0, dtype=torch.float64)

    pm_z = psim(z - d)
    ph_z = psih(z - d, x)
    pm_h = psim(h - d)
    ph_zr = psih(zr - d, x) if bool((z >= zr).item()) else ph_z
    phs_zr = phstar(zr, x)
    phs_h = phstar(h, x)

    ustar = max(0.001, float(kappa * u / (torch.log((z - d) / z0m) - pm_z)))
    ustar = torch.tensor(ustar, dtype=torch.float64)
    Kh_base = kappa * ustar * (zr - d)
    if bool(unst.item()):
        Kh = Kh_base * (1 - 16 * (h - d) / L) ** 0.5
    elif bool(st.item()):
        Kh = Kh_base / (1 + 5 * (h - d) / L)
    else:
        Kh = Kh_base
    uh = max(float(ustar / kappa * (torch.log((h - d) / z0m) - pm_h)), 0.01)
    uh = torch.tensor(uh, dtype=torch.float64)
    uz0 = uh * torch.exp(n * ((z0m + d) / h - 1))

    rai = ((torch.log((z - d) / (zr - d)) - ph_z + ph_zr) / (kappa * ustar)) if bool((z > zr).item()) else torch.tensor(0.0, dtype=torch.float64)
    rar = 1 / (kappa * ustar) * ((zr - h) / (zr - d)) - phs_zr + phs_h
    rac = h * torch.sinh(n) / (n * Kh_base) * (
        torch.log((torch.exp(n) - 1) / (torch.exp(n) + 1))
        - torch.log((torch.exp(n * (z0m + d) / h) - 1) / (torch.exp(n * (z0m + d) / h) + 1))
    )
    rws = h * torch.sinh(n) / (n * Kh_base) * (
        torch.log((torch.exp(n * (z0m + d) / h) - 1) / (torch.exp(n * (z0m + d) / h) + 1))
        - torch.log((torch.exp(n * 0.01 / h) - 1) / (torch.exp(n * 0.01 / h) + 1))
    )
    return {
        "ustar": ustar,
        "Kh": Kh,
        "uz0": uz0,
        "raa": rai + rar + rac,
        "rawc": rwc,
        "raws": rws + rbs,
    }


def test_aerodynamic_resistances_match_reference_equations():
    inputs = ResistanceInputs(
        LAI=torch.tensor(3.0, dtype=torch.float64),
        Cd=torch.tensor(0.2, dtype=torch.float64),
        rwc=torch.tensor(20.0, dtype=torch.float64),
        z0m=torch.tensor(0.1, dtype=torch.float64),
        d=torch.tensor(0.7, dtype=torch.float64),
        h=torch.tensor(2.0, dtype=torch.float64),
        z=torch.tensor(10.0, dtype=torch.float64),
        u=torch.tensor(2.5, dtype=torch.float64),
        L=torch.tensor(-100.0, dtype=torch.float64),
        rbs=torch.tensor(15.0, dtype=torch.float64),
    )
    result = aerodynamic_resistances(inputs)
    ref = _reference_resistances(inputs)

    assert torch.allclose(result.ustar, ref["ustar"], atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.Kh, ref["Kh"], atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.uz0, ref["uz0"], atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.raa, ref["raa"], atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.rawc, ref["rawc"], atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.raws, ref["raws"], atol=1e-12, rtol=1e-10)


def test_heat_fluxes_match_reference_equations():
    inputs = HeatFluxInputs(
        ra=torch.tensor([100.0], dtype=torch.float64),
        rs=torch.tensor([200.0], dtype=torch.float64),
        Tc=torch.tensor([25.0], dtype=torch.float64),
        ea=torch.tensor([15.0], dtype=torch.float64),
        Ta=torch.tensor([20.0], dtype=torch.float64),
        e_to_q=torch.tensor([18.0 / 28.96 / 970.0], dtype=torch.float64),
        Ca=torch.tensor([380.0], dtype=torch.float64),
        Ci=torch.tensor([280.0], dtype=torch.float64),
    )
    result = heat_fluxes(inputs)

    ei = saturated_vapor_pressure(inputs.Tc).to(torch.float64)
    slope = slope_saturated_vapor_pressure(ei, inputs.Tc).to(torch.float64)
    lambda_evap = (2.501 - 0.002361 * inputs.Tc) * 1e6
    qi = ei * inputs.e_to_q
    qa = inputs.ea * inputs.e_to_q
    expected_latent = 1.2047 / (inputs.ra + inputs.rs) * lambda_evap * (qi - qa)
    expected_sensible = (1.2047 * 1004.0) / inputs.ra * (inputs.Tc - inputs.Ta)
    expected_ec = inputs.ea + (ei - inputs.ea) * inputs.ra / (inputs.ra + inputs.rs)
    expected_Cc = inputs.Ca - (inputs.Ca - inputs.Ci) * inputs.ra / (inputs.ra + inputs.rs)

    assert torch.allclose(result.lambda_evap, lambda_evap, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.slope_satvap, slope, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.latent_heat, expected_latent, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.sensible_heat, expected_sensible, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.ec, expected_ec, atol=1e-12, rtol=1e-10)
    assert torch.allclose(result.Cc, expected_Cc, atol=1e-12, rtol=1e-10)
