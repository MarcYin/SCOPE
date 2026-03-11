from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass(slots=True)
class BiochemicalTemperatureResponse:
    delHaV: float = 65330.0
    delSV: float = 485.0
    delHdV: float = 149250.0
    delHaJ: float = 43540.0
    delSJ: float = 495.0
    delHdJ: float = 152040.0
    delHaP: float = 53100.0
    delSP: float = 490.0
    delHdP: float = 150650.0
    delHaR: float = 46390.0
    delSR: float = 490.0
    delHdR: float = 150650.0
    delHaKc: float = 79430.0
    delHaKo: float = 36380.0
    delHaT: float = 37830.0
    Q10: float = 2.0
    s1: float = 0.3
    s2: float = 313.15
    s3: float = 0.2
    s4: float = 288.15
    s5: float = 1.3
    s6: float = 328.15


@dataclass(slots=True)
class LeafBiochemistryInputs:
    Vcmax25: torch.Tensor | float
    BallBerrySlope: torch.Tensor | float
    Type: str = "C3"
    BallBerry0: torch.Tensor | float = 0.01
    RdPerVcmax25: torch.Tensor | float = 0.015
    Kn0: torch.Tensor | float = 2.48
    Knalpha: torch.Tensor | float = 2.83
    Knbeta: torch.Tensor | float = 0.114
    stressfactor: torch.Tensor | float = 1.0
    g_m: Optional[torch.Tensor | float] = None
    TDP: BiochemicalTemperatureResponse = field(default_factory=BiochemicalTemperatureResponse)


@dataclass(slots=True)
class LeafMeteo:
    Q: torch.Tensor | float
    Cs: torch.Tensor | float
    T: torch.Tensor | float
    eb: torch.Tensor | float
    Oa: torch.Tensor | float
    p: torch.Tensor | float


@dataclass(slots=True)
class BiochemicalOptions:
    apply_T_corr: bool = True
    ci_tol: float = 1e-7
    max_iter: int = 100


@dataclass(slots=True)
class LeafBiochemistryResult:
    A: torch.Tensor
    Ag: torch.Tensor
    Ci: torch.Tensor
    Cc: torch.Tensor
    rcw: torch.Tensor
    gs: torch.Tensor
    RH: torch.Tensor
    Vcmax: torch.Tensor
    Rd: torch.Tensor
    Ja: torch.Tensor
    ps: torch.Tensor
    ps_rel: torch.Tensor
    Kd: torch.Tensor
    Kn: torch.Tensor
    NPQ: torch.Tensor
    Kf: torch.Tensor
    Kp0: torch.Tensor
    Kp: torch.Tensor
    eta: torch.Tensor
    qE: torch.Tensor
    fs: torch.Tensor
    ft: torch.Tensor
    SIF: torch.Tensor
    fo0: torch.Tensor
    fm0: torch.Tensor
    fo: torch.Tensor
    fm: torch.Tensor
    Fm_Fo: torch.Tensor
    Ft_Fo: torch.Tensor
    qQ: torch.Tensor
    Phi_N: torch.Tensor
    CO2_per_electron: torch.Tensor
    fcount: int


class LeafBiochemistryModel:
    """Leaf-level SCOPE biochemistry and fluorescence-yield model."""

    def __init__(
        self,
        *,
        device: Optional[torch.device | str] = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.rhoa = 1.2047
        self.Mair = 28.96
        self.R = 8.314
        self.Tref = 298.15
        self.Kc25 = 405e-6
        self.Ko25 = 279e-3
        self.spfy25 = 2444.0
        self.Kf = 0.05
        self.Kp = 4.0
        self.atheta = 0.8

    def __call__(
        self,
        leafbio: LeafBiochemistryInputs,
        meteo: LeafMeteo,
        *,
        options: Optional[BiochemicalOptions] = None,
        fV: torch.Tensor | float = 1.0,
    ) -> LeafBiochemistryResult:
        opts = options or BiochemicalOptions()
        canopy_type = self._normalize_type(leafbio.Type)
        batch = self._infer_batch(
            meteo.Q,
            meteo.Cs,
            meteo.T,
            meteo.eb,
            meteo.Oa,
            meteo.p,
            leafbio.Vcmax25,
            leafbio.BallBerrySlope,
            leafbio.BallBerry0,
            leafbio.RdPerVcmax25,
            leafbio.Kn0,
            leafbio.Knalpha,
            leafbio.Knbeta,
            leafbio.stressfactor,
            fV,
            leafbio.g_m,
        )

        Q = self._expand(meteo.Q, batch)
        Cs_ppm = self._expand(meteo.Cs, batch)
        T_in = self._expand(meteo.T, batch)
        T = T_in + 273.15 * (T_in < 200.0).to(dtype=self.dtype)
        eb = self._expand(meteo.eb, batch)
        Oa = self._expand(meteo.Oa, batch)
        p = self._expand(meteo.p, batch)

        fV_tensor = self._expand(fV, batch)
        Vcmax25 = fV_tensor * self._expand(leafbio.Vcmax25, batch)
        BallBerrySlope = self._expand(leafbio.BallBerrySlope, batch)
        BallBerry0 = self._expand(leafbio.BallBerry0, batch)
        RdPerVcmax25 = self._expand(leafbio.RdPerVcmax25, batch)
        Kn0 = self._expand(leafbio.Kn0, batch)
        Knalpha = self._expand(leafbio.Knalpha, batch)
        Knbeta = self._expand(leafbio.Knbeta, batch)
        stressfactor = self._expand(leafbio.stressfactor, batch)
        if leafbio.g_m is None:
            g_m = torch.full((batch,), torch.inf, device=self.device, dtype=self.dtype)
        else:
            g_m = self._expand(leafbio.g_m, batch) * 1e6

        ppm2bar = 1e-6 * (p * 1e-3)
        Cs = Cs_ppm * ppm2bar
        O = (Oa * 1e-3) * (p * 1e-3) if canopy_type == "C3" else torch.zeros_like(Cs)
        Gamma_star25 = 0.5 * O / self.spfy25
        Rd25 = RdPerVcmax25 * Vcmax25
        effcon = torch.full_like(Cs, 1.0 / 5.0 if canopy_type == "C3" else 1.0 / 6.0)

        Kd = torch.maximum(
            torch.full_like(T, 0.8738),
            0.0301 * (T - 273.15) + 0.0773,
        )

        temp = leafbio.TDP
        Vcmax = Vcmax25 * stressfactor
        Rd = Rd25 * stressfactor
        Kc = torch.full_like(Cs, self.Kc25)
        Ko = torch.full_like(Cs, self.Ko25)
        Gamma_star = Gamma_star25
        Ke = 20000.0 * Vcmax25 if canopy_type == "C4" else torch.ones_like(Cs)

        if opts.apply_T_corr:
            if canopy_type == "C4":
                q10_term = torch.pow(torch.full_like(T, temp.Q10), 0.1 * (T - self.Tref))
                fHTv = 1.0 + torch.exp(temp.s1 * (T - temp.s2))
                fLTv = 1.0 + torch.exp(temp.s3 * (temp.s4 - T))
                Vcmax = (Vcmax25 * q10_term) / (fHTv * fLTv)
                fHTv_rd = 1.0 + torch.exp(temp.s5 * (T - temp.s6))
                Rd = (Rd25 * q10_term) / fHTv_rd
                Ke = (20000.0 * Vcmax25) * q10_term
            else:
                f_vcmax = self._temperature_function_c3(T, temp.delHaV) * self._high_temp_inhibition_c3(T, temp.delSV, temp.delHdV)
                f_rd = self._temperature_function_c3(T, temp.delHaR) * self._high_temp_inhibition_c3(T, temp.delSR, temp.delHdR)
                f_kc = self._temperature_function_c3(T, temp.delHaKc)
                f_ko = self._temperature_function_c3(T, temp.delHaKo)
                f_gamma = self._temperature_function_c3(T, temp.delHaT)
                Vcmax = Vcmax25 * f_vcmax * stressfactor
                Rd = Rd25 * f_rd * stressfactor
                Kc = self.Kc25 * f_kc
                Ko = self.Ko25 * f_ko
                Gamma_star = Gamma_star25 * f_gamma

        po0 = self.Kp / (self.Kf + Kd + self.Kp)
        Je = 0.5 * po0 * Q
        if canopy_type == "C3":
            MM_consts = Kc * (1.0 + O / Ko)
            Vs_C3 = Vcmax / 2.0
            min_ci = 0.3
        else:
            MM_consts = torch.zeros_like(Cs)
            Vs_C3 = torch.zeros_like(Cs)
            min_ci = 0.1

        RH = torch.clamp(eb / self._satvap(T - 273.15), max=1.0)
        ci_solution = self._solve_ci(
            Cs=Cs,
            RH=RH,
            min_ci=min_ci,
            BallBerrySlope=BallBerrySlope,
            BallBerry0=BallBerry0,
            ppm2bar=ppm2bar,
            canopy_type=canopy_type,
            g_m=g_m,
            Vs_C3=Vs_C3,
            MM_consts=MM_consts,
            Rd=Rd,
            Vcmax=Vcmax,
            Gamma_star=Gamma_star,
            Je=Je,
            effcon=effcon,
            Ke=Ke,
            tol=opts.ci_tol,
            max_iter=opts.max_iter,
        )
        assimilation = self._compute_assimilation(
            Ci=ci_solution["Ci"],
            canopy_type=canopy_type,
            g_m=g_m,
            Vs_C3=Vs_C3,
            MM_consts=MM_consts,
            Rd=Rd,
            Vcmax=Vcmax,
            Gamma_star=Gamma_star,
            Je=Je,
            effcon=effcon,
            Ke=Ke,
        )

        A = assimilation["A"]
        Ag = assimilation["Ag"]
        CO2_per_electron = assimilation["CO2_per_electron"]
        gs = torch.clamp(1.6 * A * ppm2bar / (Cs - ci_solution["Ci"]).clamp(min=1e-12), min=0.0)
        Ja = Ag / CO2_per_electron.clamp(min=1e-12)
        rcw = (self.rhoa / (self.Mair * 1e-3)) / gs.clamp(min=1e-12)

        ps = torch.where(Je.abs() <= 1e-12, po0, po0 * Ja / Je)
        ps_rel = torch.clamp(1.0 - ps / po0.clamp(min=1e-12), min=0.0)

        fluorescence = self._fluorescence_model(ps, ps_rel, Kn0, Knalpha, Knbeta, Kd)
        Kpa = ps / fluorescence["fs"].clamp(min=1e-12) * self.Kf

        Cc = (ci_solution["Ci"] - A / g_m) / ppm2bar
        Ci_ppm = ci_solution["Ci"] / ppm2bar
        kf = torch.full_like(ci_solution["Ci"], self.Kf)
        kp0 = torch.full_like(ci_solution["Ci"], self.Kp)

        return LeafBiochemistryResult(
            A=A,
            Ag=Ag,
            Ci=Ci_ppm,
            Cc=Cc,
            rcw=rcw,
            gs=gs,
            RH=RH,
            Vcmax=Vcmax,
            Rd=Rd,
            Ja=Ja,
            ps=ps,
            ps_rel=ps_rel,
            Kd=Kd,
            Kn=fluorescence["Kn"],
            NPQ=fluorescence["Kn"] / (self.Kf + Kd),
            Kf=kf,
            Kp0=kp0,
            Kp=Kpa,
            eta=fluorescence["eta"],
            qE=fluorescence["qE"],
            fs=fluorescence["fs"],
            ft=fluorescence["fs"],
            SIF=fluorescence["fs"] * Q,
            fo0=fluorescence["fo0"],
            fm0=fluorescence["fm0"],
            fo=fluorescence["fo"],
            fm=fluorescence["fm"],
            Fm_Fo=fluorescence["fm"] / fluorescence["fo"].clamp(min=1e-12),
            Ft_Fo=fluorescence["fs"] / fluorescence["fo"].clamp(min=1e-12),
            qQ=fluorescence["qQ"],
            Phi_N=fluorescence["Kn"] / (fluorescence["Kn"] + self.Kp + self.Kf + Kd),
            CO2_per_electron=CO2_per_electron,
            fcount=int(ci_solution["fcount"]),
        )

    def _normalize_type(self, canopy_type: str | bool) -> str:
        if isinstance(canopy_type, torch.Tensor):
            flat = canopy_type.reshape(-1)
            if flat.numel() == 0:
                raise ValueError("Leaf biochemistry Type tensor cannot be empty")
            first = flat[0]
            if flat.numel() != 1 and not torch.all(flat == first):
                raise ValueError("Leaf biochemistry does not support mixed C3/C4 batches")
            canopy_type = float(first.item())
        if isinstance(canopy_type, (int, float)):
            return "C4" if canopy_type else "C3"
        if isinstance(canopy_type, bool):
            return "C4" if canopy_type else "C3"
        return "C4" if str(canopy_type).upper() == "C4" else "C3"

    def _infer_batch(self, *values: object) -> int:
        batch = 1
        for value in values:
            if value is None:
                continue
            tensor = torch.as_tensor(value, device=self.device, dtype=self.dtype)
            if tensor.ndim == 0:
                continue
            if batch == 1:
                batch = int(tensor.shape[0])
                continue
            if tensor.shape[0] not in (1, batch):
                raise ValueError("Biochemistry inputs must broadcast to a common batch size")
        return batch

    def _expand(self, value: torch.Tensor | float, batch: int) -> torch.Tensor:
        tensor = torch.as_tensor(value, device=self.device, dtype=self.dtype)
        if tensor.ndim == 0:
            return tensor.repeat(batch)
        if tensor.shape[0] == 1 and batch != 1:
            return tensor.expand(batch)
        if tensor.shape[0] != batch:
            raise ValueError("Biochemistry inputs must broadcast to the batch dimension")
        return tensor

    def _satvap(self, temp_c: torch.Tensor) -> torch.Tensor:
        return 6.107 * torch.pow(torch.full_like(temp_c, 10.0), 7.5 * temp_c / (237.3 + temp_c))

    def _temperature_function_c3(self, temperature_k: torch.Tensor, delta_ha: float) -> torch.Tensor:
        return torch.exp((delta_ha / (self.Tref * self.R)) * (1.0 - self.Tref / temperature_k))

    def _high_temp_inhibition_c3(self, temperature_k: torch.Tensor, delta_s: float, delta_hd: float) -> torch.Tensor:
        numerator = 1.0 + torch.exp(
            torch.full_like(temperature_k, (self.Tref * delta_s - delta_hd) / (self.Tref * self.R))
        )
        denominator = 1.0 + torch.exp((delta_s * temperature_k - delta_hd) / (self.R * temperature_k))
        return numerator / denominator

    def _solve_ci(
        self,
        *,
        Cs: torch.Tensor,
        RH: torch.Tensor,
        min_ci: float,
        BallBerrySlope: torch.Tensor,
        BallBerry0: torch.Tensor,
        ppm2bar: torch.Tensor,
        canopy_type: str,
        g_m: torch.Tensor,
        Vs_C3: torch.Tensor,
        MM_consts: torch.Tensor,
        Rd: torch.Tensor,
        Vcmax: torch.Tensor,
        Gamma_star: torch.Tensor,
        Je: torch.Tensor,
        effcon: torch.Tensor,
        Ke: torch.Tensor,
        tol: float,
        max_iter: int,
    ) -> dict[str, torch.Tensor | int]:
        zero_intercept = BallBerry0 == 0
        ci = self._ball_berry(Cs, RH, None, BallBerrySlope, BallBerry0, min_ci)
        fcount = 1
        if zero_intercept.all():
            return {"Ci": ci, "fcount": fcount}

        lower = min_ci * Cs
        upper = Cs
        err_lower = self._ci_error(
            lower,
            Cs=Cs,
            RH=RH,
            min_ci=min_ci,
            BallBerrySlope=BallBerrySlope,
            BallBerry0=BallBerry0,
            ppm2bar=ppm2bar,
            canopy_type=canopy_type,
            g_m=g_m,
            Vs_C3=Vs_C3,
            MM_consts=MM_consts,
            Rd=Rd,
            Vcmax=Vcmax,
            Gamma_star=Gamma_star,
            Je=Je,
            effcon=effcon,
            Ke=Ke,
        )
        err_upper = self._ci_error(
            upper,
            Cs=Cs,
            RH=RH,
            min_ci=min_ci,
            BallBerrySlope=BallBerrySlope,
            BallBerry0=BallBerry0,
            ppm2bar=ppm2bar,
            canopy_type=canopy_type,
            g_m=g_m,
            Vs_C3=Vs_C3,
            MM_consts=MM_consts,
            Rd=Rd,
            Vcmax=Vcmax,
            Gamma_star=Gamma_star,
            Je=Je,
            effcon=effcon,
            Ke=Ke,
        )

        swap = (err_lower < 0.0) & (err_upper > 0.0)
        lower, upper = torch.where(swap, upper, lower), torch.where(swap, lower, upper)
        err_lower, err_upper = torch.where(swap, err_upper, err_lower), torch.where(swap, err_lower, err_upper)
        bracketed = (~zero_intercept) & (err_lower >= 0.0) & (err_upper <= 0.0)

        for _ in range(max_iter):
            active = bracketed & ((upper - lower).abs() > tol)
            if not active.any():
                break
            fcount += 1
            mid = 0.5 * (lower + upper)
            err_mid = self._ci_error(
                mid,
                Cs=Cs,
                RH=RH,
                min_ci=min_ci,
                BallBerrySlope=BallBerrySlope,
                BallBerry0=BallBerry0,
                ppm2bar=ppm2bar,
                canopy_type=canopy_type,
                g_m=g_m,
                Vs_C3=Vs_C3,
                MM_consts=MM_consts,
                Rd=Rd,
                Vcmax=Vcmax,
                Gamma_star=Gamma_star,
                Je=Je,
                effcon=effcon,
                Ke=Ke,
            )
            same_sign = (err_mid >= 0.0) == (err_lower >= 0.0)
            move_lower = active & same_sign
            move_upper = active & ~same_sign
            lower = torch.where(move_lower, mid, lower)
            err_lower = torch.where(move_lower, err_mid, err_lower)
            upper = torch.where(move_upper, mid, upper)
            err_upper = torch.where(move_upper, err_mid, err_upper)

        ci = torch.where(bracketed, 0.5 * (lower + upper), ci)

        fallback = (~zero_intercept) & ~bracketed
        if fallback.any():
            fixed_point = ci.clone()
            for _ in range(max_iter):
                assimilation = self._compute_assimilation(
                    Ci=fixed_point,
                    canopy_type=canopy_type,
                    g_m=g_m,
                    Vs_C3=Vs_C3,
                    MM_consts=MM_consts,
                    Rd=Rd,
                    Vcmax=Vcmax,
                    Gamma_star=Gamma_star,
                    Je=Je,
                    effcon=effcon,
                    Ke=Ke,
                )
                fixed_next = self._ball_berry(
                    Cs,
                    RH,
                    assimilation["A"] * ppm2bar,
                    BallBerrySlope,
                    BallBerry0,
                    min_ci,
                )
                fcount += 1
                converged = torch.abs(fixed_next - fixed_point) <= tol
                fixed_point = torch.where(fallback, fixed_next, fixed_point)
                if not (fallback & ~converged).any():
                    break
            ci = torch.where(fallback, fixed_point, ci)
        return {"Ci": ci, "fcount": fcount}

    def _ci_error(
        self,
        ci_in: torch.Tensor,
        *,
        Cs: torch.Tensor,
        RH: torch.Tensor,
        min_ci: float,
        BallBerrySlope: torch.Tensor,
        BallBerry0: torch.Tensor,
        ppm2bar: torch.Tensor,
        canopy_type: str,
        g_m: torch.Tensor,
        Vs_C3: torch.Tensor,
        MM_consts: torch.Tensor,
        Rd: torch.Tensor,
        Vcmax: torch.Tensor,
        Gamma_star: torch.Tensor,
        Je: torch.Tensor,
        effcon: torch.Tensor,
        Ke: torch.Tensor,
    ) -> torch.Tensor:
        assimilation = self._compute_assimilation(
            Ci=ci_in,
            canopy_type=canopy_type,
            g_m=g_m,
            Vs_C3=Vs_C3,
            MM_consts=MM_consts,
            Rd=Rd,
            Vcmax=Vcmax,
            Gamma_star=Gamma_star,
            Je=Je,
            effcon=effcon,
            Ke=Ke,
        )
        ci_out = self._ball_berry(
            Cs,
            RH,
            assimilation["A"] * ppm2bar,
            BallBerrySlope,
            BallBerry0,
            min_ci,
        )
        return ci_out - ci_in

    def _compute_assimilation(
        self,
        *,
        Ci: torch.Tensor,
        canopy_type: str,
        g_m: torch.Tensor,
        Vs_C3: torch.Tensor,
        MM_consts: torch.Tensor,
        Rd: torch.Tensor,
        Vcmax: torch.Tensor,
        Gamma_star: torch.Tensor,
        Je: torch.Tensor,
        effcon: torch.Tensor,
        Ke: torch.Tensor,
    ) -> dict[str, torch.Tensor | int]:
        if canopy_type == "C3":
            Vs = Vs_C3
            Vc = Vcmax * (Ci - Gamma_star) / (MM_consts + Ci).clamp(min=1e-12)
            CO2_per_electron = ((Ci - Gamma_star) / (Ci + 2.0 * Gamma_star).clamp(min=1e-12)) * effcon
            Ve = Je * CO2_per_electron

            finite_gm = torch.isfinite(g_m)
            if finite_gm.any():
                gm = g_m[finite_gm]
                ci = Ci[finite_gm]
                mm = MM_consts[finite_gm]
                rd = Rd[finite_gm]
                vcmax = Vcmax[finite_gm]
                gamma_star = Gamma_star[finite_gm]
                je = Je[finite_gm]
                eff = effcon[finite_gm]
                Vc[finite_gm] = self._sel_root(
                    1.0 / gm,
                    -(mm + ci + (rd + vcmax) / gm),
                    vcmax * (ci - gamma_star + rd / gm),
                    torch.full_like(ci, -1.0),
                )
                Ve[finite_gm] = self._sel_root(
                    1.0 / gm,
                    -(ci + 2.0 * gamma_star + (rd + je * eff) / gm),
                    je * eff * (ci - gamma_star + rd / gm),
                    torch.full_like(ci, -1.0),
                )
                CO2_per_electron[finite_gm] = Ve[finite_gm] / je.clamp(min=1e-12)
        else:
            Vc = Vcmax
            Vs = Ke * Ci
            CO2_per_electron = effcon
            Ve = Je * CO2_per_electron

        V = self._sel_root(
            torch.full_like(Ci, self.atheta),
            -(Vc + Ve),
            Vc * Ve,
            torch.sign(-Vc),
        )
        Ag = self._sel_root(
            torch.full_like(Ci, 0.98),
            -(V + Vs),
            V * Vs,
            torch.full_like(Ci, -1.0),
        )
        A = Ag - Rd
        return {
            "A": A,
            "Ag": Ag,
            "Vc": Vc,
            "Vs": Vs,
            "Ve": Ve,
            "CO2_per_electron": CO2_per_electron,
            "fcount": 1,
        }

    def _sel_root(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, dsign: torch.Tensor) -> torch.Tensor:
        result = torch.empty_like(b)
        linear = a == 0
        if linear.any():
            result[linear] = -c[linear] / b[linear].clamp(min=1e-12)
        if (~linear).any():
            a_nl = a[~linear]
            b_nl = b[~linear]
            c_nl = c[~linear]
            dsign_nl = dsign[~linear]
            dsign_nl = torch.where(dsign_nl == 0, torch.full_like(dsign_nl, -1.0), dsign_nl)
            disc = torch.sqrt(torch.clamp(b_nl * b_nl - 4.0 * a_nl * c_nl, min=0.0))
            result[~linear] = (-b_nl + dsign_nl * disc) / (2.0 * a_nl)
        return result

    def _ball_berry(
        self,
        Cs: torch.Tensor,
        RH: torch.Tensor,
        A: Optional[torch.Tensor],
        BallBerrySlope: torch.Tensor,
        BallBerry0: torch.Tensor,
        min_ci: float,
    ) -> torch.Tensor:
        if A is None or (BallBerry0 == 0).all():
            return torch.maximum(min_ci * Cs, Cs * (1.0 - 1.6 / (BallBerrySlope * RH).clamp(min=1e-12)))
        gs = self._gs_fun(Cs, RH, A, BallBerrySlope, BallBerry0)
        return torch.maximum(min_ci * Cs, Cs - 1.6 * A / gs.clamp(min=1e-12))

    def _gs_fun(
        self,
        Cs: torch.Tensor,
        RH: torch.Tensor,
        A: torch.Tensor,
        BallBerrySlope: torch.Tensor,
        BallBerry0: torch.Tensor,
    ) -> torch.Tensor:
        gs = BallBerrySlope * A * RH / (Cs + 1e-9) + BallBerry0
        gs = torch.maximum(BallBerry0, gs)
        return torch.where(torch.isnan(Cs), torch.full_like(gs, torch.nan), gs)

    def _fluorescence_model(
        self,
        ps: torch.Tensor,
        ps_rel: torch.Tensor,
        Kn0: torch.Tensor,
        Knalpha: torch.Tensor,
        Knbeta: torch.Tensor,
        Kd: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x_alpha = torch.where(
            ps_rel > 0.0,
            torch.exp(torch.log(ps_rel) * Knalpha),
            torch.zeros_like(ps_rel),
        )
        Kn = Kn0 * (1.0 + Knbeta) * x_alpha / (Knbeta + x_alpha).clamp(min=1e-12)
        fo0 = self.Kf / (self.Kf + self.Kp + Kd)
        fo = self.Kf / (self.Kf + self.Kp + Kd + Kn)
        fm = self.Kf / (self.Kf + Kd + Kn)
        fm0 = self.Kf / (self.Kf + Kd)
        fs = fm * (1.0 - ps)
        eta = fs / fo0.clamp(min=1e-12)
        qQ = 1.0 - (fs - fo) / (fm - fo).clamp(min=1e-12)
        qE = 1.0 - (fm - fo) / (fm0 - fo0).clamp(min=1e-12)
        return {
            "Kn": Kn,
            "fo0": fo0,
            "fo": fo,
            "fm": fm,
            "fm0": fm0,
            "fs": fs,
            "eta": eta,
            "qQ": qQ,
            "qE": qE,
        }
