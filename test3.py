"""
coldflow_blowdown_local.py
- Solver iterativo (energia + continuidade + EoS) com cap HEM
- Blowdown do tanque (M(t), H(t)) usando saturação por T (rápido)
- Salva CSV com histórico temporal
- Mostra plots (não guarda imagens)

Requisitos:
    pip install CoolProp matplotlib
"""

import os
import csv
from dataclasses import dataclass, asdict
from math import sqrt, pi
from typing import Optional, Dict, Any

from CoolProp.CoolProp import PropsSI
from matplotlib import pyplot as plt

# ---------------- util ----------------

def _bar_to_Pa(p_bar: float) -> float: return p_bar * 1e5
def _mm2_to_m2(a_mm2: float) -> float: return a_mm2 * 1e-6
def _cm2_to_m2(a_cm2: float) -> float: return a_cm2 * 1e-4


# ---------------- casos ----------------

@dataclass
class Case:
    fluid: str = "NitrousOxide"
    P1_bar: float = 50.0
    T1_K: float = 273.15
    P2_bar: float = 1.01325
    # NOVO: defina N furos e D por furo; Ao pode ficar None
    N_holes: int = 12
    D_hole_m: float = 0.0015
    Ao_mm2: Optional[float] = None  # opção antiga; se fornecida, tem prioridade
    Cd: float = 0.8
    zeta: float = 0.7
    A1_cm2: Optional[float] = None
    use_NHNE: bool = False
    max_iter: int = 120
    tol_rel: float = 1e-5

@dataclass
class Result:
    mdot: float; mdot_kin: float; mdot_inc: float; mdot_HEM_cap: float
    V1: float; V2: float; rho1: float; rho2: float; h1: float; h2: float
    x2: Optional[float]; T2: Optional[float]
    n_iters: int; converged: bool
    def as_dict(self) -> Dict[str, Any]: return asdict(self)


# ---------------- propriedades ----------------

def upstream_from_PT(fluid: str, P_bar: float, T: float):
    P = _bar_to_Pa(P_bar)
    Psat_T = PropsSI('P', 'T', T, 'Q', 0, fluid)
    if abs(P - Psat_T) / max(P, 1.0) < 1e-6:
        rho = PropsSI('D', 'T', T, 'Q', 0, fluid)
        h   = PropsSI('H', 'T', T, 'Q', 0, fluid)
        s   = PropsSI('S', 'T', T, 'Q', 0, fluid)
        return rho, h, s
    rho = PropsSI('D', 'P', P, 'T', T, fluid)
    h   = PropsSI('H', 'P', P, 'T', T, fluid)
    s   = PropsSI('S', 'P', P, 'T', T, fluid)
    return rho, h, s

def sat_props_at_T(fluid: str, T: float):
    """Props de saturação por T (rápidas): hL,hV,rhoL,rhoV, P_sat."""
    hL = PropsSI('H', 'T', T, 'Q', 0, fluid)
    hV = PropsSI('H', 'T', T, 'Q', 1, fluid)
    rhoL = PropsSI('D', 'T', T, 'Q', 0, fluid)
    rhoV = PropsSI('D', 'T', T, 'Q', 1, fluid)
    P = PropsSI('P', 'T', T, 'Q', 0, fluid)
    return hL, hV, rhoL, rhoV, P

def state_P_h(fluid: str, P: float, h: float):
    """Estado jusante dado P,h; usa fallback por T_sat se P-H falhar."""
    try:
        rho = PropsSI('D', 'P', P, 'H', h, fluid)
        T   = PropsSI('T', 'P', P, 'H', h, fluid)
        return rho, T, None
    except Exception:
        T_sat = PropsSI('T', 'P', P, 'Q', 0, fluid)
        hL, hV, rhoL, rhoV, _ = sat_props_at_T(fluid, T_sat)
        if hL <= h <= hV:
            x = (h - hL) / max(hV - hL, 1e-12)
            rho = 1.0 / (x / rhoV + (1 - x) / rhoL)
            return rho, T_sat, x
        T0 = T_sat - 1e-3 if h < hL else T_sat + 1e-3
        rho = PropsSI('D', 'P', P, 'T', T0, fluid)
        return rho, T0, None


# ---------------- Solver estação única ----------------

class ColdFlowSimulator:
    def __init__(self, case: Case):
        self.c = case

    @staticmethod
    def _total_orifice_area_m2(c: Case) -> float:
        # Prioridade para Ao_mm2 se fornecida explicitamente
        if c.Ao_mm2 is not None:
            return _mm2_to_m2(c.Ao_mm2)
        # Caso padrão: N furos e diâmetro por furo
        return c.N_holes * (pi * (c.D_hole_m * 0.5) ** 2)

    def _critical_mass_flux_HEM(self, fluid, P1, s1, h1, rho2_for_cap, P2_eff):
        """Estimativa conservadora de G* (cap HEM)."""
        try:
            T2s  = PropsSI('T', 'P', P2_eff, 'S', s1, fluid)
            h2s  = PropsSI('H', 'P', P2_eff, 'S', s1, fluid)
            rho2s= PropsSI('D', 'P', P2_eff, 'S', s1, fluid)
            dh = max(h1 - h2s, 0.0)
            return rho2s * sqrt(2.0 * dh)
        except Exception:
            # fallback: vapor sat em P2_eff
            T_sat = PropsSI('T', 'P', P2_eff, 'Q', 0, fluid)
            hL, hV, _, _, _ = sat_props_at_T(fluid, T_sat)
            dh = max(h1 - hV, 0.0)
            return rho2_for_cap * sqrt(2.0 * dh)

    def solve(self) -> Result:
        c = self.c
        fluid = c.fluid
        P1 = _bar_to_Pa(c.P1_bar)
        P2 = _bar_to_Pa(c.P2_bar)
        Ao = self._total_orifice_area_m2(c)
        Cd = c.Cd
        zeta = max(c.zeta, 0.0)
        A1 = None if c.A1_cm2 is None else _cm2_to_m2(c.A1_cm2)

        # limites físicos
        P_tr = PropsSI('p_triple', fluid)
        P2_eff = max(P2, P_tr * (1.0 + 1e-6))

        # Montante
        rho1, h1, s1 = upstream_from_PT(fluid, c.P1_bar, c.T1_K)
        dP = max(P1 - P2_eff, 0.0)
        V2 = sqrt(2.0 * dP / (rho1 * (1 + zeta)))
        V1 = 0.0
        mdot_prev = 0.0
        rho2_prev = 0.0

        for it in range(1, c.max_iter + 1):
            # Energia
            h2 = h1 + 0.5 * (V1 ** 2 - (1.0 + zeta) * V2 ** 2)
            # Estado jusante
            rho2, T2, x2 = state_P_h(fluid, P2_eff, h2)
            # Continuidade
            mdot_kin = Cd * Ao * rho2 * V2
            # Montante (quase sempre 0 se plano 1 é o tanque)
            V1 = mdot_kin / (rho1 * A1) if A1 else 0.0
            # Velocidade por energia com rho2 atual
            V2_energy = sqrt(2.0 * dP / (rho2 * (1.0 + zeta)))
            # Cap crítico HEM
            G_cap = self._critical_mass_flux_HEM(fluid, P1, s1, h1, rho2, P2_eff)
            mdot_cap = Cd * Ao * G_cap
            V2_cap = mdot_cap / (Cd * Ao * rho2)
            # alvo do passo
            V2_target = min(V2_energy, V2_cap)
            # convergência
            rel_m = abs(mdot_kin - mdot_prev) / max(abs(mdot_kin), 1.0)
            rel_r = abs(rho2 - rho2_prev) / max(abs(rho2), 1.0)
            rel_v = abs(V2_target - V2) / max(abs(V2_target), 1.0)
            # relaxação forte
            V2 = 0.25 * V2 + 0.75 * V2_target
            mdot_prev, rho2_prev = mdot_kin, rho2
            at_cap = V2_energy > V2_cap * (1 - 1e-6)
            if (rel_m < c.tol_rel and rel_r < c.tol_rel) or (rel_v < c.tol_rel) or (at_cap and rel_v < 5e-4):
                break

        # Recalcular final e respeitar cap
        h2 = h1 + 0.5 * (V1 ** 2 - (1.0 + zeta) * V2 ** 2)
        rho2, T2, x2 = state_P_h(fluid, P2_eff, h2)
        mdot_kin = Cd * Ao * rho2 * V2
        G_cap = self._critical_mass_flux_HEM(fluid, P1, s1, h1, rho2, P2_eff)
        mdot_cap = Cd * Ao * G_cap
        if mdot_kin > mdot_cap:
            mdot_kin = mdot_cap
            V2 = mdot_kin / (Cd * Ao * rho2)
        mdot_inc = Cd * Ao * sqrt(2.0 * rho1 * dP)
        converged = (it < c.max_iter)
        return Result(mdot_kin, mdot_kin, mdot_inc, mdot_cap,
                      V1, V2, rho1, rho2, h1, h2, x2, T2, it, converged)


# ---------------- Blowdown (rápido, por T de saturação) ----------------

@dataclass
class BlowdownCase:
    Vtank_m3: float = 0.0499
    M0_kg: float = 12.5
    T0_K: float = 293.15
    P0_bar: float = 50.0
    # Usa-se N_holes e D_hole_m do Case para A0 total
    N_holes: int = 12
    D_hole_m: float = 0.0015
    Cd: float = 0.8
    zeta: float = 0.7
    P2_bar: float = 1.01325
    dt: float = 0.05
    t_end: float = 60.0

def _mix_h_from_rho_T(fluid: str, rho_mix: float, T: float):
    hL, hV, rhoL, rhoV, Psat = sat_props_at_T(fluid, T)
    rho = max(rho_mix, 1e-6)
    x = (rhoL * (rhoL - rho)) / max(rho * (rhoL - rhoV), 1e-12)
    x = max(0.0, min(1.0, x))
    h = x * hV + (1 - x) * hL
    return h, x, Psat

def _solve_T_for_h_rho_T(fluid: str, h_target: float, rho_mix: float, T_lo: float, T_hi: float):
    # Bissecção apenas em T de saturação (rápido)
    f_lo = _mix_h_from_rho_T(fluid, rho_mix, T_lo)[0] - h_target
    f_hi = _mix_h_from_rho_T(fluid, rho_mix, T_hi)[0] - h_target
    a, b = T_lo, T_hi
    for _ in range(60):
        m = 0.5 * (a + b)
        fm = _mix_h_from_rho_T(fluid, rho_mix, m)[0] - h_target
        if abs(fm) < 1e-4:
            return m
        if f_lo * fm <= 0:
            b = m; f_hi = fm
        else:
            a = m; f_lo = fm
    return 0.5 * (a + b)

class BlowdownSimulator:
    def __init__(self, fluid: str, bcase: BlowdownCase):
        self.fluid = fluid
        self.b = bcase

    def run(self):
        b = self.b; fluid = self.fluid
        # Estado inicial
        rho1, h1, s1 = upstream_from_PT(fluid, b.P0_bar, b.T0_K)
        M = b.M0_kg
        H = M * h1
        V = b.Vtank_m3
        T = b.T0_K
        Ttr = PropsSI('T_triple', fluid) + 0.5
        Tcr = PropsSI('T_critical', fluid) - 0.5

        sim = ColdFlowSimulator(Case(
            fluid=fluid, P1_bar=b.P0_bar, T1_K=b.T0_K,
            P2_bar=b.P2_bar,
            N_holes=b.N_holes, D_hole_m=b.D_hole_m, Ao_mm2=None,
            Cd=b.Cd, zeta=b.zeta, A1_cm2=None, use_NHNE=False,
            max_iter=120, tol_rel=1e-5
        ))

        t = 0.0
        t_s=[]; M_s=[]; Pbar_s=[]; T_s=[]; mdot_s=[]; V2_s=[]
        rho1_s=[]; rho2_s=[]; h1_s=[]; h2_s=[]

        while t <= b.t_end and M > 1e-6:
            rho_mix = M / V
            h_mix = H / max(M, 1e-12)
            T = _solve_T_for_h_rho_T(fluid, h_mix, rho_mix, Ttr, Tcr)
            _, _, Psat = _mix_h_from_rho_T(fluid, rho_mix, T)
            P_bar = Psat / 1e5

            sim.c.P1_bar = P_bar
            sim.c.T1_K = T
            r = sim.solve()
            mdot = r.mdot

            # log
            t_s.append(t); M_s.append(M); Pbar_s.append(P_bar); T_s.append(T)
            mdot_s.append(mdot); V2_s.append(r.V2)
            rho1_s.append(r.rho1); rho2_s.append(r.rho2); h1_s.append(r.h1); h2_s.append(r.h2)

            # Euler: sai com h≈h1 (montante)
            dM = mdot * b.dt
            dH = r.h1 * mdot * b.dt
            M = max(M - dM, 0.0)
            H = max(H - dH, 0.0)
            t += b.dt

        return {
            "t_s": t_s, "M_kg": M_s, "P_bar": Pbar_s, "T_K": T_s,
            "mdot_kg_s": mdot_s, "V2_m_s": V2_s,
            "rho1_kg_m3": rho1_s, "rho2_kg_m3": rho2_s,
            "h1_J_kg": h1_s, "h2_J_kg": h2_s,
            "t_empty_s": (t_s[-1] if len(t_s) > 0 and M <= 1e-6 else None)
        }


# ---------------- main ----------------

if __name__ == "__main__":
    # 1) Resolver ponto único com N=12 e D=1.5 mm
    case = Case(N_holes=12, D_hole_m=0.0015, Ao_mm2=None)
    r = ColdFlowSimulator(case).solve()
    print("Convergiu:", r.converged, "iters:", r.n_iters)
    for k, v in r.as_dict().items():
        print(f"{k}: {v}")

    # 2) Blowdown + CSV e plots mostrados (não guardar imagens)
    bcase = BlowdownCase(N_holes=12, D_hole_m=0.0015)
    hist = BlowdownSimulator("NitrousOxide", bcase).run()

    # CSV no mesmo diretório do script
    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "blowdown_history.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["t_s","M_kg","P_bar","T_K","mdot_kg_s","V2_m_s",
                  "rho1_kg_m3","rho2_kg_m3","h1_J_kg","h2_J_kg"]
        w.writerow(header)
        for i in range(len(hist["t_s"])):
            w.writerow([
                hist["t_s"][i], hist["M_kg"][i], hist["P_bar"][i], hist["T_K"][i],
                hist["mdot_kg_s"][i], hist["V2_m_s"][i],
                hist["rho1_kg_m3"][i], hist["rho2_kg_m3"][i],
                hist["h1_J_kg"][i], hist["h2_J_kg"][i]
            ])
    print("CSV salvo em:", csv_path)

    if hist["t_empty_s"] is not None:
        print("t_empty_s:", hist["t_empty_s"])
    else:
        print("Aviso: tanque não esvaziou dentro do t_end (aumente t_end ou dt).")

    # Plots mostrados no ecrã
    plt.figure(); plt.plot(hist["t_s"], hist["M_kg"])
    plt.xlabel("time [s]"); plt.ylabel("tank mass [kg]"); plt.title("Blowdown mass"); plt.tight_layout()

    plt.figure(); plt.plot(hist["t_s"], hist["P_bar"])
    plt.xlabel("time [s]"); plt.ylabel("pressure [bar]"); plt.title("Blowdown pressure"); plt.tight_layout()

    plt.figure(); plt.plot(hist["t_s"], hist["T_K"])
    plt.xlabel("time [s]"); plt.ylabel("temperature [K]"); plt.title("Blowdown temperature"); plt.tight_layout()

    plt.figure(); plt.plot(hist["t_s"], hist["mdot_kg_s"])
    plt.xlabel("time [s]"); plt.ylabel("mdot [kg/s]"); plt.title("Blowdown mdot"); plt.tight_layout()

    plt.figure(); plt.plot(hist["t_s"], hist["V2_m_s"])
    plt.xlabel("time [s]"); plt.ylabel("V2 [m/s]"); plt.title("Blowdown velocity"); plt.tight_layout()

    plt.show()
