"""
coldflow_blowdown_solomon_NHNE.py
- NHNE (Solomon) com salto adiabático em entalpia TOTAL:
    h1 + V1^2/2 = h2 + (1+zeta) V2^2/2
- mdot = blend_NHNE(mdot_inc, mdot_HEM; kappa)
    kappa = sqrt( (P1-P2) / (Pv1-P2) ) com limites numéricos
- Blowdown do tanque: integra M e H com mdot_NHNE e h_out ≈ h1
- N furos e D por furo -> A_total
- Salva CSV e MOSTRA plots

Requisitos:
    pip install CoolProp matplotlib
"""

import os
import csv
from dataclasses import dataclass, asdict
from math import sqrt, pi
from typing import Optional, Dict, Any, Tuple

from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt


# ---------------- util ----------------

def _bar_to_Pa(p_bar: float) -> float: return p_bar * 1e5
def _Pa_to_bar(p: float) -> float: return p / 1e5
def _cm2_to_m2(a_cm2: float) -> float: return a_cm2 * 1e-4

def _clip_T(fluid: str, T: float) -> float:
    Ttr = PropsSI('Ttriple', fluid) + 1e-3
    Tcr = PropsSI('Tcrit', fluid)   - 1e-3
    return max(min(T, Tcr), Ttr)


# ---------------- casos ----------------

@dataclass
class Case:
    fluid: str = "NitrousOxide"
    P1_bar: float = 50.0
    T1_K: float = 293.15
    P2_bar: float = 1.01325
    # orifícios
    N_holes: int = 12
    D_hole_m: float = 0.0015
    # hidráulica
    Cd: float = 0.8
    zeta: float = 0.7            # perdas adicionais no salto energético
    A1_cm2: Optional[float] = None  # secção montante (opcional); se None => V1≈0
    # solver
    max_iter: int = 120
    tol_rel: float = 1e-6

@dataclass
class Result:
    mdot: float; mdot_inc: float; mdot_HEM: float; mdot_cap_HEM: float
    V1: float; V2: float; rho1: float; rho2: float; h1: float; h2: float
    x2: Optional[float]; T2: Optional[float]
    kappa: float
    n_iters: int; converged: bool
    def as_dict(self) -> Dict[str, Any]: return asdict(self)


# ---------------- propriedades ----------------

def upstream_from_PT(fluid: str, P_bar: float, T: float) -> Tuple[float,float,float]:
    """
    Estado montante a partir de P,T. Robusto quando P≈Psat(T).
    Cai para saturado (Q=0) se P e T estiverem na curva de saturação.
    """
    P = _bar_to_Pa(P_bar)
    T = _clip_T(fluid, T)
    Psat_T = PropsSI('P','T',T,'Q',0,fluid)
    # tolerância relativa 1e-4 OU absoluta 50 Pa
    if abs(P - Psat_T)/max(P,1.0) < 1e-4 or abs(P - Psat_T) < 50.0:
        rho = PropsSI('D','T',T,'Q',0,fluid)
        h   = PropsSI('H','T',T,'Q',0,fluid)
        s   = PropsSI('S','T',T,'Q',0,fluid)
        return rho,h,s
    try:
        rho = PropsSI('D','P',P,'T',T,fluid)
        h   = PropsSI('H','P',P,'T',T,fluid)
        s   = PropsSI('S','P',P,'T',T,fluid)
        return rho,h,s
    except ValueError:
        rho = PropsSI('D','T',T,'Q',0,fluid)
        h   = PropsSI('H','T',T,'Q',0,fluid)
        s   = PropsSI('S','T',T,'Q',0,fluid)
        return rho,h,s

def sat_props_at_T(fluid: str, T: float):
    T = _clip_T(fluid, T)
    hL = PropsSI('H','T',T,'Q',0,fluid)
    hV = PropsSI('H','T',T,'Q',1,fluid)
    rhoL = PropsSI('D','T',T,'Q',0,fluid)
    rhoV = PropsSI('D','T',T,'Q',1,fluid)
    P = PropsSI('P','T',T,'Q',0,fluid)
    return hL,hV,rhoL,rhoV,P

def state_P_h(fluid: str, P: float, h: float) -> Tuple[float, float, Optional[float]]:
    """
    Estado jusante dado P,h; mistura saturada se P-H falhar.
    Protege para P < p_triple usando P_eff >= p_triple.
    """
    p_tr = PropsSI('ptriple', fluid)
    P_eff = max(P, p_tr*(1.0+1e-6))
    try:
        rho = PropsSI('D','P',P_eff,'H',h,fluid)
        T   = PropsSI('T','P',P_eff,'H',h,fluid)
        return rho, T, None
    except Exception:
        T_sat = PropsSI('T','P',P_eff,'Q',0,fluid)
        hL,hV,rhoL,rhoV,_ = sat_props_at_T(fluid, T_sat)
        if hL <= h <= hV:
            x = (h - hL)/max(hV-hL, 1e-12)
            rho = 1.0/(x/rhoV + (1-x)/rhoL)
            return rho, T_sat, x
        # fora da cúpula: pequeno deslocamento térmico
        T0 = T_sat - 1e-3 if h < hL else T_sat + 1e-3
        rho = PropsSI('D','P',P_eff,'T',T0,fluid)
        return rho, T0, None


# ---------------- Solver NHNE (Solomon) ----------------

class ColdFlowNHNE:
    def __init__(self, case: Case):
        self.c = case

    @staticmethod
    def _Ao_total(c: Case) -> float:
        return c.N_holes * (pi * (c.D_hole_m*0.5)**2)

    @staticmethod
    def _kappa(P1: float, P2: float, Pv1: float) -> float:
        num = max(P1 - P2, 0.0)
        den = max(Pv1 - P2, 1e-6)
        k = sqrt(num / den)
        return max(0.0, min(k, 1e6))

    def solve(self) -> Result:
        c = self.c
        fluid = c.fluid
        P1 = _bar_to_Pa(c.P1_bar)
        P2 = _bar_to_Pa(c.P2_bar)
        p_tr = PropsSI('ptriple', fluid)
        P2_eff = max(P2, p_tr*(1.0+1e-6))  # evita P2 abaixo do triplo

        Ao = self._Ao_total(c)
        Cd = c.Cd
        zeta = max(c.zeta, 0.0)
        A1 = None if c.A1_cm2 is None else _cm2_to_m2(c.A1_cm2)

        # montante
        rho1, h1, s1 = upstream_from_PT(fluid, c.P1_bar, c.T1_K)
        Pv1 = PropsSI('P', 'T', _clip_T(fluid, c.T1_K), 'Q', 0, fluid)

        # chute inicial
        mdot = Cd * Ao * sqrt(2.0 * rho1 * max(P1-P2_eff,0.0)) * 0.8
        V1 = 0.0 if A1 is None else mdot / max(rho1*A1, 1e-12)

        mdot_prev = mdot
        rho2_prev = rho1
        V2_prev = 0.0

        for it in range(1, c.max_iter+1):
            # velocidades do passo
            V1 = 0.0 if A1 is None else mdot / max(rho1*A1, 1e-12)
            # semente para h2
            h2_guess = h1 - 0.5*(1.0+zeta)*(mdot/max(Cd*Ao*max(rho1,1e-9),1e-9))**2

            # fechar (h2, rho2, V2)
            for _ in range(3):
                rho2, T2, x2 = state_P_h(fluid, P2_eff, h2_guess)
                V2 = mdot / max(Cd*Ao*rho2, 1e-12)
                h2_guess = h1 + 0.5*(V1**2 - (1.0+zeta)*V2**2)

            rho2, T2, x2 = state_P_h(fluid, P2_eff, h2_guess)
            V2 = mdot / max(Cd*Ao*rho2, 1e-12)
            h2 = h1 + 0.5*(V1**2 - (1.0+zeta)*V2**2)

            # fluxos parciais
            dP = max(P1-P2_eff, 0.0)
            mdot_inc = Cd * Ao * sqrt(2.0 * rho1 * dP)
            dh = max(h1 - h2, 0.0)
            mdot_HEM = Cd * Ao * rho2 * sqrt(2.0 * dh)

            # blend NHNE
            kappa = self._kappa(P1, P2_eff, Pv1)
            w_HEM = 1.0/(1.0 + kappa)
            w_inc = 1.0 - w_HEM
            mdot_new = w_inc*mdot_inc + w_HEM*mdot_HEM

            # relaxação e paragem
            rel = abs(mdot_new - mdot)/max(abs(mdot_new), 1.0)
            mdot = 0.6*mdot + 0.4*mdot_new
            if rel < c.tol_rel and abs(rho2 - rho2_prev)/max(rho2,1.0) < 1e-6 and abs(V2 - V2_prev)/max(abs(V2),1.0) < 1e-6:
                break
            mdot_prev, rho2_prev, V2_prev = mdot, rho2, V2

        # cap crítico HEM (opcional)
        try:
            h2s  = PropsSI('H','P',P2_eff,'S',s1,fluid)
            rho2s= PropsSI('D','P',P2_eff,'S',s1,fluid)
            dh_c = max(h1 - h2s, 0.0)
            Gcap = rho2s * sqrt(2.0 * dh_c)
        except Exception:
            Gcap = rho2 * sqrt(2.0 * max(h1 - h2, 0.0))

        mdot_cap = Cd * Ao * Gcap
        if mdot > mdot_cap:
            mdot = mdot_cap
            V2 = mdot / max(Cd*Ao*rho2, 1e-12)
            h2 = h1 + 0.5*(V1**2 - (1.0+zeta)*V2**2)

        converged = (it < c.max_iter)
        return Result(mdot=mdot, mdot_inc=mdot_inc, mdot_HEM=mdot_HEM, mdot_cap_HEM=mdot_cap,
                      V1=V1, V2=V2, rho1=rho1, rho2=rho2, h1=h1, h2=h2, x2=x2, T2=T2,
                      kappa=kappa, n_iters=it, converged=converged)


# ---------------- Blowdown por entalpia ----------------

@dataclass
class BlowdownCase:
    Vtank_m3: float = 0.0499
    M0_kg: float = 12.5
    T0_K: float = 293.15
    P0_bar: float = 50.0
    # orifícios
    N_holes: int = 12
    D_hole_m: float = 0.0015
    # hidráulica
    Cd: float = 0.8
    zeta: float = 0.7
    P2_bar: float = 1.01325
    # integração
    dt: float = 0.05
    t_end: float = 60.0

def _mix_h_from_rho_T(fluid: str, rho_mix: float, T: float):
    """h_mix(T,rho) em saturação."""
    hL,hV,rhoL,rhoV,Psat = sat_props_at_T(fluid, T)
    rho = max(rho_mix, 1e-9)
    x = (rhoL*(rhoL - rho))/max(rho*(rhoL - rhoV), 1e-12)
    x = max(0.0, min(1.0, x))
    h = x*hV + (1-x)*hL
    return h, x, Psat

def _solve_T_for_h_rho_T(fluid: str, h_target: float, rho_mix: float, T_lo: float, T_hi: float):
    """
    Resolve T tal que h_mix(T,rho)=h_target por bissecção segura.
    Garante bracketing; se não houver, cola ao limite mais próximo.
    """
    a, b = T_lo, T_hi
    def f(T): return _mix_h_from_rho_T(fluid, rho_mix, T)[0] - h_target

    fa = f(a); fb = f(b)
    # Se não há bracketing, aproxima pelo limite mais próximo
    if fa == 0.0: return a
    if fb == 0.0: return b
    if fa*fb > 0:
        # monotonia típica: h_mix ~ crescente em T; escolher lado mais perto
        return a if abs(fa) < abs(fb) else b

    for _ in range(80):
        m = 0.5*(a+b)
        fm = f(m)
        if abs(fm) < 1e-4 or (b-a) < 1e-6:
            return m
        # passo de bissecção
        if fa*fm <= 0:
            b = m; fb = fm
        else:
            a = m; fa = fm
    return 0.5*(a+b)


class BlowdownSimulator:
    def __init__(self, fluid: str, bcase: BlowdownCase):
        self.fluid = fluid
        self.b = bcase

    def run(self):
        b = self.b; fluid = self.fluid
        # estado inicial do tanque
        rho1, h1, _ = upstream_from_PT(fluid, b.P0_bar, b.T0_K)
        M = b.M0_kg
        H = M * h1
        V = b.Vtank_m3

        Ttr = PropsSI('Ttriple', fluid) + 0.5
        Tcr = PropsSI('Tcrit',   fluid) - 0.5

        sim = ColdFlowNHNE(Case(
            fluid=fluid, P1_bar=b.P0_bar, T1_K=b.T0_K,
            P2_bar=b.P2_bar, N_holes=b.N_holes, D_hole_m=b.D_hole_m,
            Cd=b.Cd, zeta=b.zeta, A1_cm2=None, max_iter=120, tol_rel=1e-6
        ))

        t = 0.0
        # histórico
        t_s=[]; M_s=[]; Pbar_s=[]; T_s=[]
        mdot_s=[]; V2_s=[]; rho1_s=[]; rho2_s=[]; h1_s=[]; h2_s=[]; kappa_s=[]

        while t <= b.t_end and M > 1e-9:
            # fecha estado do tanque por (H,M,V)
            rho_mix = M / V
            h_mix = H / max(M, 1e-12)
            T = _solve_T_for_h_rho_T(fluid, h_mix, rho_mix, Ttr, Tcr)
            _, _, Psat = _mix_h_from_rho_T(fluid, rho_mix, T)
            P_bar = _Pa_to_bar(Psat)

            # resolver NHNE no orifício
            sim.c.P1_bar = P_bar
            sim.c.T1_K = T
            r = sim.solve()
            mdot = r.mdot

            # registo
            t_s.append(t); M_s.append(M); Pbar_s.append(P_bar); T_s.append(T)
            mdot_s.append(mdot); V2_s.append(r.V2)
            rho1_s.append(r.rho1); rho2_s.append(r.rho2); h1_s.append(r.h1); h2_s.append(r.h2)
            kappa_s.append(r.kappa)

            # balanços do tanque
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
            "kappa": kappa_s,
            "t_empty_s": (t_s[-1] if len(t_s)>0 and M<=1e-9 else None)
        }


# ---------------- main ----------------

if __name__ == "__main__":
    # 1) Ponto único NHNE com cinética
    case = Case(N_holes=12, D_hole_m=0.0015)
    r = ColdFlowNHNE(case).solve()
    print("Convergiu:", r.converged, "iters:", r.n_iters)
    print("mdot [kg/s]:", r.mdot, " | mdot_inc:", r.mdot_inc, " | mdot_HEM:", r.mdot_HEM, " | cap_HEM:", r.mdot_cap_HEM)
    print("V1 [m/s]:", r.V1, " V2 [m/s]:", r.V2, " kappa:", r.kappa)
    print("rho1:", r.rho1, " rho2:", r.rho2, " h1:", r.h1, " h2:", r.h2, " T2:", r.T2, "x2:", r.x2)

    # 2) Blowdown + CSV + plots
    bcase = BlowdownCase(N_holes=12, D_hole_m=0.0015)
    hist = BlowdownSimulator("NitrousOxide", bcase).run()

    # CSV
    out_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(out_dir, "blowdown_history.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["t_s","M_kg","P_bar","T_K","mdot_kg_s","V2_m_s",
                  "rho1_kg_m3","rho2_kg_m3","h1_J_kg","h2_J_kg","kappa"]
        w.writerow(header)
        for i in range(len(hist["t_s"])):
            w.writerow([
                hist["t_s"][i], hist["M_kg"][i], hist["P_bar"][i], hist["T_K"][i],
                hist["mdot_kg_s"][i], hist["V2_m_s"][i],
                hist["rho1_kg_m3"][i], hist["rho2_kg_m3"][i],
                hist["h1_J_kg"][i], hist["h2_J_kg"][i],
                hist["kappa"][i]
            ])
    print("CSV salvo em:", csv_path)
    if hist["t_empty_s"] is not None:
        print("t_empty_s:", hist["t_empty_s"])
    else:
        print("Aviso: tanque não esvaziou dentro de t_end.")

    # Plots mostrados
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
