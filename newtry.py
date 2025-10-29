# NHNE (Solomon) - Simulador de Blowdown N2O
# Requisitos: pip install numpy scipy CoolProp matplotlib

import numpy as np
from math import pi, sqrt
from dataclasses import dataclass
from scipy.optimize import root
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt

FLUID = "NitrousOxide"

# ---------------- UTILIDADES TERMODINÂMICAS ---------------- #

def _T_bounds():
    Tmin = float(PropsSI("Tmin", FLUID)) + 1.0
    Tcrit = float(PropsSI("Tcrit", FLUID)) - 1.0
    return Tmin, Tcrit

def _clip_T(T):
    T_lo, T_hi = _T_bounds()
    return float(max(min(T, T_hi), T_lo))

def state_DT(T, rho):
    T = _clip_T(T); rho = max(float(rho), 1e-9)
    p = PropsSI("P", "D", rho, "T", T, FLUID)
    h = PropsSI("H", "D", rho, "T", T, FLUID)
    s = PropsSI("S", "D", rho, "T", T, FLUID)
    return p, h, s

def _psat(T): return float(PropsSI("P","T",_clip_T(T),"Q",0,FLUID))
def _rhoL_sat(T): return float(PropsSI("D","T",_clip_T(T),"Q",0,FLUID))

def T_sat_at_P(P): return float(PropsSI("T","P",P,"Q",0,FLUID))

# ---------------- MODELOS DE CAUDAL ---------------- #

def m_dot_incompressible(Cd, Ac, rho1, P1, P2):
    dP = max(P1 - P2, 0.0)
    return Cd * Ac * sqrt(2.0 * rho1 * dP)

def m_dot_HEM(Cd, Ac, rho2, h1, h2):
    dh = max(h1 - h2, 0.0)
    return Cd * Ac * rho2 * sqrt(2.0 * dh)

def _Pvap_from_T(T):
    try:
        return float(PropsSI("P", "T", _clip_T(T), "Q", 0, FLUID))
    except:
        return float(PropsSI("P", "T", _clip_T(T), "Q", 0.5, FLUID))

def kappa(P1, P2, Pv1):
    return sqrt(max(P1 - P2, 0.0) / max(Pv1 - P2, 1e-6))

def flash_weight(P1, T, P2, Pv1, dP_flash=3e5):
    dP_sub = max(P1 - Pv1, 0.0)
    w = 1.0 - np.clip(dP_sub / (dP_flash + 1e-6), 0.0, 1.0)
    w2 = np.clip((Pv1 - P2) / max(P1 - P2, 1e-6), 0.0, 1.0)
    return max(w, w2)

def m_dot_NHNE(Cd, Ac, P1, P2, rho1, rho2, h1, h2, Pv1, T=None):
    k = kappa(P1, P2, Pv1)
    w_hem = 1.0 / (1.0 + k)
    if T is not None:
        w_hem = max(w_hem, flash_weight(P1, T, P2, Pv1))
    w_inc = 1.0 - w_hem
    minc = m_dot_incompressible(Cd, Ac, rho1, P1, P2)
    mhem = m_dot_HEM(Cd, Ac, rho2, h1, h2)
    return w_inc * minc + w_hem * mhem

# ---------------- CINÉTICA DA VÁLVULA ---------------- #

def Cd_time(Cd_nom, t, t_open=0.15):
    if t <= 0.0:
        return 0.0
    if t >= t_open:
        return Cd_nom
    x = t / t_open
    return Cd_nom * 0.5 * (1 - np.cos(np.pi * x))

# ---------------- CLASSES DE DADOS ---------------- #

@dataclass
class Tank:
    V: float
    M0: float
    T0: float

@dataclass
class Injector:
    Cd: float
    D: float
    P2: float
    N: int

# ---------------- MASSA INICIAL ---------------- #

def rho_liq_PT(T, P):
    """Densidade robusta do líquido: usa Q=0 em saturação e T,P acima."""
    T = _clip_T(T)
    Ps = _psat(T)
    # tolerância relativa de saturação
    if P <= Ps*(1.0 + 1e-5):
        # trata como saturado no lado líquido
        return float(PropsSI("D", "T", T, "Q", 0, FLUID))
    # líquido comprimido
    return float(PropsSI("D", "T", T, "P", P, FLUID))

def M0_from_fill(V_total, fL, T0, P0):
    """Massa inicial com proteção a saturação."""
    T0 = _clip_T(T0)
    Ps = _psat(T0)
    # se P0 < Ps(T0), arrefece até T_s(P0); se P0 ≈ Ps(T0), usa Q=0
    if P0 < Ps*(1.0 - 1e-5):
        T_use = float(PropsSI("T", "P", P0, "Q", 0, FLUID))
        rhoL = rho_liq_PT(T_use, P0)
    else:
        rhoL = float(PropsSI("D", "T", T0, "Q", 0, FLUID)) if abs(P0-Ps)/Ps <= 1e-5 \
               else rho_liq_PT(T0, P0)
    return rhoL * fL * V_total


# ---------------- SIMULAÇÃO ---------------- #

def run_blowdown(tank, inj, mode_tank="pressurized",
                 P_const=5.0e6, dt=0.01, t_stop=30.0, mode="isentropic"):
    Ac = inj.N * 0.25 * pi * inj.D**2
    M = float(tank.M0)
    T = _clip_T(float(tank.T0))
    V = float(tank.V)
    t = 0.0
    hist = {k: [] for k in ["t", "M", "T", "p", "mdot"]}

    while t <= t_stop and M > 1e-6:
                # pressão do tanque
        if mode_tank == "pressurized":
            P1 = P_const
        elif mode_tank == "autopressurized":
            P1 = _psat(T)
        # evitar P==Psat para chamadas D(T,P)
        P1 = max(P1, _psat(T) * (1.0 + 1e-5))

        rhoL = rho_liq_PT(T, P1)
        p1, h1, s1 = state_DT(T, rhoL)
        Pv1 = _Pvap_from_T(T)

        # downstream (isentropic)
        def downstream_state(P2, s1):
            T_lo, T_hi = _T_bounds()
            x0 = np.array([0.5*(T_lo+T_hi), rhoL])
            def F(v):
                T2, rho2 = _clip_T(v[0]), max(v[1], 1e-8)
                p = PropsSI("P","D",rho2,"T",T2,FLUID)
                s = PropsSI("S","D",rho2,"T",T2,FLUID)
                return [p-P2, s-s1]
            sol = root(F, x0)
            T2, rho2 = _clip_T(sol.x[0]), max(sol.x[1], 1e-8)
            h2 = PropsSI("H","D",rho2,"T",T2,FLUID)
            return T2, rho2, h2

        T2, rho2, h2 = downstream_state(inj.P2, s1)
        Cd_eff = Cd_time(inj.Cd, t)
        mdot = m_dot_NHNE(Cd_eff, Ac, P1, inj.P2, rhoL, rho2, h1, h2, Pv1, T=T)

        # atualização massa/energia
        M_new = max(M - mdot*dt, 1e-9)
        H_old = M * h1
        H_new = H_old - h1 * mdot * dt
        h_new = H_new / M_new
        Cp = PropsSI("Cpmass","T",T,"P",P1,FLUID)
        T_new = _clip_T(T + (h_new - h1)/Cp)

        hist["t"].append(t)
        hist["M"].append(M)
        hist["T"].append(T)
        hist["p"].append(P1)
        hist["mdot"].append(mdot)

        M, T = M_new, T_new
        t += dt
        if P1 <= inj.P2:
            break

    for k in hist:
        hist[k] = np.array(hist[k], dtype=float)
    return hist, t

# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    V = 0.009
    fL = 0.9
    T0 = 283.15
    P0 = 4.4e6  # 44 bar
    M0 = M0_from_fill(V, fL, T0, P0)

    tank = Tank(V=V, M0=M0, T0=T0)
    inj  = Injector(Cd=0.67, D=0.0015, P2=1e5, N=12)

    # --------- escolher modo ---------
    mode_tank = "autopressurized"   # "pressurized" ou "autopressurized"
    P_const   = 5.0e6               # só usado no modo pressurized
    # ---------------------------------

    out, t_empty = run_blowdown(tank, inj, mode_tank=mode_tank,
                                P_const=P_const, dt=0.01, t_stop=30.0)

    print(f"Modo do tanque: {mode_tank}")
    print(f"P0={P0/1e5:.1f} bar | T0={T0:.1f} K | M0={M0:.3f} kg")
    print(f"N={inj.N} furos, D={inj.D*1e3:.2f} mm")
    print(f"Tempo até esvaziar: {t_empty:.2f} s")
    print(f"Débito médio: {out['mdot'].mean():.3f} kg/s")

    plt.plot(out["t"], out["mdot"], "r", label=f"{mode_tank}")
    plt.title("Mass Flow vs Time")
    plt.xlabel("Time [s]")
    plt.ylabel("Mass Flow [kg/s]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
