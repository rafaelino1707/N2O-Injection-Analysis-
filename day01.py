# solomon_blowdown.py
# Two-Phase Enthalpy Model + NHNE (estilo Brian Solomon, caso CO2 cold-flow)

import math
from dataclasses import dataclass
from typing import Tuple, Literal

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from CoolProp.CoolProp import PropsSI

# ============================================================================
# Inputs: caso "Solomon" para comparação 1:1 (CO2 cold-flow)
# ============================================================================
FLUID: Literal["CarbonDioxide"] = "CarbonDioxide"  # CO2

# Tanque
V_tank        = 0.0498      # [m^3]

# Estado inicial por (M0, T0) — como no relatório
M0            = 12.76       # [kg]
T0            = 274.25      # [K]
# (opção alternativa usada noutro caso do relatório)
# M0, T0     = 10.15, 277.90

# Injetor (um único orifício 0.178 in)
N_holes       = 1
d_hole        = 0.178 * 0.0254           # [m] -> 0.0045212 m
A_c           = N_holes * math.pi/4.0 * d_hole**2
Cd            = 0.80

# Jusante ambiente do ensaio (Logan ~ 85.9 kPa)
P_down        = 85_900      # [Pa]

# Expansão no injetor
expansion_mode = "isentropic"  # "isentropic" ou "adiabatic"

# Integração temporal
t_stop        = 30.0        # [s]
dt            = 0.5         # [s]
PRINT_EVERY_S = 2.0         # [s]

# NHNE: densidade usada na perna incompressível
use_rhoL_in_inc = True      # perna incompressível com rhoL(T1), como em Solomon

# ============================================================================
# Utilitários de saturação e mistura
# ============================================================================
@dataclass
class SatProps:
    P: float; T: float
    rhoL: float; rhoV: float
    hL: float; hV: float
    sL: float; sV: float

def sat_props_T(T: float) -> SatProps:
    T = float(T)
    P    = PropsSI("P","T",T,"Q",0,FLUID)
    rhoL = PropsSI("D","T",T,"Q",0,FLUID)
    rhoV = PropsSI("D","T",T,"Q",1,FLUID)
    hL   = PropsSI("H","T",T,"Q",0,FLUID)
    hV   = PropsSI("H","T",T,"Q",1,FLUID)
    sL   = PropsSI("S","T",T,"Q",0,FLUID)
    sV   = PropsSI("S","T",T,"Q",1,FLUID)
    return SatProps(P,T,rhoL,rhoV,hL,hV,sL,sV)

def mix_from_Tx(T: float, x: float) -> Tuple[float,float,float,float,float]:
    sp  = sat_props_T(T)
    x   = np.clip(x, 0.0, 1.0)
    rho = 1.0/(x/sp.rhoV + (1.0-x)/sp.rhoL)
    h   = x*sp.hV + (1.0-x)*sp.hL
    s   = x*sp.sV + (1.0-x)*sp.sL
    P   = sp.P
    return P, rho, h, s, sp.rhoL  # devolve também rhoL(T) para NHNE

def x_from_rho_T(rho: float, T: float) -> float:
    sp = sat_props_T(T)
    v  = 1.0/rho
    vL = 1.0/sp.rhoL
    vV = 1.0/sp.rhoV
    return float(np.clip((v - vL)/(vV - vL), 0.0, 1.0))

# ============================================================================
# #Least squares solving — montante (fecha estado no tanque)
# Resolver [T, x] tal que:
#   (1) rho(T,x) = M/V
#   (2) h(T,x)   = H/M
# ============================================================================
def solve_upstream_TX_from_MH(M: float, H: float, T_guess: float) -> Tuple[float,float,float,float,float,float]:
    rho_target = M / V_tank
    h_target   = H / M

    Tmin = PropsSI("Tmin", FLUID) + 0.5
    Tcrit= PropsSI("Tcrit", FLUID) - 0.5

    def residuals(y):
        T, x = y
        T = np.clip(T, Tmin, Tcrit)
        x = np.clip(x, 0.0, 1.0)
        P, rho, h, s, rhoL = mix_from_Tx(T, x)
        return np.array([rho - rho_target, h - h_target], dtype=float)

    # palpites limpos para CO2 cold-flow
    x0 = x_from_rho_T(M / V_tank, T_guess)
    y0 = np.array([T_guess, x0], dtype=float)

    sol = least_squares(residuals, y0, method="trf",
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=300)

    T, x = sol.x
    T = float(np.clip(T, Tmin, Tcrit))
    x = float(np.clip(x, 0.0, 1.0))
    P, rho, h, s, rhoL = mix_from_Tx(T, x)
    return T, x, P, rho, h, s

# ============================================================================
# #Least squares solving — jusante (estado 2)
# Isentrópico: resolver [T2, rho2] tal que P(T2,rho2)=P2 e S(T2,rho2)=s1
# Adiabático:  resolver [T2, rho2] tal que P(T2,rho2)=P2 e H(T2,rho2)=h1
# ============================================================================
def solve_downstream_state(P2: float, h1: float, s1: float, T1: float, mode: str) -> Tuple[float,float,float]:
    Tmin = PropsSI("Tmin", FLUID) + 0.5
    Tcrit= PropsSI("Tcrit", FLUID) - 0.5

    rhoL1 = sat_props_T(T1).rhoL
    y0 = np.array([T1, max(0.5*rhoL1, 1e-3)], dtype=float)

    def residuals_isentropic(y):
        T, rho = y
        T = np.clip(T, Tmin, Tcrit); rho = max(float(rho), 1e-3)
        P = PropsSI("P","T",T,"D",rho,FLUID)
        S = PropsSI("S","T",T,"D",rho,FLUID)
        return np.array([P - P2, S - s1], dtype=float)

    def residuals_adiabatic(y):
        T, rho = y
        T = np.clip(T, Tmin, Tcrit); rho = max(float(rho), 1e-3)
        P = PropsSI("P","T",T,"D",rho,FLUID)
        H = PropsSI("H","T",T,"D",rho,FLUID)
        return np.array([P - P2, H - h1], dtype=float)

    res_fun = residuals_isentropic if mode == "isentropic" else residuals_adiabatic
    sol = least_squares(res_fun, y0, method="trf",
                        ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=500)

    T2, rho2 = sol.x
    T2 = float(T2)
    rho2 = float(max(rho2, 1e-3))
    h2 = PropsSI("H","T",T2,"D",rho2,FLUID)
    return T2, rho2, h2

# ============================================================================
# #Initial condition — construir H0 e x0 a partir de (M0, T0)
# ============================================================================
def initial_state_from_M0_T0():
    rho0 = M0 / V_tank
    x0   = x_from_rho_T(rho0, T0)
    P0, _, h0, _, _ = mix_from_Tx(T0, x0)
    H0  = M0 * h0
    return M0, H0, T0, x0, P0

# ============================================================================
# #Two phase enthalpy model — integração temporal + NHNE
# ============================================================================
def run():
    Tmin = PropsSI("Tmin", FLUID) + 0.5
    Tcrit= PropsSI("Tcrit", FLUID) - 0.5

    # estado inicial por (M0, T0)
    M, H, Tguess, x_init, P_init = initial_state_from_M0_T0()

    # armazenamento
    nstep = int(t_stop/dt) + 1
    t_arr    = np.linspace(0.0, dt*(nstep-1), nstep)
    P_arr    = np.zeros(nstep)
    T_arr    = np.zeros(nstep)
    M_arr    = np.zeros(nstep)
    x_arr    = np.zeros(nstep)
    mdot_arr = np.zeros(nstep)

    for k, t in enumerate(t_arr):
        # ---- montante via LSQ: fecha [T1,x1] para (rho=M/V, h=H/M)
        T1, x1, P1, rho1, h1, s1 = solve_upstream_TX_from_MH(M, H, Tguess)

        # ---- jusante via LSQ 2D
        T2, rho2, h2 = solve_downstream_state(P_down, h1, s1, T1, expansion_mode)

        # ---- NHNE
        Pv1 = sat_props_T(T1).P  # em saturação, Pv1 = P1
        kappa = math.sqrt(max(P1 - P_down, 1.0) / max(Pv1 - P_down, 1.0))
        W = 1.0/(1.0 + kappa)    # peso HEM

        rho_inc = sat_props_T(T1).rhoL if use_rhoL_in_inc else rho1
        mdot_inc = Cd * A_c * math.sqrt(2.0 * rho_inc * max(P1 - P_down, 0.0))
        mdot_HEM = Cd * A_c * rho2 * math.sqrt(max(2.0*(h1 - h2), 0.0))
        mdot = (1.0 - W) * mdot_inc + W * mdot_HEM

        # ---- registo
        P_arr[k]    = P1
        T_arr[k]    = T1
        M_arr[k]    = M
        x_arr[k]    = x1
        mdot_arr[k] = mdot

        # ---- logging periódico
        if PRINT_EVERY_S:
            stride = max(int(PRINT_EVERY_S/dt), 1)
            if k % stride == 0:
                print(f"t={t:6.2f}s  P={P1/1e6:6.3f} MPa  T={T1:7.2f} K  x={x1:5.3f}  "
                      f"rho1={rho1:7.1f}  mdot_inc={mdot_inc:6.3f}  mdot_HEM={mdot_HEM:6.3f}  "
                      f"kappa={kappa:4.2f}  W={W:5.3f}  mdot={mdot:6.3f}")

        # ---- avanço temporal conservando H e M
        M_new = max(M - mdot*dt, 1e-9)
        H_new = H - h1*mdot*dt

        M, H = M_new, H_new
        Tguess = float(np.clip(T1, Tmin, Tcrit))

        if M <= 1e-8:
            last = k+1
            t_arr    = t_arr[:last]
            P_arr    = P_arr[:last]
            T_arr    = T_arr[:last]
            M_arr    = M_arr[:last]
            x_arr    = x_arr[:last]
            mdot_arr = mdot_arr[:last]
            break

    return t_arr, P_arr, T_arr, M_arr, x_arr, mdot_arr

# ============================================================================
# #Plots e export
# ============================================================================
def main():
    t, P, T, M, x, mdot = run()

    # Plots
    plt.figure(); plt.plot(t, P/1e6); plt.grid(True, ls=":")
    plt.xlabel("t [s]"); plt.ylabel("Tank pressure [MPa]"); plt.title(f"{FLUID} blowdown: pressure")

    plt.figure(); plt.plot(t, T); plt.grid(True, ls=":")
    plt.xlabel("t [s]"); plt.ylabel("Tank temperature [K]"); plt.title(f"{FLUID} blowdown: temperature")

    plt.figure(); plt.plot(t, M); plt.grid(True, ls=":")
    plt.xlabel("t [s]"); plt.ylabel("Tank mass [kg]"); plt.title(f"{FLUID} blowdown: mass")

    plt.figure(); plt.plot(t, mdot); plt.grid(True, ls=":")
    plt.xlabel("t [s]"); plt.ylabel("Mass flow [kg/s]"); plt.title(f"{FLUID} blowdown: $\dot m$ (NHNE)")
    plt.ylim(0, 1.6)

    plt.tight_layout()
    plt.show()

    # CSV + NPZ
    header = "t [s],P [Pa],T [K],M [kg],x [-],mdot [kg/s]"
    data = np.column_stack([t, P, T, M, x, mdot])
    np.savetxt("solomon_blowdown_results.csv", data, delimiter=",", fmt="%.6e", header=header, comments="")
    np.savez("solomon_blowdown_results.npz", t=t, P=P, T=T, M=M, x=x, mdot=mdot)

if __name__ == "__main__":
    main()
