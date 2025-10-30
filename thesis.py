# Solomon-style two-phase blowdown with NHNE weighting (Appendix B.1 replica)
# Requirements: pip install CoolProp numpy scipy matplotlib

import math
from dataclasses import dataclass
from typing import Tuple, Literal

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from CoolProp.CoolProp import PropsSI

# ---------------------------- Fluid selection ---------------------------- #
# Appendix B.1 uses CO2
FLUID: Literal["CarbonDioxide","NitrousOxide"] = "CarbonDioxide"  # set to "NitrousOxide" if desired

# ---------------------------- Tank + injector inputs ---------------------------- #
# Solomon Appendix B.1 default case
V_tank   = 0.0498   # m^3
M0       = 12.76    # kg
T0       = 274.25   # K
# Injector
d_inj_in = 0.178             # inches (diameter)
d_inj    = d_inj_in * 0.0254 # m
A_c      = math.pi/4.0 * d_inj**2  # m^2
Cd       = 0.8               # discharge coefficient
P_amb    = 85.9e3            # Pa (85.9 kPa)
inj_mode = 1                 # 1=isentrope (s2=s1), 2=adiabatic (h2=h1)

# ---------------------------- NHNE options ---------------------------- #
# Appendix B.1 uses constant k=0.69 and rho_L in incompressible branch

rho_inc_liq  = True

# ---------------------------- Time marching ---------------------------- #
t_stop = 30.0  # s
dt     = 0.5   # s

# Optional console logging
PRINT_EVERY_S = 2.0  # set None to disable

# ---------------------------- Utility: saturation props ---------------------------- #
@dataclass
class SatProps:
    P: float; T: float
    rhoL: float; rhoV: float
    hL: float; hV: float
    sL: float; sV: float

def sat_props_T(T: float) -> SatProps:
    T  = float(T)
    P  = PropsSI("P","T",T,"Q",0,FLUID)
    rhoL = PropsSI("D","T",T,"Q",0,FLUID)
    rhoV = PropsSI("D","T",T,"Q",1,FLUID)
    hL = PropsSI("H","T",T,"Q",0,FLUID)
    hV = PropsSI("H","T",T,"Q",1,FLUID)
    sL = PropsSI("S","T",T,"Q",0,FLUID)
    sV = PropsSI("S","T",T,"Q",1,FLUID)
    return SatProps(P,T,rhoL,rhoV,hL,hV,sL,sV)

def sat_props_P(P: float) -> SatProps:
    T  = PropsSI("T","P",P,"Q",0,FLUID)
    return sat_props_T(T)

# ---------------------------- Mixture relations in two-phase dome ---------------------------- #
def quality_from_rho(T: float, rho: float) -> float:
    sp = sat_props_T(T)
    x = (sp.rhoV/rho)*(sp.rhoL - rho)/(sp.rhoL - sp.rhoV)
    return float(np.clip(x, 0.0, 1.0))

def rho_from_quality(T: float, x: float) -> float:
    sp = sat_props_T(T)
    v = x/sp.rhoV + (1.0 - x)/sp.rhoL
    return 1.0/v

def h_mix(T: float, x: float) -> float:
    sp = sat_props_T(T)
    return x*sp.hV + (1.0 - x)*sp.hL

def s_mix(T: float, x: float) -> float:
    sp = sat_props_T(T)
    return x*sp.sV + (1.0 - x)*sp.sL

def f_T_rho(T: float, rho: float) -> Tuple[float,float,float,float]:
    sp = sat_props_T(T)
    x  = quality_from_rho(T, rho)
    h  = h_mix(T, x)
    s  = s_mix(T, x)
    return sp.P, x, h, s

def M0_fill(V:float, P:float, T:float, fL:float):
    rhoL = PropsSI("D", "P", P, "T", T)
    VL = V * fL
    return rho * VL

# ---------------------------- Bracketing ranges ---------------------------- #
T_min = PropsSI("Tmin", FLUID) + 1e-3
T_crit= PropsSI("Tcrit", FLUID) - 1e-3
P_trip= PropsSI("ptriple", FLUID)

# ---------------------------- Initialize state from M0, T0 ---------------------------- #
#rho = M0 / V_tank
rho_initial = PropsSI("D", "P", SatProps.P, "T", T0)
M0 = V_tank * rho_initial
P, x, h, s = f_T_rho(T0, rho_initial)
H_total = M0 * h
M = M0
T = T0

# ---------------------------- Storage ---------------------------- #
N = int(t_stop / dt) + 1
t_arr   = np.linspace(0.0, t_stop, N)
M_arr   = np.zeros(N)
T_arr   = np.zeros(N)
P_arr   = np.zeros(N)
x_arr   = np.zeros(N)
mdot_arr= np.zeros(N)

# ---------------------------- March in time ---------------------------- #
for i, t in enumerate(t_arr):
    # Solve T such that h_mix(T, x(T,rho)) = h (at current rho, total h)
    def g_T(Tguess: float) -> float:
        xg = quality_from_rho(Tguess, rho)
        return h_mix(Tguess, xg) - h

    T = float(np.clip(T, T_min+1e-2, T_crit-1e-2))
    try:
        T = brentq(g_T, T_min+1e-2, T_crit-1e-2, maxiter=100)
    except ValueError:
        pass

    sp_up = sat_props_T(T)
    P = sp_up.P
    x = quality_from_rho(T, rho)
    h1 = h_mix(T, x)
    s1 = s_mix(T, x)


    # Downstream at P_amb: use real-gas flash to avoid PQ outside dome
    if inj_mode == 1:
        try:
            T2 = PropsSI("T","P",P_amb,"S",s1,FLUID)
            h2 = PropsSI("H","T",T2,"P",P_amb,FLUID)
            rho2 = PropsSI("D","T",T2,"P",P_amb,FLUID)
        except Exception:
            sp_dn = sat_props_P(P_trip)
            sL, sV = sp_dn.sL, sp_dn.sV
            x2 = float(np.clip(0.0 if sV==sL else (s1 - sL)/(sV - sL), 0.0, 1.0))
            h2 = h_mix(sp_dn.T, x2)
            rho2 = rho_from_quality(sp_dn.T, x2)
    else:
        try:
            T2 = PropsSI("T","P",P_amb,"H",h1,FLUID)
            h2 = h1
            rho2 = PropsSI("D","T",T2,"P",P_amb,FLUID)
        except Exception:
            sp_dn = sat_props_P(P_trip)
            hL, hV = sp_dn.hL, sp_dn.hV
            x2 = float(np.clip(0.0 if hV==hL else (h1 - hL)/(hV - hL), 0.0, 1.0))
            h2 = h_mix(sp_dn.T, x2)
            rho2 = rho_from_quality(sp_dn.T, x2)

    
    # NHNE weighting
    dP1 = max(P - P_amb, 0.0)
    dPv = max(sp_up.P - P_amb, 1.0)  # avoid zero
    kappa = math.sqrt(dP1 / dPv)

    rho_inc_use = sp_up.rhoL if rho_inc_liq else rho
    mdot_inc = Cd * A_c * math.sqrt(2.0 * rho_inc_use * dP1)
    dh = max(h1 - h2, 0.0)
    mdot_HEM = Cd * A_c * rho2 * math.sqrt(2.0 * dh)

    if x <=1:
        # Two-phase NHNE
        w_hem = 1.0 / (1.0 + kappa)
        w_inc = 1.0 - w_hem
        mdot = w_inc * mdot_inc + w_hem * mdot_HEM
    else:
        # Pure gas: choked flow model
        R  = PropsSI("gas_constant", FLUID)
        cp = PropsSI("CPMASS", "T", T, "P", P, FLUID)
        cv = PropsSI("CVMASS", "T", T, "P", P, FLUID)
        gamma = cp / cv
        crit = (2.0/(gamma+1.0))**(gamma/(gamma-1.0))
        if (P_amb/P) <= crit:
            G = (P/np.sqrt(T)) * np.sqrt(gamma/R) * (2.0/(gamma+1.0))**((gamma+1.0)/(2.0*(gamma-1.0)))
        else:
            G = (P/np.sqrt(T)) * np.sqrt(gamma/R)*(P_amb/P)**(1.0/gamma)*np.sqrt(2.0*gamma/(gamma-1.0)*(1.0-(P_amb/P)**((gamma-1.0)/gamma)))
        mdot = Cd * A_c * G




    # Record
    M_arr[i]    = M
    T_arr[i]    = T
    P_arr[i]    = P
    x_arr[i]    = x
    mdot_arr[i] = mdot

    # Euler update of mass and total enthalpy
    M_new = max(M - mdot * dt, 1e-9)
    H_total = H_total - h1 * mdot * dt
    rho   = M_new / V_tank
    h     = H_total / M_new
    M     = M_new

    # Console logging
    if PRINT_EVERY_S is not None:
        k = max(int(PRINT_EVERY_S/dt), 1)
        if i % k == 0:
            print(f"t={t:.1f}  P={P/1e6:.3f} MPa  T={T:.2f} K  x={x:.3f}  rhoL={sp_up.rhoL:.1f}  mdot_inc={mdot_inc:.3f}  mdot_HEM={mdot_HEM:.3f}  k={kappa:.2f}  mdot={mdot:.3f}")

# ---------------------------- Plots ---------------------------- #
plt.figure(); plt.plot(t_arr, P_arr/1e6)
plt.xlabel("t [s]"); plt.ylabel("Tank pressure [MPa]"); plt.title(f"{FLUID} blowdown: pressure")

plt.figure(); plt.plot(t_arr, T_arr)
plt.xlabel("t [s]"); plt.ylabel("Tank temperature [K]"); plt.title(f"{FLUID} blowdown: temperature")

plt.figure(); plt.plot(t_arr, M_arr)
plt.xlabel("t [s]"); plt.ylabel("Tank mass [kg]"); plt.title(f"{FLUID} blowdown: mass")

plt.figure(); plt.plot(t_arr, mdot_arr)
plt.ylim(0,1.6)
plt.xlabel("t [s]"); plt.ylabel("Mass flow rate [kg/s]"); plt.title(f"{FLUID} blowdown: $\dot m$ (NHNE)")

plt.tight_layout()
plt.show()

# ---------------------------- Export ---------------------------- #
# ---------------------------- Export ---------------------------- #
header = "t [s],P [Pa],T [K],M [kg],x [-],mdot [kg/s]"
data = np.column_stack([t_arr, P_arr, T_arr, M_arr, x_arr, mdot_arr])
np.savetxt(
    "solomon_blowdown_results.csv",
    data,
    delimiter=",",
    fmt="%.6e",
    header=header,
    comments=""
)

# Também guarda em NPZ
np.savez(
    "solomon_blowdown_results.npz",
    t=t_arr, P=P_arr, T=T_arr, M=M_arr, x=x_arr, mdot=mdot_arr
)

