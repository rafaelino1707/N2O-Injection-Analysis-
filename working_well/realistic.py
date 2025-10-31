# Blowdown NHNE with 90% liquid fill and tank pressurized to 50 bar (no partial-pressure sum)
# pip install CoolProp numpy scipy matplotlib

import math
from dataclasses import dataclass
from typing import Tuple, Literal

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

# ---------------------------- Fluid selection ---------------------------- #
FLUID: Literal["CarbonDioxide","NitrousOxide"] = "NitrousOxide"  # or "NitrousOxide"

# ---------------------------- Tank + injector inputs ---------------------------- #
V_tank   = 0.009   # m^3
fill_vol_frac = 0.90  # 90% volume of liquid N2O/CO2
P0_tot  = 40.0e5     # Pa (tank pressurized to 50 bar by N2; use as initial total pressure)
# Injector
d_inj = 0.003 # m
N = 12
A_c      = N*math.pi/4.0 * d_inj**2  # m^2
Cd       = 0.67              # discharge coefficient
P_amb    = 3e6            # Pa (ambient)
inj_mode = 1                 # 1=isentrope (s2=s1), 2=adiabatic (h2=h1)

# ---------------------------- NHNE options ---------------------------- #
rho_inc_liq  = True  # incompressible branch uses rhoL(T)

# ---------------------------- Time marching ---------------------------- #
t_stop = 20.0  # s
dt     = 0.5   # s
PRINT_EVERY_S = 2.0

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

# ---------------------------- Bracketing ranges ---------------------------- #
T_min = PropsSI("Tmin", FLUID) + 1e-3
T_crit= PropsSI("Tcrit", FLUID) - 1e-3
P_trip= PropsSI("ptriple", FLUID)

# ---------------------------- Initialize from P0_tot and 90% volume fill ---------------------------- #
# "Pressurized to 50 bar" => set initial state on saturation line with P = 50 bar
sp0   = sat_props_P(P0_tot)
T0    = sp0.T
# Mass from volume fill: M0 = V_liq*rhoL + V_gas*rhoV
V_liq0 = fill_vol_frac * V_tank
V_gas0 = V_tank - V_liq0
M0     = V_liq0*sp0.rhoL + V_gas0*sp0.rhoV
rho    = M0 / V_tank
# Thermo state on the dome at (T0,rho)
P, x, h, s = f_T_rho(T0, rho)
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
    # upstream "two-phase" pressure from the dome (no partial sums)
    P1 = sp_up.P
    x  = quality_from_rho(T, rho)
    h1 = h_mix(T, x)
    s1 = s_mix(T, x)

    # Downstream at P_amb
    if inj_mode == 1:
        try:
            T2  = PropsSI("T","P",P_amb,"S",s1,FLUID)
            h2  = PropsSI("H","T",T2,"P",P_amb,FLUID)
            rho2= PropsSI("D","T",T2,"P",P_amb,FLUID)
        except Exception:
            sp_dn = sat_props_P(P_trip)
            sL, sV = sp_dn.sL, sp_dn.sV
            x2 = float(np.clip(0.0 if sV==sL else (s1 - sL)/(sV - sL), 0.0, 1.0))
            h2 = x2*sp_dn.hV + (1.0-x2)*sp_dn.hL
            rho2 = 1.0/(x2/sp_dn.rhoV + (1.0-x2)/sp_dn.rhoL)
    else:
        try:
            T2  = PropsSI("T","P",P_amb,"H",h1,FLUID)
            h2  = h1
            rho2= PropsSI("D","T",T2,"P",P_amb,FLUID)
        except Exception:
            sp_dn = sat_props_P(P_trip)
            hL,hV = sp_dn.hL, sp_dn.hV
            x2 = float(np.clip(0.0 if hV==hL else (h1 - hL)/(hV - hL), 0.0, 1.0))
            h2 = x2*sp_dn.hV + (1.0-x2)*sp_dn.hL
            rho2 = 1.0/(x2/sp_dn.rhoV + (1.0-x2)/sp_dn.rhoL)

    # NHNE weighting (Spencer & Stanford eq.: kappa = sqrt((P1-P2)/(Pv1-P2)))
    # Here Pv1 = saturation pressure at T -> equals P1; thus kappa -> 1.0 identically.
    dP_num = max(P1 - P_amb, 1.0)
    dP_den = max(sp_up.P - P_amb, 1.0)  # = dP_num, so kappa=1
    kappa  = math.sqrt(dP_num / dP_den)
    W      = 1.0/(1.0 + kappa)

    rho_inc_use = sp_up.rhoL if rho_inc_liq else rho
    mdot_inc = Cd * A_c * math.sqrt(2.0 * rho_inc_use * max(P1 - P_amb, 0.0))
    mdot_HEM = Cd * A_c * rho2 * math.sqrt(2.0 * max(h1 - h2, 0.0))

    # Two-phase vs gas-only

    mdot = (1.0 - W) * mdot_inc + W * mdot_HEM


    # Record
    M_arr[i]    = M
    T_arr[i]    = T
    P_arr[i]    = P1
    x_arr[i]    = x
    mdot_arr[i] = mdot

    # Euler update of mass and total enthalpy (N2 mass not tracked; pressurization only set P0)
    M_new   = max(M - mdot * dt, 1e-9)
    H_total = H_total - h1 * mdot * dt
    rho     = M_new / V_tank
    h       = H_total / M_new
    M       = M_new

    if M <=0:
        break

    # Console logging
    if PRINT_EVERY_S is not None:
        klog = max(int(PRINT_EVERY_S/dt), 1)
        if i % klog == 0:
            print(f"t={t:.1f}  P={P1/1e6:.3f} MPa  T={T:.2f} K  x={x:.3f}  rhoL={sp_up.rhoL:.1f}  mdot_inc={mdot_inc:.3f}  mdot_HEM={mdot_HEM:.3f}  k={kappa:.2f}  mdot={mdot:.3f}")

# ---------------------------- Plots ---------------------------- #
plt.figure(); plt.plot(t_arr, P_arr/1e6); plt.grid(True, ls=":")
plt.xlabel("t [s]"); plt.ylabel("Tank pressure [MPa]"); plt.title(f"{FLUID} blowdown: pressure")

plt.figure(); plt.plot(t_arr, T_arr); plt.grid(True, ls=":")
plt.xlabel("t [s]"); plt.ylabel("Tank temperature [K]"); plt.title(f"{FLUID} blowdown: temperature")

plt.figure(); plt.plot(t_arr, M_arr); plt.grid(True, ls=":")
plt.xlabel("t [s]"); plt.ylabel("Tank mass [kg]"); plt.title(f"{FLUID} blowdown: mass")

plt.figure(); plt.plot(t_arr, mdot_arr); plt.ylim(0,1.6); plt.grid(True, ls=":")
plt.xlabel("t [s]"); plt.ylabel("Mass flow rate [kg/s]"); plt.title(f"{FLUID} blowdown: $\\dot m$ (NHNE)")

plt.tight_layout()
plt.show()

# ---------------------------- Export ---------------------------- #
header = "t [s],P [Pa],T [K],M [kg],x [-],mdot [kg/s]"
data = np.column_stack([t_arr, P_arr, T_arr, M_arr, x_arr, mdot_arr])
np.savetxt("solomon_blowdown_results.csv", data, delimiter=",", fmt="%.6e", header=header, comments="")
np.savez("solomon_blowdown_results.npz", t=t_arr, P=P_arr, T=T_arr, M=M_arr, x=x_arr, mdot=mdot_arr)
