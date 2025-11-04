# NHNE simples: m_dot vs P_tank, com uma linha por temperatura
# Requisitos: pip install CoolProp numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from math import pi

FLUID = "NitrousOxide"

# ---- Geometria / placa ----
N_holes = 32
D_hole  = 1.5e-3        # m
Cd      = 0.67          # geométrico (constante neste modelo)
A_hole  = 0.25*pi*D_hole**2
A_tot   = N_holes * A_hole

# ---- Câmara (jusante) ----
P_cham_bar = 30.0
Pa_per_bar = 1e5
P2 = P_cham_bar * Pa_per_bar

# ---- Varreduras ----
P_range_bar = np.linspace(25.0, 50.0, 11)   # eixo X
T_list_K    = [270, 275, 280, 285, 290, 295]  # uma linha por T

EPS_REL = 1e-6

def is_near_saturation(P, T, fluid):
    Psat = PropsSI("P","T",T,"Q",0,fluid)
    return abs(P - Psat) <= max(1.0, EPS_REL*P)

def rho_sat_mixture_from_h(P, T, h, fluid):
    rhoL = PropsSI("D","T",T,"Q",0,fluid)
    rhoV = PropsSI("D","T",T,"Q",1,fluid)
    hL   = PropsSI("H","T",T,"Q",0,fluid)
    hV   = PropsSI("H","T",T,"Q",1,fluid)
    x = 0.0 if hV == hL else np.clip((h - hL)/max(hV - hL, 1e-12), 0.0, 1.0)
    rho_mix = 1.0 / (x/rhoV + (1.0 - x)/rhoL)
    return rho_mix, x

def rho_from_PT_safe(P, T, fluid, h_hint=None):
    try:
        return PropsSI("D","P",P,"T",T,fluid), None
    except Exception:
        pass
    if is_near_saturation(P, T, fluid):
        if h_hint is None:
            h_hint = PropsSI("H","P",P,"T",T,fluid)
        rho_mix, x = rho_sat_mixture_from_h(P, T, h_hint, fluid)
        return rho_mix, x
    return PropsSI("D","P",P*(1.0 - EPS_REL),"T",T*(1.0 - EPS_REL),fluid), None

def mdot_incompressible_per_hole(A, Cd, rho_l, dP):
    return Cd * A * np.sqrt(2.0 * rho_l * dP)

def mdot_HEM_true_per_hole(A, Cd, P1, T1, P2, P_sat):
    h1 = PropsSI("H","T",T1,"P",max(P1, 1.01*P_sat), FLUID)
    s1 = PropsSI("S","T",T1,"P",max(P1, 1.01*P_sat), FLUID)
    try:
        T2 = PropsSI("T","P",P2,"S",s1,FLUID)
        h2 = PropsSI("H","P",P2,"T",T2,FLUID)
    except Exception:
        T2 = PropsSI("T","P",P2,"Q",0,FLUID)
        hL = PropsSI("H","T",T2,"Q",0,FLUID)
        hV = PropsSI("H","T",T2,"Q",1,FLUID)
        h2 = 0.5*(hL + hV)
    rho2, _ = rho_from_PT_safe(P2, T2, FLUID, h_hint=h2)
    dh = max(h1 - h2, 0.0)
    return Cd * A * rho2 * np.sqrt(2.0 * dh)

def mdot_NHNE_total(P1, T1):
    P_sat = PropsSI("P","T",T1,"Q",0,FLUID)
    rho_l = PropsSI("D","T",T1,"P",max(P1, 1.01*P_sat), FLUID)
    dP = max(P1 - P2, 0.0)

    m_inc_hole = mdot_incompressible_per_hole(A_hole, Cd, rho_l, dP)
    m_hem_hole = mdot_HEM_true_per_hole(A_hole, Cd, P1, T1, P2, P_sat)

    kappa = np.sqrt(max(P1-P2,0.0)/max(P_sat-P2,1.0))
    W = 1.0/(1.0 + kappa)  # peso do HEM
    m_hole = (1.0 - W)*m_inc_hole + W*m_hem_hole
    return N_holes * m_hole

# ---- plot ----
fig, ax = plt.subplots(figsize=(7,4))
for T1 in T_list_K:
    y = [mdot_NHNE_total(Pbar*Pa_per_bar, T1) for Pbar in P_range_bar]
    ax.plot(P_range_bar, y, marker="o", label=f"{T1:.0f} K")

ax.set_xlabel("Tank Pressure [bar]")
ax.set_ylabel(r"Total $\dot m_{\mathrm{NHNE}}$ [kg/s]")
ax.grid(True, which="both", ls=":")
ax.legend(title="Tank temperature", fontsize=8)
plt.title("NHNE mass flow vs tank pressure (one curve per tank temperature)")
plt.tight_layout()
plt.show()
