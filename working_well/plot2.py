# NHNE: m_dot vs P_tank
# pip install CoolProp numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from math import pi

FLUID = "NitrousOxide"

# --- Geometria da placa ---
N_holes = 12
D_hole  = 1.5e-3
Cd      = 0.67
A_hole  = 0.25*pi*D_hole**2

# --- Câmara ---
P_cham_bar = 1
Pa_per_bar = 1e5
P2 = P_cham_bar * Pa_per_bar

# --- Varredura de pressão do tanque ---
P_range_bar = np.linspace(5.0, 50.0, 200)

# --- Utilidades robustas na saturação ---
EPS_REL = 1e-6

def rho_from_PT_safe(P, T, h_hint=None):
    try:
        return PropsSI("D","P",P,"T",T,FLUID), None
    except Exception:
        Psat = PropsSI("P","T",T,"Q",0,FLUID)
        if abs(P-Psat) <= max(1.0, EPS_REL*P):
            hL = PropsSI("H","T",T,"Q",0,FLUID)
            hV = PropsSI("H","T",T,"Q",1,FLUID)
            if h_hint is None:
                h_hint = PropsSI("H","P",P,"T",T,FLUID)
            x = 0.0 if hV==hL else np.clip((h_hint - hL)/max(hV-hL,1e-12),0.0,1.0)
            rhoL = PropsSI("D","T",T,"Q",0,FLUID)
            rhoV = PropsSI("D","T",T,"Q",1,FLUID)
            rho_mix = 1.0/(x/rhoV + (1.0-x)/rhoL)
            return rho_mix, x
        return PropsSI("D","P",P*(1.0-EPS_REL),"T",T*(1.0-EPS_REL),FLUID), None

def mdot_inc_per_hole(A, Cd, rho_l, dP):
    return Cd * A * np.sqrt(2.0 * rho_l * max(dP,0.0))

def mdot_HEM_per_hole(A, Cd, P1, T1, P2):
    P_sat = PropsSI("P","T",T1,"Q",0,FLUID)
    h1 = PropsSI("H","T",T1,"P",max(P1,1.01*P_sat),FLUID)
    s1 = PropsSI("S","T",T1,"P",max(P1,1.01*P_sat),FLUID)
    try:
        T2 = PropsSI("T","P",P2,"S",s1,FLUID)
        h2 = PropsSI("H","P",P2,"T",T2,FLUID)
    except Exception:
        T2 = PropsSI("T","P",P2,"Q",0,FLUID)
        hL = PropsSI("H","T",T2,"Q",0,FLUID)
        hV = PropsSI("H","T",T2,"Q",1,FLUID)
        h2 = 0.5*(hL+hV)
    rho2, _ = rho_from_PT_safe(P2, T2, h_hint=h2)
    dh = max(h1 - h2, 0.0)
    return Cd * A * rho2 * np.sqrt(2.0 * dh)

def mdot_NHNE_total(P1, T1):
    P_sat = PropsSI("P","T",T1,"Q",0,FLUID)
    rho_l = PropsSI("D","T",T1,"P",max(P1,1.01*P_sat),FLUID)
    dP    = max(P1 - P2, 0.0)
    m_inc = mdot_inc_per_hole(A_hole, Cd, rho_l, dP)
    m_hem = mdot_HEM_per_hole(A_hole, Cd, P1, T1, P2)
    kappa = np.sqrt(max(P1-P2,0.0)/max(P_sat-P2,1.0))
    m_hole = (1.0/(1.0+kappa))*m_inc + (kappa/(1.0+kappa))*m_hem
    return N_holes * m_hole

# ========== PLOTS ==========
mode = "fixed_T"   # "self_pressurized" ou "fixed_T"

if mode == "self_pressurized":
    # Tanque auto-pressurizado: T1 = Tsat(P1)
    mdot_list = []
    T_sat_list_K = []

    for Pbar in P_range_bar:
        P1 = Pbar*Pa_per_bar
        T1 = PropsSI("T","P",P1,"Q",0,FLUID)  # T_saturação a P1
        T_sat_list_K.append(T1)
        mdot_list.append(mdot_NHNE_total(P1, T1))

    mdot_arr = np.array(mdot_list)
    T_sat_C  = np.array(T_sat_list_K) - 273.15

    fig, axs = plt.subplots(2, 1, figsize=(7, 8))

    # topo: m_dot vs P_tank
    axs[0].plot(P_range_bar, mdot_arr)
    axs[0].set_xlabel("Tank Pressure [bar] (on saturation)")
    axs[0].set_ylabel(r"Total $\dot m_{\mathrm{NHNE}}$ [kg/s]")
    axs[0].set_title(
        f"Self-pressurized blowdown: T1 = Tsat(P1)\n"
        f"N={N_holes} | D={D_hole*1e3:.1f} mm | Cd={Cd}"
    )
    axs[0].grid(True, ls=":")

    # baixo: m_dot vs T_sat
    axs[1].plot(T_sat_C, mdot_arr)
    axs[1].set_xlabel("Tank saturation temperature [°C]")
    axs[1].set_ylabel(r"Total $\dot m_{\mathrm{NHNE}}$ [kg/s]")
    axs[1].set_title("Same case, reparametrized as $\dot m$ vs $T_{sat}(P)$")
    axs[1].grid(True, ls=":")

    plt.tight_layout()
    plt.show()

elif mode == "fixed_T":
    # Curvas a T fixa (útil para sobre-pressurizado)
    T_list_C = [-20, -10, -5, 0, 5, 10, 15, 20, 25, 30]
    plt.figure(figsize=(7,4))
    for T_C in T_list_C:
        T_K = T_C + 273.15
        y = []
        for Pbar in P_range_bar:
            P1 = Pbar*Pa_per_bar
            y.append(mdot_NHNE_total(P1, T_K))
        plt.plot(P_range_bar, y, label=f"{T_C:.0f}°C")
    plt.xlabel("Tank Pressure [bar]")
    plt.ylabel(r"Total $\dot m_{\mathrm{NHNE}}$ [kg/s]")
    plt.title(f"Over-pressurized study: fixed T1 curves\nN={N_holes} | D={D_hole*1e3:.1f} mm | Cd={Cd}")
    plt.grid(True, ls=":")
    plt.legend(title="Tank temperature")
    plt.tight_layout()
    plt.show()
