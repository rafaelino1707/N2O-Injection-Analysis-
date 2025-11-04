# Heatmap: NHNE "simples" (Solomon blend) — cor = m_dot_total
# Requisitos: pip install CoolProp numpy matplotlib

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from math import pi

FLUID = "NitrousOxide"

# ---- Geometria da placa ----
N_holes = 65
D_hole  = 1.5e-3           # m
Cd      = 0.67             # apenas geométrico neste modelo
A_hole  = 0.25*pi*D_hole**2
A_tot   = N_holes * A_hole

# ---- Câmara (jusante) ----
P_cham_bar = 30.0
Pa_per_bar = 1e5
P2 = P_cham_bar * Pa_per_bar

# ---- Grelha do heatmap ----
P_range_bar = np.linspace(25, 50, 50)   # eixo X [bar]
T_range_K   = np.linspace(270, 295, 40) # eixo Y [K]

# ---- Utilidades "safe" para saturação ----
EPS_REL = 1e-6

def is_near_saturation(P, T, fluid):
    Psat = PropsSI("P","T",T,"Q",0,fluid)
    return abs(P - Psat) <= max(1.0, EPS_REL*P)

def rho_sat_mixture_from_h(P, T, h, fluid):
    rhoL = PropsSI("D","T",T,"Q",0,fluid)
    rhoV = PropsSI("D","T",T,"Q",1,fluid)
    hL   = PropsSI("H","T",T,"Q",0,fluid)
    hV   = PropsSI("H","T",T,"Q",1,fluid)
    x    = 0.0 if hV==hL else np.clip((h - hL)/max(hV - hL, 1e-12), 0.0, 1.0)
    rho_mix = 1.0 / ((1-x)/rhoL + x/rhoV)
    return rho_mix, x

def rho_from_PT_safe(P, T, fluid, h_hint=None):
    # 1) tenta direto
    try:
        return PropsSI("D","P",P,"T",T,fluid), None
    except Exception:
        pass
    # 2) se estiver (quase) na saturação, usa mistura saturada com pista de h
    if is_near_saturation(P, T, fluid):
        if h_hint is None:
            h_hint = PropsSI("H","P",P,"T",T,fluid)
        rho_mix, x = rho_sat_mixture_from_h(P, T, h_hint, fluid)
        return rho_mix, x
    # 3) último recurso: micro-nudge
    return PropsSI("D","P",P*(1.0 - EPS_REL),"T",T*(1.0 - EPS_REL),fluid), None

# ---- Modelo NHNE "simples" ----
def mdot_incompressible_per_hole(A, Cd, rho_l, dP):
    # ramo "SPI" simples (hidráulico): orifício líquido
    return Cd * A * np.sqrt(2.0 * rho_l * dP)

def mdot_HEM_true_per_hole(A, Cd, P1, T1, P2, P_sat):
    # entalpia a montante (líquido comprimido)
    h1 = PropsSI("H","T",T1,"P",max(P1, 1.01*P_sat), FLUID)
    s1 = PropsSI("S","T",T1,"P",max(P1, 1.01*P_sat), FLUID)
    # tenta mapear para P2 isentropicamente; se falhar, usa T_sat(P2)
    try:
        T2 = PropsSI("T","P",P2,"S",s1,FLUID)
        h2 = PropsSI("H","P",P2,"T",T2,FLUID)
    except Exception:
        T2 = PropsSI("T","P",P2,"Q",0,FLUID)  # T de saturação a P2
        hL = PropsSI("H","T",T2,"Q",0,FLUID)
        hV = PropsSI("H","T",T2,"Q",1,FLUID)
        h2 = 0.5*(hL + hV)
    # densidade de saída robusta (aceita estar na sat.)
    rho2, _ = rho_from_PT_safe(P2, T2, FLUID, h_hint=h2)
    dh = max(h1 - h2, 0.0)
    return Cd * A * rho2 * np.sqrt(2.0 * dh)

def mdot_total_NHNE(P1, T1):
    P_sat = PropsSI("P","T",T1,"Q",0,FLUID)
    rho_l = PropsSI("D","T",T1,"P",max(P1, 1.01*P_sat), FLUID)
    dP    = max(P1 - P2, 0.0)

    m_inc_hole = mdot_incompressible_per_hole(A_hole, Cd, rho_l, dP)
    m_hem_hole = mdot_HEM_true_per_hole(A_hole, Cd, P1, T1, P2, P_sat)

    # Peso NHNE (Dyer/Solomon)
    denom = max(P_sat - P2, 1.0)  # evita zero/negativo
    kappa = np.sqrt(max(P1 - P2, 0.0) / denom)
    W = 1.0/(1.0 + kappa)         # peso do HEM
    m_hole = (1.0 - W)*m_inc_hole + W*m_hem_hole
    return N_holes * m_hole

# ---- Avalia grelha (P,T) ----
P_grid_bar, T_grid = np.meshgrid(P_range_bar, T_range_K)
m_grid = np.zeros_like(P_grid_bar, dtype=float)

for i in range(T_grid.shape[0]):
    for j in range(T_grid.shape[1]):
        P1 = P_grid_bar[i,j] * Pa_per_bar
        T1 = T_grid[i,j]
        m_grid[i,j] = mdot_total_NHNE(P1, T1)

# ---- Plot heatmap ----
fig, ax = plt.subplots(figsize=(7.2,5.2))
levels = 30
im = ax.contourf(P_grid_bar, T_grid, m_grid, levels=levels, cmap="RdBu")
cb = plt.colorbar(im, ax=ax)
cb.set_label(r"Total $\dot m_{\mathrm{NHNE}}$ [kg/s]")

ax.set_xlabel("Tank Pressure [bar]")
ax.set_ylabel("Tank Temperature [K]")
ax.set_title(f"NHNE Mass-Flow Map (N₂O)\n N={N_holes} Holes | D={D_hole*1e3} mm")
ax.grid(False)
plt.tight_layout()
plt.show()
