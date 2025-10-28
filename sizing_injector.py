# --- Requirements: pip install CoolProp numpy ---
from CoolProp.CoolProp import PropsSI
import numpy as np
from math import pi, ceil

FLUID = "NitrousOxide"  # CoolProp name

# ===== Inputs (SI) =====
m_dot_total = 1.5          # kg/s  (caudal desejado)
P_tank_bar  = 40.0         # bar   (pressão do tanque)
T_tank_K    = 288.15       # K     (~10 °C)
P_cham_bar  = 30.0         # bar   (alvo de câmara para sizing inicial)
Cd          = 0.67         # coef. descarga
D_hole      = 1.5e-3       # m     (diâmetro escolhido para varrer N)
L_over_D    = 4.0          # razão geométrica

# ===== Conversões =====
Pa_per_bar = 1e5
P1 = P_tank_bar * Pa_per_bar
P2 = P_cham_bar * Pa_per_bar
dP = max(P1 - P2, 1.0)  # evita zero

# ===== Propriedades N2O (fase líquida à montante) =====
# Saturação a T_tank (para flash check e kappa)
P_sat = PropsSI("P","T",T_tank_K,"Q",0,FLUID)

# --- Properties block corrected ---
try:
    rho_l = PropsSI("D","T",T_tank_K,"P",max(P1,P_sat*1.01),FLUID)
except ValueError:
    rho_l = 785.0  # kg/m3

try:
    mu_l  = PropsSI("V","T",T_tank_K,"P",max(P1,P_sat*1.01),FLUID)
except ValueError:
    mu_l  = 0.6e-3  # Pa·s

try:
    sigma_s = PropsSI("I","T",T_tank_K,"P",max(P1,P_sat*1.01),FLUID)
except ValueError:
    sigma_s = 0.018  # N/m

try:
    cp_l = PropsSI("C","T",T_tank_K,"P",max(P1,P_sat*1.01),FLUID)
except ValueError:
    cp_l = 1550.0  # J/kg/K

# Entalpia e saturação ainda disponíveis normalmente
T_sat_P2 = PropsSI("T","P",P2,"Q",0,FLUID)
h_f_P2   = PropsSI("H","P",P2,"Q",0,FLUID)
h_g_P2   = PropsSI("H","P",P2,"Q",1,FLUID)
h_fg_P2  = h_g_P2 - h_f_P2
h_in     = PropsSI("H","T",T_tank_K,"P",max(P1,P_sat*1.01),FLUID)

# Estado saturado na saída a P2

# ===== Números adimensionais úteis para checks =====
# Jakob (para avaliar potencial de flashing)
Ja = cp_l * max(T_tank_K - T_sat_P2, 0.0) / max(h_fg_P2, 1e-6)
# Parâmetro kappa (Solomon/NHNE): sqrt( (P1-P2)/(P_sat(Tin)-P2) )
kappa = np.sqrt(max(P1 - P2,0.0) / max(P_sat - P2,1.0))

# ===== Modelos de vazão por furo =====
def A_from_D(d): 
    return 0.25*pi*d*d

def mdot_incompressible_per_hole(A):
    """Orifício líquido incompressível (limite superior)."""
    return Cd * A * np.sqrt(2.0 * rho_l * dP)

def mdot_HEM_like_per_hole(A):
    """
    Aproximação HEM-like simples para flashing:
    Usa densidade efetiva e head de pressão. Conservador vs NHNE.
    """
    # Qualidade isentálpica na saída (clipped 0..1)
    x = (h_in - h_f_P2) / max(h_fg_P2, 1e-6)
    x = float(np.clip(x, 0.0, 1.0))
    # Densidades saturadas na saída
    rho_l2 = PropsSI("D","P",P2,"Q",0,FLUID)
    rho_g2 = PropsSI("D","P",P2,"Q",1,FLUID)
    # Densidade de mistura HEM
    rho_mix = 1.0 / (x / rho_g2 + (1.0 - x) / rho_l2)
    # Fluxo "tipo HEM": substitui rho_l por rho_mix no orifício
    return Cd * A * np.sqrt(2.0 * rho_mix * dP)

def mdot_NHNE_per_hole(A):
    """
    Mistura NHNE de Solomon: combina inc e HEM por kappa.
    Forma convexa: fração HEM aumenta com superheat.
    """
    m_inc = mdot_incompressible_per_hole(A)
    m_hem = mdot_HEM_like_per_hole(A)
    w_hem = 1.0/(1.0 + kappa)          # peso HEM
    w_inc = 1.0 - w_hem                # peso incompressível
    return w_inc * m_inc + w_hem * m_hem

# ===== Solver simples para A dado m_dot por furo =====
def area_for_target_mdot_per_hole(mdot_h, model="NHNE"):
    """
    Resolve A por busca monotónica, já que mdot(A) ~ A*sqrt(...).
    model in {"INC","HEM","NHNE"}.
    """
    f = {"INC": mdot_incompressible_per_hole,
         "HEM": mdot_HEM_like_per_hole,
         "NHNE": mdot_NHNE_per_hole}[model]
    # limites de busca para A
    A_lo, A_hi = 1e-10, 1e-4  # 0.0001 m2 = D≈11.3 mm
    for _ in range(60):
        A_mid = 0.5*(A_lo + A_hi)
        m_mid = f(A_mid)
        if m_mid < mdot_h:
            A_lo = A_mid
        else:
            A_hi = A_mid
    return 0.5*(A_lo + A_hi)

# ===== Pipeline de dimensionamento =====
# 1) Área por furo necessária para o caudal desejado com NHNE
#    Primeiro decide N ou D. Aqui mostramos as duas vias.

# ---- VIA A: escolher N e calcular D ----
def size_for_given_N(N):
    mdot_h = m_dot_total / N
    A_req  = area_for_target_mdot_per_hole(mdot_h, model="NHNE")
    D_req  = np.sqrt(4.0*A_req/pi)
    # Checks (internos do furo usando líquido)
    v_exit = mdot_h / (rho_l * A_req)
    Re     = rho_l * v_exit * D_req / mu_l
    We     = rho_l * v_exit**2 * D_req / max(sigma_s,1e-6)
    return {
        "N": N, "A_per_hole": A_req, "D_per_hole": D_req,
        "v_exit_liq": v_exit, "Re": Re, "We": We,
        "Ja": Ja, "kappa": kappa, "P_sat_bar": P_sat/Pa_per_bar
    }

# ---- VIA B: escolher D e obter N inteiro mínimo ----
def size_for_given_D(D):
    A = A_from_D(D)
    mdot_h = mdot_NHNE_per_hole(A)
    N = int(ceil(m_dot_total / mdot_h))
    # recompute with integer N to get actual per-hole mdot and checks
    mdot_h_final = m_dot_total / N
    A_req  = area_for_target_mdot_per_hole(mdot_h_final, model="NHNE")
    D_req  = np.sqrt(4.0*A_req/pi)
    v_exit = mdot_h_final / (rho_l * A_req)
    Re     = rho_l * v_exit * D_req / mu_l
    We     = rho_l * v_exit**2 * D_req / max(sigma_s,1e-6)
    return {
        "D_input": D, "N_min": N,
        "A_per_hole_req": A_req, "D_per_hole_req": D_req,
        "v_exit_liq": v_exit, "Re": Re, "We": We,
        "Ja": Ja, "kappa": kappa, "P_sat_bar": P_sat/Pa_per_bar
    }

# ===== Exemplos de uso =====
outA = size_for_given_N(N=6)         # escolhe N e obtém D
outB = size_for_given_D(D=D_hole)    # escolhe D e obtém N mínimo

print("=== VIA A: dar N, obter D ===")
for k,v in outA.items():
    print(f"{k}: {v}")
print("\n=== VIA B: dar D, obter N ===")
for k,v in outB.items():
    print(f"{k}: {v}")
