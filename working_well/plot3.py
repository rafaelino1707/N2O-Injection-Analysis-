# NHNE + sistema: tanque -> linha -> solenoide (Kv) -> placa
# Requer: CoolProp, numpy, matplotlib, scipy

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from math import pi
from scipy.optimize import brentq

FLUID = "NitrousOxide"

# --- Geometria da placa ---
N_holes  = 65
D_hole   = 1.5e-3       # m
Cd_plate = 0.67
A_hole   = 0.25 * pi * D_hole**2

# --- Câmara ---
P_cham_bar = 1
Pa_per_bar = 1e5
P_cham     = P_cham_bar * Pa_per_bar

# --- Linha (feedline N2O) ---
D_line = 0.0127        # m (1/2" ~ 12.7 mm)
L_line = 0.544         # m (equivalente)
f_D    = 0.02          # Darcy
K_misc = 5.0           # somatório de K locais

A_line = 0.25 * pi * D_line**2

# --- Solenoide principal: Kv da tabela ---
KV_sol = 4.1           # m^3/h

# --- Varredura de pressão do tanque ---
P_range_bar = np.linspace(25.0, 50.0, 60)

# --- utilitário para mistura saturada (NHNE) ---
EPS_REL = 1e-6

def rho_from_PT_safe(P, T, h_hint=None):
    try:
        return PropsSI("D", "P", P, "T", T, FLUID), None
    except Exception:
        Psat = PropsSI("P", "T", T, "Q", 0, FLUID)
        if abs(P - Psat) <= max(10.0, EPS_REL * Psat):
            if h_hint is None:
                rhoL = PropsSI("D", "T", T, "Q", 0, FLUID)
                return rhoL, 0.0
            hL = PropsSI("H", "T", T, "Q", 0, FLUID)
            hV = PropsSI("H", "T", T, "Q", 1, FLUID)
            denom = max(hV - hL, 1e-8)
            x = np.clip((h_hint - hL) / denom, 0.0, 1.0)
            rhoL = PropsSI("D", "T", T, "Q", 0, FLUID)
            rhoV = PropsSI("D", "T", T, "Q", 1, FLUID)
            rho_mix = 1.0 / (x / rhoV + (1.0 - x) / rhoL)
            return rho_mix, x
        return PropsSI("D",
                       "P", P * (1.0 - EPS_REL),
                       "T", T * (1.0 - EPS_REL),
                       FLUID), None

# ---------- NHNE da placa ----------

def mdot_inc_per_hole(A, Cd, rho_l, dP):
    return Cd * A * np.sqrt(2.0 * rho_l * max(dP, 0.0))

def mdot_HEM_per_hole(A, Cd, P1, T1, P2):
    P_sat = PropsSI("P", "T", T1, "Q", 0, FLUID)
    h1 = PropsSI("H", "T", T1, "P", max(P1, 1.01 * P_sat), FLUID)
    s1 = PropsSI("S", "T", T1, "P", max(P1, 1.01 * P_sat), FLUID)
    try:
        T2 = PropsSI("T", "P", P2, "S", s1, FLUID)
        h2 = PropsSI("H", "P", P2, "T", T2, FLUID)
    except Exception:
        T2 = PropsSI("T", "P", P2, "Q", 0, FLUID)
        hL = PropsSI("H", "T", T2, "Q", 0, FLUID)
        hV = PropsSI("H", "T", T2, "Q", 1, FLUID)
        h2 = 0.5 * (hL + hV)
    rho2, _ = rho_from_PT_safe(P2, T2, h_hint=h2)
    dh = max(h1 - h2, 0.0)
    return Cd * A * rho2 * np.sqrt(2.0 * dh)

def mdot_NHNE_plate(P1, T1):
    P_sat = PropsSI("P", "T", T1, "Q", 0, FLUID)
    rho_l = PropsSI("D", "T", T1, "P", max(P1, 1.01 * P_sat), FLUID)
    dP    = max(P1 - P_cham, 0.0)
    m_inc = mdot_inc_per_hole(A_hole, Cd_plate, rho_l, dP)
    m_hem = mdot_HEM_per_hole(A_hole, Cd_plate, P1, T1, P_cham)
    kappa = np.sqrt(max(P1 - P_cham, 0.0) / max(P_sat - P_cham, 1.0))
    m_hole = (1.0 / (1.0 + kappa)) * m_inc + (kappa / (1.0 + kappa)) * m_hem
    return N_holes * m_hole

# ---------- Solenoide via Kv ----------

def mdot_solenoid_Kv(P_up, P_down, rho):
    """
    mdot [kg/s] através da solenoide para dado ΔP, usando Kv.
    P_up, P_down em Pa, rho em kg/m3.
    """
    dP = max(P_up - P_down, 0.0)
    if dP <= 0.0 or KV_sol <= 0.0:
        return 0.0, False
    dP_bar = dP / 1e5                     # bar
    Q_m3ph = KV_sol * np.sqrt(dP_bar * (rho / 1000.0))
    Q_m3ps = Q_m3ph / 3600.0
    m_dot  = rho * Q_m3ps
    return m_dot, False   # não estamos a modelar choke explícito aqui

def deltaP_solenoid_from_mdot_Kv(mdot, rho):
    """
    ΔP [Pa] necessário na solenoide para um mdot dado, usando Kv.
    """
    if mdot <= 0.0 or KV_sol <= 0.0:
        return 0.0
    # Q [m3/s]
    Q_m3ps = mdot / rho
    Q_m3ph = Q_m3ps * 3600.0
    # ΔP_bar = (Qh/Kv)^2 * 1000/rho
    dP_bar = (Q_m3ph / KV_sol)**2 * (1000.0 / rho)
    dP_Pa  = dP_bar * 1e5
    return dP_Pa

# ---------- Perdas na linha ----------

def deltaP_line(mdot, rho):
    if mdot <= 0.0:
        return 0.0
    v = mdot / (rho * A_line)
    return (f_D * L_line / D_line + K_misc) * 0.5 * rho * v**2

# ---------- Solver do sistema ----------

def mdot_system(P_tank):
    # tanque auto-pressurizado
    T_tank = PropsSI("T", "P", P_tank, "Q", 0, FLUID)
    rho_liq = PropsSI("D", "T", T_tank, "Q", 0, FLUID)

    # upper bound: placa sozinha com P1 = P_tank
    mdot_max = mdot_NHNE_plate(P_tank, T_tank)
    if mdot_max <= 0:
        return 0.0, T_tank, False, False

    def residual(mdot):
        # perdas na linha
        dP_l = deltaP_line(mdot, rho_liq)
        P_after_line = P_tank - dP_l
        if P_after_line <= P_cham:
            return mdot

        # capacidade da solenoide
        m_sol, _ = mdot_solenoid_Kv(P_after_line, P_cham, rho_liq)
        if mdot > m_sol:
            return mdot - m_sol

        # ΔP exata na solenoide para este mdot
        dP_sol = deltaP_solenoid_from_mdot_Kv(mdot, rho_liq)
        P_plate_up = P_after_line - dP_sol
        P_plate_up = max(P_plate_up, P_cham + 100.0)

        # caudal que a placa aceitaria com esta pressão
        mdot_plate = mdot_NHNE_plate(P_plate_up, T_tank)
        return mdot - mdot_plate

    try:
        mdot_sol = brentq(residual, 1e-4, mdot_max)
    except ValueError:
        mdot_sol = 0.0

    # flags simples
    T_tank = PropsSI("T", "P", P_tank, "Q", 0, FLUID)
    rho_liq = PropsSI("D", "T", T_tank, "Q", 0, FLUID)
    dP_l = deltaP_line(mdot_sol, rho_liq)
    P_after_line = P_tank - dP_l
    m_sol, choked_sol = mdot_solenoid_Kv(P_after_line, P_cham, rho_liq)

    P_sat_tank = PropsSI("P", "T", T_tank, "Q", 0, FLUID)
    choked_plate = P_after_line > P_sat_tank and P_cham < P_sat_tank

    return mdot_sol, T_tank, choked_sol, choked_plate

# ---------- Cálculo e plots ----------

mdot_list      = []
T_list         = []
ch_sol_list    = []
ch_plate_list  = []

for Pbar in P_range_bar:
    P_tank = Pbar * Pa_per_bar
    md, Tt, cs, cp = mdot_system(P_tank)
    mdot_list.append(md)
    T_list.append(Tt - 273.15)
    ch_sol_list.append(cs)
    ch_plate_list.append(cp)

mdot_arr = np.array(mdot_list)
T_arr    = np.array(T_list)

fig, axs = plt.subplots(2, 1, figsize=(7, 8))

axs[0].plot(P_range_bar, mdot_arr)
axs[0].set_xlabel("Tank Pressure [bar] (self-pressurized)")
axs[0].set_ylabel(r"System $\dot m$ [kg/s]")
axs[0].set_title("Cold-flow system: tank -> line -> solenoid(Kv) -> plate")
axs[0].grid(True, ls=":")

axs[1].plot(T_arr, mdot_arr)
axs[1].set_xlabel("Tank saturation temperature [°C]")
axs[1].set_ylabel(r"System $\dot m$ [kg/s]")
axs[1].set_title("Same case, as $\dot m$ vs $T_{sat}(P)$")
axs[1].grid(True, ls=":")

plt.tight_layout()
plt.show()

# --- Debug num ponto (por ex. 40 bar) ---

P_tank = 40.0 * Pa_per_bar
T_tank = PropsSI("T", "P", P_tank, "Q", 0, FLUID)
rho_liq = PropsSI("D", "T", T_tank, "Q", 0, FLUID)

md_guess = 0.5   # valor de teste
dP_l = deltaP_line(md_guess, rho_liq)
print("ΔP linha [bar] =", dP_l / 1e5)
P_after_line = P_tank - dP_l

m_sol, _ = mdot_solenoid_Kv(P_after_line, P_cham, rho_liq)
print("Capacidade solenoide [kg/s] =", m_sol)

md_plate = mdot_NHNE_plate(P_after_line - deltaP_solenoid_from_mdot_Kv(md_guess, rho_liq),
                            T_tank)
print("Capacidade placa [kg/s] =", md_plate)
