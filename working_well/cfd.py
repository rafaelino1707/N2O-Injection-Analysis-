# -*- coding: utf-8 -*-
"""
Modelo 0D/2D de placa de injeção multi-furo sem CFD

- NHNE/HEM para o escoamento através de cada furo
- Distribuição geométrica por anéis
- Campo de mass flux na base da câmara via superposição Gaussiana
- Métrica de uniformidade (coeficiente de variação)

Requer:
    pip install CoolProp numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from math import pi

# =========================
# Parâmetros globais
# =========================

FLUID = "NitrousOxide"

Pa_per_bar = 1e5

# Câmara
P_cham_bar = 1.0
P_cham = P_cham_bar * Pa_per_bar

# --- Funções termodinâmicas robustas (adaptadas do teu código) ---

EPS_REL = 1e-6

def rho_from_PT_safe(P, T, h_hint=None):
    """
    Densidade a partir de P,T tentando contornar problemas na saturação.
    Devolve (rho, x) onde x é a qualidade (ou None se não em saturação tratada).
    """
    try:
        return PropsSI("D", "P", P, "T", T, FLUID), None
    except Exception:
        Psat = PropsSI("P", "T", T, "Q", 0, FLUID)
        if abs(P - Psat) <= max(1.0, EPS_REL * P):
            hL = PropsSI("H", "T", T, "Q", 0, FLUID)
            hV = PropsSI("H", "T", T, "Q", 1, FLUID)
            if h_hint is None:
                h_hint = PropsSI("H", "P", P, "T", T, FLUID)
            x = 0.0 if hV == hL else np.clip((h_hint - hL) / max(hV - hL, 1e-12), 0.0, 1.0)
            rhoL = PropsSI("D", "T", T, "Q", 0, FLUID)
            rhoV = PropsSI("D", "T", T, "Q", 1, FLUID)
            rho_mix = 1.0 / (x / rhoV + (1.0 - x) / rhoL)
            return rho_mix, x
        # fallback ligeiramente deslocado de P,T
        return PropsSI("D", "P", P * (1.0 - EPS_REL), "T", T * (1.0 - EPS_REL), FLUID), None


def mdot_inc_per_hole(A, Cd, rho_l, dP):
    """
    Escoamento incompressível por orifício (liquido).
    """
    return Cd * A * np.sqrt(2.0 * rho_l * max(dP, 0.0))


def mdot_HEM_per_hole(A, Cd, P1, T1, P2):
    """
    Homogeneous Equilibrium Model (HEM) aproximado para flashing.
    """
    P_sat = PropsSI("P", "T", T1, "Q", 0, FLUID)
    # Estado 1: compressão leve acima de sat para evitar problemas numéricos
    P1_eff = max(P1, 1.01 * P_sat)
    h1 = PropsSI("H", "T", T1, "P", P1_eff, FLUID)
    s1 = PropsSI("S", "T", T1, "P", P1_eff, FLUID)

    try:
        T2 = PropsSI("T", "P", P2, "S", s1, FLUID)
        h2 = PropsSI("H", "P", P2, "T", T2, FLUID)
    except Exception:
        # fallback: T2 saturado, entalpia média líquido+vapor
        T2 = PropsSI("T", "P", P2, "Q", 0, FLUID)
        hL = PropsSI("H", "T", T2, "Q", 0, FLUID)
        hV = PropsSI("H", "T", T2, "Q", 1, FLUID)
        h2 = 0.5 * (hL + hV)

    rho2, _ = rho_from_PT_safe(P2, T2, h_hint=h2)
    dh = max(h1 - h2, 0.0)
    return Cd * A * rho2 * np.sqrt(2.0 * dh)


def mdot_NHNE_per_hole(P1, T1, P2, D_hole, Cd):
    """
    Modelo NHNE simples por furo:
      - combina orifício líquido incompressível e HEM com interpolação
    """
    A_hole = 0.25 * pi * D_hole**2

    P_sat = PropsSI("P", "T", T1, "Q", 0, FLUID)
    rho_l = PropsSI("D", "T", T1, "P", max(P1, 1.01 * P_sat), FLUID)
    dP = max(P1 - P2, 0.0)

    m_inc = mdot_inc_per_hole(A_hole, Cd, rho_l, dP)
    m_hem = mdot_HEM_per_hole(A_hole, Cd, P1, T1, P2)

    # fator de interpolação (podes afinar se quiseres)
    kappa = np.sqrt(max(P1 - P2, 0.0) / max(P_sat - P2, 1.0))
    m_hole = (1.0 / (1.0 + kappa)) * m_inc + (kappa / (1.0 + kappa)) * m_hem
    return m_hole


# =========================
# Geometria da placa
# =========================

def generate_ring_layout(rings):
    """
    Gera lista de furos com (x, y, ring_index) a partir de uma lista de anéis.

    rings: lista de dicts, cada um com:
        "r": raio do anel [m]
        "n": número de furos no anel
    """
    holes = []
    for k, ring in enumerate(rings):
        r = ring["r"]
        n = ring["n"]
        for j in range(n):
            theta = 2.0 * pi * j / n
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            holes.append({"x": x, "y": y, "ring": k})
    return holes


# =========================
# Modelo de distribuição de caudal por furo
# =========================

def compute_mdot_per_hole(tank_P_bar,
                          mode_T="self_pressurized",
                          T_fixed_C=0.0,
                          P_cham=P_cham,
                          D_hole=1.5e-3,
                          Cd_global=0.67,
                          rings=None,
                          ring_Cd_factors=None):
    """
    Calcula mdot por furo para uma dada placa, assumindo:
      - um único nó de plenum (mesma pressão para todos os furos)
      - NHNE por furo
      - opção de T1 = Tsat(P1) (self_pressurized) ou T1 fixo

    ring_Cd_factors: lista de fatores multiplicativos para Cd por anel
                     (len = nº de anéis) ou None para tudo igual.
    """
    if rings is None:
        raise ValueError("É necessário fornecer a lista de anéis 'rings'.")

    P_tank = tank_P_bar * Pa_per_bar

    # temperatura de entrada
    if mode_T == "self_pressurized":
        T1 = PropsSI("T", "P", P_tank, "Q", 0, FLUID)
    elif mode_T == "fixed_T":
        T1 = T_fixed_C + 273.15
    else:
        raise ValueError("mode_T deve ser 'self_pressurized' ou 'fixed_T'.")

    # Neste modelo simples: P_plenum = P_tank
    P_plenum = P_tank

    holes = generate_ring_layout(rings)
    mdot_list = []

    n_rings = len(rings)
    if ring_Cd_factors is None:
        ring_Cd_factors = [1.0] * n_rings

    for h in holes:
        ring_idx = h["ring"]
        Cd_eff = Cd_global * ring_Cd_factors[ring_idx]
        m_hole = mdot_NHNE_per_hole(P_plenum, T1, P_cham, D_hole, Cd_eff)
        mdot_list.append(m_hole)

    mdot_arr = np.array(mdot_list)
    return holes, mdot_arr, T1


# =========================
# Campo de mass flux na base da câmara
# =========================

def compute_flux_map(holes,
                     mdot_arr,
                     D_hole,
                     sigma_factor=1.5,
                     n_grid=151,
                     R_domain=None):
    """
    Cria um mapa 2D de mass flux na base da câmara, por superposição Gaussiana.

    sigma_factor: sigma = sigma_factor * D_hole
    n_grid: resolução da grelha
    R_domain: raio do domínio plotado (se None, usa 1.2 * raio máximo dos furos)
    """
    coords = np.array([[h["x"], h["y"]] for h in holes])
    x_h = coords[:, 0]
    y_h = coords[:, 1]

    if R_domain is None:
        r_max = np.max(np.sqrt(x_h**2 + y_h**2))
        R_domain = 1.2 * r_max

    sigma = sigma_factor * D_hole

    x = np.linspace(-R_domain, R_domain, n_grid)
    y = np.linspace(-R_domain, R_domain, n_grid)
    X, Y = np.meshgrid(x, y)

    flux = np.zeros_like(X)

    two_sigma2 = 2.0 * sigma**2

    for (xi, yi, m) in zip(x_h, y_h, mdot_arr):
        # Distribuição Gaussiana normalizada para integrar a mdot do furo
        G = np.exp(-((X - xi)**2 + (Y - yi)**2) / two_sigma2)
        norm = 2.0 * pi * sigma**2
        flux += m * G / norm

    return X, Y, flux


def flux_uniformity_metrics(flux, mask=None):
    """
    Calcula média, desvio padrão e coeficiente de variação do campo de flux.
    Se for fornecida mask (boolean), usa só a região True.
    """
    if mask is not None:
        vals = flux[mask]
    else:
        vals = flux.flatten()

    mu = np.mean(vals)
    sigma = np.std(vals)
    cv = sigma / mu if mu > 0 else np.nan
    return mu, sigma, cv


# =========================
# Exemplo de utilização
# =========================

if __name__ == "__main__":
    # Definir placa: 3 anéis + eventual furo central
    rings = [
        {"r": 0.0e-3, "n": 1},   # furo central (n=1 em r=0)
        {"r": 5.0e-3, "n": 8},
        {"r": 9.0e-3, "n": 16},
        {"r": 13.0e-3, "n": 24},
    ]

    D_hole = 1.5e-3
    Cd_global = 0.67

    tank_P_bar = 44.0

    # 1) mdot por furo
    holes, mdot_arr, T1 = compute_mdot_per_hole(
        tank_P_bar=tank_P_bar,
        mode_T="fixed_T",
        T_fixed_C=0.0,
        P_cham=P_cham,
        D_hole=D_hole,
        Cd_global=Cd_global,
        rings=rings,
        ring_Cd_factors=None,  # ou algo tipo [0.95, 1.0, 1.02, 1.05]
    )

    print(f"T1 = {T1 - 273.15:.2f} °C")
    print(f"mdot total = {mdot_arr.sum():.4f} kg/s")
    print(f"mdot médio por furo = {mdot_arr.mean():.6f} kg/s")

    # 2) Campo de mass flux
    X, Y, flux = compute_flux_map(
        holes,
        mdot_arr,
        D_hole=D_hole,
        sigma_factor=1.5,
        n_grid=151,
        R_domain=None,
    )

    # Máscara circular para métricas (dentro do raio da câmara, por exemplo)
    R_cam = 20e-3  # 20 mm de raio
    R = np.sqrt(X**2 + Y**2)
    mask = R <= R_cam

    mu, sigma, cv = flux_uniformity_metrics(flux, mask=mask)
    print(f"flux médio = {mu:.2f} kg/(m²·s)")
    print(f"flux desvio padrão = {sigma:.2f} kg/(m²·s)")
    print(f"CV = {cv*100:.2f} %")

    # 3) Plots básicos

    # 3a) mdot por furo
    plt.figure(figsize=(6, 4))
    plt.plot(mdot_arr * 1e3, "o-")
    plt.xlabel("Índice do furo")
    plt.ylabel(r"$\dot m_i$ [g/s]")
    plt.title("Caudal mássico por furo")
    plt.grid(True, ls=":")
    plt.tight_layout()

    # 3b) Mapa de mass flux
    plt.figure(figsize=(5, 5))
    cf = plt.contourf(X * 1e3, Y * 1e3, flux, levels=40,cmap="jet")
    plt.colorbar(cf, label=r"$\dot m''$ [kg/(m²·s)]")
    plt.gca().set_aspect("equal", "box")
    plt.xlabel("x [mm]")
    plt.ylabel("y [mm]")
    plt.title("Campo de mass flux na base da câmara")
    plt.tight_layout()

    plt.show()

T = -0 + 273.15   # converter 4 ºC para Kelvin
Psat = PropsSI("P","T",T,"Q",0, "NitrousOxide")   # Q=0 → líquido saturado

print("Psat =", Psat, "Pa")
print("Psat =", Psat/1e5, "bar")