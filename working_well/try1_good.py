# nhne_tank_sim.py

import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from dataclasses import dataclass
from typing import Tuple
#from scipy.optimize import fsolve
from scipy.optimize import least_squares

# -------------------------
# Constantes globais
# -------------------------

FLUID   = "NitrousOxide"
EPS_REL = 1e-4

# Geometria da placa (exemplo)
N_holes = 65
D_hole  = 1.5e-3  # m
Cd      = 0.67
A_hole  = 0.25*np.pi*D_hole**2

# Pressão a jusante (cold-flow, 1 bar abs)
P2 = 30.0e5  # Pa


# -------------------------
# NHNE helpers (teu bloco + pequenos ajustes)
# -------------------------

def rho_from_PT_safe(P, T, h_hint=None):
    """
    Tenta obter densidade a partir de (P,T).
    Se falhar porque está exactamente em saturação, interpola com base em h_hint.
    """
    try:
        return PropsSI("D", "P", P, "T", T, FLUID), None
    except Exception:
        Psat = PropsSI("P", "T", T, "Q", 0, FLUID)
        if abs(P - Psat) <= max(1.0, EPS_REL*P):
            hL = PropsSI("H", "T", T, "Q", 0, FLUID)
            hV = PropsSI("H", "T", T, "Q", 1, FLUID)
            if h_hint is None:
                h_hint = PropsSI("H", "T", T, "P", P, FLUID)
            if hV == hL:
                x = 0.0
            else:
                x = np.clip((h_hint - hL) / max(hV - hL, 1e-12), 0.0, 1.0)
            rhoL = PropsSI("D", "T", T, "Q", 0, FLUID)
            rhoV = PropsSI("D", "T", T, "Q", 1, FLUID)
            rho_mix = 1.0 / (x / rhoV + (1.0 - x) / rhoL)
            return rho_mix, x
        # ligeiro “desvio” em P e T para fugir à fronteira
        return PropsSI("D", "P", P * (1.0 - EPS_REL),
                       "T", T * (1.0 - EPS_REL), FLUID), None


def mdot_inc_per_hole(A, Cd, P1, T1, P2):
    """
    Branch incompressível: m_inc = Cd A sqrt(2 rho_l (P1-P2)).
    Devolve também h_exit_inc ~ entalpia de líquido à montante.
    """
    Psat = PropsSI("P", "T", T1, "Q", 0, FLUID)
    rho_l = PropsSI("D", "T", T1, "P", max(P1, 1.01*Psat), FLUID)
    dP    = max(P1 - P2, 0.0)
    mdot_inc = Cd * A * np.sqrt(2.0 * rho_l * dP)
    h_exit_inc = PropsSI("H", "T", T1, "Q", 0, FLUID)  # líquido saturado
    return mdot_inc, h_exit_inc


def mdot_HEM_per_hole(A, Cd, P1, T1, P2):
    """
    Branch HEM igual ao que tinhas originalmente:
      - h1, s1 a montante (quase líquido)
      - estado 2 em P2 por S constante
      - G = rho2 * sqrt(2 * (h1 - h2))
      - m_HEM = Cd A G

    Devolve também h_exit_HEM = h2.
    """
    P_sat = PropsSI("P", "T", T1, "Q", 0, FLUID)
    P1_eff = max(P1, 1.01*P_sat)

    h1 = PropsSI("H", "T", T1, "P", P1_eff, FLUID)
    s1 = PropsSI("S", "T", T1, "P", P1_eff, FLUID)

    try:
        T2 = PropsSI("T", "P", P2, "S", s1, FLUID)
        h2 = PropsSI("H", "P", P2, "T", T2, FLUID)
    except Exception:
        # fallback se a isoentrópica falhar
        T2 = PropsSI("T", "P", P2, "Q", 0, FLUID)
        hL = PropsSI("H", "T", T2, "Q", 0, FLUID)
        hV = PropsSI("H", "T", T2, "Q", 1, FLUID)
        h2 = 0.5*(hL + hV)

    rho2, _ = rho_from_PT_safe(P2, T2, h_hint=h2)
    dh = max(h1 - h2, 0.0)
    G = rho2 * np.sqrt(2.0 * dh)  # mass flux HEM
    mdot_hem = Cd * A * G
    h_exit_hem = h2
    return mdot_hem, h_exit_hem


def mdot_NHNE_total(P1, T1):
    """
    NHNE com κ e blend EXACTAMENTE como tinhas:
      κ = sqrt((P1-P2)/(Psat(T1)-P2))
      m_hole = [1/(1+κ)] m_inc + [κ/(1+κ)] m_HEM

    Para o balanço de energia, fazemos o mesmo blend em h_exit.
    Devolve (mdot_total, h_exit_total).
    """
    Psat = PropsSI("P", "T", T1, "Q", 0, FLUID)

    if P1 <= P2 or Psat <= P2:
        # sem ΔP útil -> praticamente sem escoamento
        hL = PropsSI("H", "T", T1, "Q", 0, FLUID)
        return 0.0, hL

    # κ do Solomon/Dyer
    kappa = np.sqrt(max(P1 - P2, 0.0) / max(Psat - P2, 1e-3))

    # branches
    m_inc, h_inc = mdot_inc_per_hole(A_hole, Cd, P1, T1, P2)
    m_hem, h_hem = mdot_HEM_per_hole(A_hole, Cd, P1, T1, P2)

    w_inc = 1.0/(1.0 + kappa)
    w_hem = kappa/(1.0 + kappa)

    m_hole = w_inc*m_inc + w_hem*m_hem
    h_exit = w_inc*h_inc + w_hem*h_hem

    mdot_total = N_holes * m_hole
    return mdot_total, h_exit



# -------------------------
# Modelo de tanque saturado
# -------------------------

@dataclass
class TankState:
    m: float   # massa total [kg]
    U: float   # energia interna total [J]
    V: float   # volume do tanque [m^3]


def tank_state_to_TxP(state: TankState,
                      T_guess: float = 293.0,
                      x_guess: float = 0.1) -> Tuple[float, float, float]:
    """
    Dado (m, U, V) resolve (T, x, P_sat) com bounds:
      185 K <= T <= 305 K
      0 <= x <= 1
    Usa least_squares com penalização se a CoolProp falhar.
    """
    m, U, V = state.m, state.U, state.V

    def residuals(vars_):
        T, x = vars_
        # bounds "hard" em x; T é controlado pelo least_squares
        x = np.clip(x, 0.0, 1.0)
        try:
            uL = PropsSI("U", "T", T, "Q", 0, FLUID)
            uV = PropsSI("U", "T", T, "Q", 1, FLUID)
            rhoL = PropsSI("D", "T", T, "Q", 0, FLUID)
            rhoV = PropsSI("D", "T", T, "Q", 1, FLUID)
        except Exception:
            # se der erro, manda um resíduo enorme para esse ponto
            return [1e8, 1e8]

        u_mix = (1.0 - x)*uL + x*uV
        rho_mix = 1.0 / ((1.0 - x)/rhoL + x/rhoV)

        m_calc = rho_mix * V
        U_calc = m_calc * u_mix
        return [m_calc - m, U_calc - U]

    # bounds em T e x
    lower = np.array([185.0, 0.0])
    upper = np.array([305.0, 1.0])

    x0 = np.array([T_guess, x_guess])

    res = least_squares(residuals, x0, bounds=(lower, upper),
                        xtol=1e-10, ftol=1e-10, gtol=1e-10, max_nfev=200)

    T_sol, x_sol = res.x
    P_sat = PropsSI("P", "T", T_sol, "Q", 0, FLUID)
    return T_sol, x_sol, P_sat



def step_tank(state: TankState, mdot: float, h_exit: float, dt: float) -> TankState:
    """
    Integra massa e energia do tanque num passo forward-Euler.
    """
    m_new = state.m - mdot*dt
    m_new = max(m_new, 1e-6)  # evitar massa negativa
    U_new = state.U - mdot*dt*h_exit
    return TankState(m=m_new, U=U_new, V=state.V)


def init_tank_from_fill(T0: float, V_tank: float, fill_l_vol: float) -> TankState:
    """
    Estado inicial do tanque:
      T0          - temperatura inicial [K]
      V_tank      - volume geométrico [m^3]
      fill_l_vol  - fração volumétrica ocupada por líquido (0..1)
    """
    rhoL = PropsSI("D", "T", T0, "Q", 0, FLUID)
    rhoV = PropsSI("D", "T", T0, "Q", 1, FLUID)
    uL   = PropsSI("U", "T", T0, "Q", 0, FLUID)
    uV   = PropsSI("U", "T", T0, "Q", 1, FLUID)

    V_L = fill_l_vol * V_tank
    V_V = (1.0 - fill_l_vol) * V_tank

    mL = rhoL * V_L
    mV = rhoV * V_V
    m0 = mL + mV
    x0 = mV / m0 if m0 > 0 else 0.0

    u_mix = (1.0 - x0)*uL + x0*uV
    U0 = m0 * u_mix

    return TankState(m=m0, U=U0, V=V_tank)



def simulate_coldflow(t_final: float,
                      dt: float,
                      tank_init: TankState,
                      m_min: float = 1e-5) -> Tuple[np.ndarray, ...]:
    """
    Simulação cold-flow adiabática com tanque autopressurizado e
    injecção NHNE, até t_final ou até m <= m_min.
    """
    n_steps = int(t_final / dt)
    t_hist = np.zeros(n_steps+1)
    P_hist = np.zeros(n_steps+1)
    T_hist = np.zeros(n_steps+1)
    mdot_hist = np.zeros(n_steps+1)
    m_hist = np.zeros(n_steps+1)

    state = tank_init

    # chute inicial para o solver de tanque
    T_guess, x_guess = 293.0, 0.1
    T, x, P = tank_state_to_TxP(state, T_guess, x_guess)
    T_guess, x_guess = T, x

    t_hist[0] = 0.0
    P_hist[0] = P
    T_hist[0] = T
    m_hist[0] = state.m
    mdot_hist[0] = 0.0

    for i in range(1, n_steps+1):
        # 1) estado do tanque
        T, x, P = tank_state_to_TxP(state, T_guess, x_guess)
        T_guess, x_guess = T, x

        # 2) massflow NHNE + entalpia de saída DO PRÓPRIO NHNE
        mdot, h_exit = mdot_NHNE_total(P, T)

        # 3) integrar tanque

        # 4) guardar histórico ...


        # 4) integrar tanque
        state = step_tank(state, mdot, h_exit, dt)

        # 5) guardar histórico
        t = i*dt
        t_hist[i] = t
        P_hist[i] = P
        T_hist[i] = T
        mdot_hist[i] = mdot
        m_hist[i] = state.m

        # condição de paragem "tanque vazio"
        if state.m <= m_min:
            t_hist = t_hist[:i+1]
            P_hist = P_hist[:i+1]
            T_hist = T_hist[:i+1]
            mdot_hist = mdot_hist[:i+1]
            m_hist = m_hist[:i+1]
            break

    return t_hist, P_hist, T_hist, mdot_hist, m_hist



# -------------------------
# Teste simples (inputzito)
# -------------------------
if __name__ == "__main__":
    # Exemplo de tanque:
    #  - Volume 10 L
    #  - T0 = 293 K
    #  - 70% volume líquido
    V_tank = 9e-3   # m^3
    T0 = 7 + 273.15       # K
    fill_l_vol = 0.9

    # Estado inicial a partir de volume e fração de enchimento
    tank0 = init_tank_from_fill(T0, V_tank, fill_l_vol)

    # Resolver T, x, P do tanque inicial (usar T0, 0.1 como chute)
    T_init, x_init, P_init = tank_state_to_TxP(tank0, T_guess=T0, x_guess=0.1)

    print("Estado inicial do tanque:")
    print(f"T0 = {T_init:.2f} K")
    print(f"P0 = {P_init/1e5:.2f} bar")
    print(f"m0 = {tank0.m:.3f} kg")
    print(f"x0 = {x_init:.3f} (qualidade)")

    # Teste direto do mdot_NHNE_total para o estado inicial
    # Teste direto do mdot_NHNE_total para o estado inicial
    mdot0, h_exit0 = mdot_NHNE_total(P_init, T_init)
    print(f"\nmdot_NHNE_total inicial = {mdot0:.4f} kg/s")
    print(f"h_exit inicial = {h_exit0/1e3:.2f} kJ/kg")


    # Simulação até 5 s ou até massa <= m_min
    t_final = 5.0
    dt = 0.001
    m_min = 1e-4  # kg

    t_hist, P_hist, T_hist, mdot_hist, m_hist = simulate_coldflow(
        t_final=t_final,
        dt=dt,
        tank_init=tank0,
        m_min=m_min,
    )

    # Plots básicos
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)

    axs[0].plot(t_hist, P_hist/1e5)
    axs[0].set_ylabel("P_tank [bar]")
    axs[0].grid(True)

    axs[1].plot(t_hist, mdot_hist)
    axs[1].set_ylabel("mdot [kg/s]")
    axs[1].grid(True)

    axs[2].plot(t_hist, m_hist)
    axs[2].set_ylabel("m_tank [kg]")
    axs[2].set_xlabel("t [s]")
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()
