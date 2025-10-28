# NHNE (Solomon) + blowdown com CoolProp (HEOS) e integração de Euler
# Requisitos: pip install numpy scipy CoolProp

import numpy as np
from math import pi, sqrt
from dataclasses import dataclass
from scipy.optimize import root
from CoolProp.CoolProp import PropsSI

FLUID = "NitrousOxide"   # muda para "CarbonDioxide" se precisares

# --------------------- limites e estados (sem assumir saturação) ---------------------

def _T_bounds():
    Tmin = float(PropsSI("Tmin", FLUID)) + 1.0
    Tcrit = float(PropsSI("Tcrit", FLUID)) - 1.0
    return Tmin, Tcrit

def _clip_T(T):
    T_lo, T_hi = _T_bounds()
    return float(max(min(T, T_hi), T_lo))

def state_DT(T, rho):
    """Estado geral para qualquer fase: devolve p, h, s."""
    T = _clip_T(T)
    rho = max(float(rho), 1e-9)
    p = PropsSI("P", "D", rho, "T", T, FLUID)
    h = PropsSI("H", "D", rho, "T", T, FLUID)
    s = PropsSI("S", "D", rho, "T", T, FLUID)
    return p, h, s

def solve_T_from_h_rho(h_target, rho):
    """Resolve T tal que H(D=rho, T)=h_target; bissecção com bracketing robusto."""
    T_lo, T_hi = _T_bounds()
    rho = max(float(rho), 1e-9)

    def f(T):
        T = _clip_T(T)
        return PropsSI("H", "D", rho, "T", T, FLUID) - h_target

    Ts = np.linspace(T_lo, T_hi, 60)
    fs = [f(T) for T in Ts]

    # se algum ponto bate, devolve já
    idx0 = np.argmin(np.abs(fs))
    if abs(fs[idx0]) < 1e-6:
        return float(Ts[idx0])

    # procura mudança de sinal
    br = None
    for i in range(len(Ts)-1):
        if fs[i] * fs[i+1] < 0:
            br = (Ts[i], Ts[i+1])
            break

    # se não há bracketing, escolhe o T mais próximo em valor de h
    if br is None:
        return float(Ts[idx0])

    a, b = br
    fa, fb = f(a), f(b)
    for _ in range(80):
        m = 0.5*(a+b)
        fm = f(m)
        if fm == 0.0 or abs(b-a) < 1e-6:
            return float(m)
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return float(0.5*(a+b))

def downstream_state(P2, s1=None, h1=None):
    """Resolve (T2, rho2) com P(D,T)=P2 e S(D,T)=s1  OU  H(D,T)=h1."""
    assert (s1 is None) ^ (h1 is None), "Escolhe isentrópico (s1) OU adiabático (h1)."
    T_lo, T_hi = _T_bounds()
    x0 = np.array([0.5*(T_lo+T_hi), 5.0], dtype=float)  # chute T[K], rho[kg/m3]

    def F(v):
        T = _clip_T(v[0])
        rho = max(float(v[1]), 1e-8)
        p = PropsSI("P", "D", rho, "T", T, FLUID)
        s = PropsSI("S", "D", rho, "T", T, FLUID)
        h = PropsSI("H", "D", rho, "T", T, FLUID)
        g1 = p - P2
        g2 = (s - s1) if s1 is not None else (h - h1)
        return np.array([g1, g2], dtype=float)

    sol = root(F, x0)
    T2 = _clip_T(sol.x[0])
    rho2 = max(float(sol.x[1]), 1e-8)
    p2, h2, s2 = state_DT(T2, rho2)
    return T2, rho2, p2, h2, s2

# --------------------- modelos de caudal (inc, HEM, NHNE) ---------------------

def m_dot_incompressible(Cd, Ac, rho1, P1, P2):
    dP = max(P1 - P2, 0.0)
    return Cd * Ac * sqrt(2.0 * rho1 * dP)

def m_dot_HEM(Cd, Ac, rho2, h1, h2):
    dh = max(h1 - h2, 0.0)
    return Cd * Ac * rho2 * sqrt(2.0 * dh)

def _Pvap_from_T(T):
    """Pressão de saturação a T; corta T ao domínio de saturação do fluido."""
    T = _clip_T(T)
    try:
        return float(PropsSI("P", "T", T, "Q", 0, FLUID))
    except Exception:
        # fallback: usa Q=0.5 se Q=0 falhar por arredores numéricos
        return float(PropsSI("P", "T", T, "Q", 0.5, FLUID))

def kappa(P1, P2, Pv1):
    num = max(P1 - P2, 0.0)
    den = max(Pv1 - P2, 1e-6)
    return sqrt(num / den)

def m_dot_NHNE(Cd, Ac, P1, P2, rho1, rho2, h1, h2, Pv1):
    k = kappa(P1, P2, Pv1)
    w_hem = 1.0 / (1.0 + k)
    w_inc = 1.0 - w_hem
    minc = m_dot_incompressible(Cd, Ac, rho1, P1, P2)
    mhem = m_dot_HEM(Cd, Ac, rho2, h1, h2)
    return w_inc*minc + w_hem*mhem

# --------------------- dados e integração ---------------------

@dataclass
class Tank:
    V: float    # m^3
    M0: float   # kg
    T0: float   # K

@dataclass
class Injector:
    Cd: float
    D: float    # m
    P2: float   # Pa
    N: int      # Number of Holes


def _psat(T):
    T = _clip_T(T)
    return float(PropsSI("P","T",T,"Q",0,FLUID))

def _rhoL_sat(T):
    T = _clip_T(T)
    return float(PropsSI("D","T",T,"Q",0,FLUID))

def _betaT_liq(T):
    # compressibilidade isotérmica β_T [1/Pa] do líquido saturado
    T = _clip_T(T)
    return float(PropsSI("isothermal_compressibility","T",T,"Q",0,FLUID))

def rho_liq_PT(T, P):
    T = _clip_T(T)
    return float(PropsSI("D","T",T,"P",P,FLUID))  # densidade do líquido comprimido

def M0_from_fill(V_total, fL, T0, P0):
    rhoL = rho_liq_PT(T0, P0)
    Vl0  = fL*V_total
    return rhoL*Vl0  # massa inicial de N2O líquido


def run_blowdown(tank, inj, t_stop=30.0, dt=0.01, mode="isentropic",
                 fL=0.90, pressurante_constante=True, P_const=5.0e6):
    Ac = inj.N*0.25*pi*inj.D**2

    # estado inicial
    M = float(tank.M0)           # massa de N2O líquido
    T = _clip_T(float(tank.T0))
    V = float(tank.V)

    # pressão no tanque
    P = P_const if pressurante_constante else None

    # histórico
    t = 0.0
    hist = {k: [] for k in ["t","M","T","p","mdot"]}

    while t <= t_stop and M > 1e-8:
        # volume de líquido e gás
        if pressurante_constante:
            P1 = P_const
        else:
            P1 = P  # se implementares o modelo do N2, atualizarás isto em baixo

        rhoL = rho_liq_PT(T, P1)    # densidade do N2O líquido no estado real do tanque
        Vl   = M / rhoL
        Vg   = max(V - Vl, 1e-8)

        # propriedades do N2O líquido (usa rhoL, NÃO M/V)
        p1, h1, s1 = state_DT(T, rhoL)  # atenção: 'p1' aqui é o calculado termodinâmico; usa P1 para hidráulica
        # para hidráulica, a pressão montante correta é P1 (do pressurizante)
        P_up = P1

        # estado jusante no injetor
        if mode == "isentropic":
            T2, rho2, p2, h2, s2 = downstream_state(inj.P2, s1=s1)
        else:
            T2, rho2, p2, h2, s2 = downstream_state(inj.P2, h1=h1)

        # Pvap para kappa (em T do tanque)
        Pv1 = _Pvap_from_T(T)

        # caudal NHNE usa P_up como pressão montante
        mdot = m_dot_NHNE(inj.Cd, Ac, P_up, inj.P2, rhoL, rho2, h1, h2, Pv1)

        # integração de Euler sobre a massa e energia do LÍQUIDO
        M_new = max(M - mdot*dt, 1e-9)
        H_old = M * h1
        H_new = H_old - h1*mdot*dt   # entalpia que sai é ~ h1 (saída extraída do tanque)

        h_new = H_new / M_new
        # resolve T novo a partir de h_new e rho_liq(T,P_up) ≈ dependente de T
        # aproxima: itera T mantendo P_up
        def fT(Tguess):
            rho_guess = rho_liq_PT(Tguess, P_up)
            return PropsSI("H","D",rho_guess,"T",_clip_T(Tguess),FLUID) - h_new
        # bissecção simples
        T_lo, T_hi = _T_bounds()
        a,b = T_lo,T_hi
        for _ in range(60):
            m = 0.5*(a+b)
            fm = fT(m)
            fa = fT(a)
            if fm==0 or abs(b-a)<1e-6: T_new=m; break
            if fa*fm<0: b=m
            else: a=m
        else:
            T_new = 0.5*(a+b)

        # registo
        hist["t"].append(t)
        hist["M"].append(M)
        hist["T"].append(T)
        hist["p"].append(P_up)
        hist["mdot"].append(mdot)

        # avanço
        M, T = M_new, T_new
        t += dt

        # critério de paragem hidráulico
        if P_up <= inj.P2: break

        # se quiseres modelo de N2 selado (pressão variável), aqui atualizas P via gás ideal:
        # n_N2 constante, P = n_N2*R*N2 * T / Vg
        if not pressurante_constante:
            P = max(1e5, P)  # placeholder: implementar n_N2 e atualizar P
    for k in hist: hist[k]=np.array(hist[k],dtype=float)
    return hist


# --------------------- exemplo ---------------------

if __name__ == "__main__":
    V   = 0.009
    fL  = 0.90
    T0  = 274.25
    P0  = 4.4e6  # 50 bar

    M0  = M0_from_fill(V, fL, T0, P0)

    tank = Tank(V=V, M0=M0, T0=T0)
    inj  = Injector(Cd=0.80, D=0.0015, P2=1e5, N=12)

    out = run_blowdown(tank, inj, t_stop=30.0, dt=0.05,
                       mode="isentropic", fL=fL,
                       pressurante_constante=True, P_const=P0)


    print(f"N={inj.N} furos, D={inj.D*1000} mm")
    print("Amostras:", out["t"].size)
    print("mdot médio 15 s [kg/s]:", out["mdot"][out["t"] <= 15.0].mean() if out["t"].size else 0.0)

    print("p inicial [MPa]:", out["p"][0]/1e6 if out["p"].size else np.nan)
    print("p final   [MPa]:", out["p"][-1]/1e6 if out["p"].size else np.nan)
