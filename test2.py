# Solomon-like N2O blowdown: NHNE + Helmholtz (CoolProp) + Euler (H,H/M)
# pip install numpy scipy CoolProp matplotlib

import numpy as np
from dataclasses import dataclass
from math import pi, sqrt
from scipy.optimize import root
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt
import csv

FLUID = "NitrousOxide"

# ---------------- utilidades termodinâmicas ---------------- #

def T_bounds():
    Tmin = float(PropsSI("Tmin", FLUID)) + 1.0
    Tcr  = float(PropsSI("Tcrit", FLUID)) - 1.0
    return Tmin, Tcr

def clipT(T):
    lo, hi = T_bounds()
    return float(max(min(T, hi), lo))

def sat_T(T):
    T = clipT(T)
    p   = float(PropsSI("P","T",T,"Q",0,FLUID))          # p @ sat
    rhoL= float(PropsSI("D","T",T,"Q",0,FLUID))
    rhoV= float(PropsSI("D","T",T,"Q",1,FLUID))
    hL  = float(PropsSI("H","T",T,"Q",0,FLUID))
    hV  = float(PropsSI("H","T",T,"Q",1,FLUID))
    sL  = float(PropsSI("S","T",T,"Q",0,FLUID))
    sV  = float(PropsSI("S","T",T,"Q",1,FLUID))
    return p, rhoL, rhoV, hL, hV, sL, sV

def mix_from_T_rho(T, rho):
    """Mistura bifásica por 1/rho = x/rhoV + (1-x)/rhoL. Fora de [0,1] usa HEOS direto."""
    T   = clipT(T)
    rho = max(float(rho), 1e-9)
    p, rhoL, rhoV, hL, hV, sL, sV = sat_T(T)
    invrho = 1.0/rho
    x = (invrho - 1.0/rhoL) / (1.0/rhoV - 1.0/rhoL)
    if 0.0 <= x <= 1.0:
        h = (1.0-x)*hL + x*hV
        s = (1.0-x)*sL + x*sV
        return p, x, h, s
    # monofásico
    p1 = float(PropsSI("P","D",rho,"T",T,FLUID))
    h1 = float(PropsSI("H","D",rho,"T",T,FLUID))
    s1 = float(PropsSI("S","D",rho,"T",T,FLUID))
    return p1, None, h1, s1

def solve_T_from_h_rho(h_target, rho):
    """Resolve T tal que h(D=rho, T)=h_target (considera mistura)."""
    rho = max(float(rho), 1e-9)
    lo, hi = T_bounds()
    def f(T):
        _, _, h, _ = mix_from_T_rho(T, rho)
        return h - h_target
    Ts = np.linspace(lo, hi, 80)
    fs = [f(T) for T in Ts]
    i0 = int(np.argmin(np.abs(fs)))
    if abs(fs[i0]) < 1e-6: return float(Ts[i0])
    br = None
    for i in range(len(Ts)-1):
        if fs[i]*fs[i+1] < 0: br = (Ts[i], Ts[i+1]); break
    if br is None: return float(Ts[i0])
    a,b = br; fa,fb = f(a), f(b)
    for _ in range(80):
        m = 0.5*(a+b); fm = f(m)
        if fm == 0.0 or abs(b-a) < 1e-6: return float(m)
        if fa*fm < 0: b,fb = m,fm
        else: a,fa = m,fm
    return float(0.5*(a+b))

# ---------------- modelos de caudal ---------------- #

def m_incomp(Cd, Ac, rho1, P1, P2):
    dP = max(P1-P2, 0.0)
    return Cd*Ac*sqrt(2.0*rho1*dP)

def m_HEM(Cd, Ac, rho2, h1, h2):
    dh = max(h1-h2, 0.0)
    return Cd*Ac*rho2*sqrt(2.0*dh)

def kappa(P1, P2, Pv1):
    num = max(P1-P2, 0.0)
    den = max(Pv1-P2, 1e-6)
    return sqrt(num/den)

def m_NHNE(Cd, Ac, P1, P2, rho1, rho2, h1, h2, Pv1):
    k = kappa(P1,P2,Pv1)
    wHEM  = 1.0/(1.0+k)
    wINC  = 1.0 - wHEM
    return wINC*m_incomp(Cd,Ac,rho1,P1,P2) + wHEM*m_HEM(Cd,Ac,rho2,h1,h2)

# ---------------- downstream solver ---------------- #

def downstream_state(P2, s1=None, h1=None):
    """Resolve T2,rho2 com p=P2 e s2=s1 (isent.) ou h2=h1 (adiab.)."""
    assert (s1 is None) ^ (h1 is None)
    lo,hi = T_bounds()
    x0 = np.array([0.5*(lo+hi), 10.0], float)
    def F(v):
        T  = clipT(v[0]); rho = max(v[1], 1e-7)
        p  = PropsSI("P","D",rho,"T",T,FLUID)
        s  = PropsSI("S","D",rho,"T",T,FLUID)
        h  = PropsSI("H","D",rho,"T",T,FLUID)
        return np.array([p-P2, (s-s1) if s1 is not None else (h-h1)], float)
    sol = root(F, x0)
    T2  = clipT(sol.x[0]); rho2 = max(sol.x[1], 1e-7)
    p2,h2,s2 = PropsSI("P","D",rho2,"T",T2,FLUID), PropsSI("H","D",rho2,"T",T2,FLUID), PropsSI("S","D",rho2,"T",T2,FLUID)
    return T2, rho2, p2, h2, s2

# ---------------- dados ---------------- #

@dataclass
class Tank:
    V: float     # m^3
    M0: float    # kg
    T0: float    # K

@dataclass
class Injector:
    Cd: float
    D: float     # furo [m]
    N: int       # nº furos

def M0_from_fill(V_total, fL, T0, P0):
    """Estimativa para tanque pressurizado: usa ρ(T0,P0) do líquido."""
    T0 = clipT(T0)
    rhoL = float(PropsSI("D","T",T0,"P",P0,FLUID))
    return rhoL * fL * V_total

# ---------------- simulação (sem linha, P2 fixo) ---------------- #
def Cd_time(Cd_nom, t, t_start=0.0, t_open=0.15):
    """Cd efetivo: 0 antes de t_start; rampa cosseno até Cd_nom em t_start+t_open."""
    if t <= t_start: 
        return 0.0
    x = (t - t_start)/max(t_open,1e-9)
    if x >= 1.0:
        return Cd_nom
    return Cd_nom*0.5*(1.0 - np.cos(np.pi*x))  # S-curve suave


def run_blowdown(tank: Tank, inj: Injector, mode_tank="autopressurized",
                 P_const=5.0e6, P2=1.0e5, dt=0.05, t_stop=30.0,
                 downstream_mode="isentropic",t_start=0.0, t_open=0.9):
    Ac = inj.N * 0.25*pi*inj.D**2

    # estado inicial
    M  = float(tank.M0)
    Vt = float(tank.V)
    rho= M / Vt
    T  = clipT(float(tank.T0))

    # mistura inicial -> h0 e H0
    p_m, x0, h0, s0 = mix_from_T_rho(T, rho)
    H  = M * h0

    t = 0.0
    hist = {k: [] for k in ["t","M","T","p1","mdot"]}

    while t <= t_stop and M > 1e-8:
        # pressão montante
        if mode_tank == "autopressurized":
            P1 = sat_T(T)[0]        # Psat(T)
        elif mode_tank == "pressurized":
            P1 = float(P_const)
        else:
            raise ValueError("mode_tank inválido")

        # upstream props a partir de mistura (ρ=M/V, T)
        p1, x, h1, s1 = mix_from_T_rho(T, rho)
        Pv1 = sat_T(T)[0]           # para kappa

        # downstream state (P2 fixo)
        if downstream_mode == "isentropic":
            T2, rho2, _, h2, _ = downstream_state(P2, s1=s1)
        else:
            T2, rho2, _, h2, _ = downstream_state(P2, h1=h1)

        # caudal NHNE
        rho_up_for_inc = float(PropsSI("D","T",T,"P",max(P1,1e5),FLUID)) if x is None else float(PropsSI("D","T",T,"Q",0,FLUID))
        Cd_eff = Cd_time(inj.Cd, t, t_start=t_start, t_open=t_open)
        mdot   = m_NHNE(Cd_eff, Ac, P1, P2, rho_up_for_inc, rho2, h1, h2, Pv1)


        # balanços por Euler em H e M
        M_new = max(M - mdot*dt, 1e-9)
        H_new = H - h1*mdot*dt
        rho_new = M_new / Vt
        h_new = H_new / M_new
        T_new = solve_T_from_h_rho(h_new, rho_new)

        # registo
        hist["t"].append(t);   hist["M"].append(M);   hist["T"].append(T)
        hist["p1"].append(P1); hist["mdot"].append(mdot)

        # avanço
        M, H, rho, T = M_new, H_new, rho_new, T_new
        t += dt

        if P1 <= P2: break

    for k in hist: hist[k] = np.array(hist[k], float)
    return hist, t

# ---------------- exemplo ---------------- #

if __name__ == "__main__":
    # parâmetros de referência
    Vtank = 0.009
    fL    = 0.90
    T0    = 283.15        # K
    Ppress= 5.0e6         # Pa (50 bar) se usar modo pressurized
    P2    = 3.0e6         # Pa backpressure fixo

    M0 = M0_from_fill(Vtank, fL, T0, Ppress)  # ou introduz manualmente

    tank = Tank(V=Vtank, M0=M0, T0=T0)
    inj  = Injector(Cd=0.67, D=1.5e-3, N=40)

    # escolhe o modo
    mode_tank = "pressurized"   # ou "pressurized"
    hist, t_end = run_blowdown(tank, inj, mode_tank=mode_tank,
                               P_const=Ppress, P2=P2,
                               dt=0.1, t_stop=30.0,
                               downstream_mode="isentropic")

    print(f"fim @ t={t_end:.2f} s | amostras={len(hist['t'])}")
    print(f"ṁ médio 5 s: {hist['mdot'][hist['t']<=5].mean():.3f} kg/s")

    # CSV
    with open("solomon_min.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["t","M","T","P1","mdot"])
        for i in range(len(hist["t"])):
            w.writerow([hist["t"][i], hist["M"][i], hist["T"][i], hist["p1"][i], hist["mdot"][i]])
    print("gravado: solomon_min.csv")

    # gráficos
    fig,ax=plt.subplots(2,1,figsize=(7,6),sharex=True)
    ax[0].plot(hist["t"], hist["mdot"], "r"); ax[0].set_ylabel("ṁ [kg/s]"); ax[0].set_title(f"Mass Flow vs Time for Nholes={inj.N} & D={inj.D}")
    ax[1].plot(hist["t"], hist["p1"]/1e5);   ax[1].set_ylabel("P1 [bar]"); ax[1].set_xlabel("t [s]")
    ax[1].grid(alpha=0.3); plt.tight_layout(); plt.show()
