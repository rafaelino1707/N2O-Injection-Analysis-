# N2O Blowdown tipo Solomon: NHNE + CoolProp + Euler
# modos: tank_mode = "pressurized" | "autopressurized"
import csv
import numpy as np
from math import pi, sqrt
from dataclasses import dataclass
from scipy.optimize import root
from CoolProp.CoolProp import PropsSI
import matplotlib.pyplot as plt

FLUID = "NitrousOxide"

# ---------------- utilidades ---------------- #

def _T_bounds():
    Tmin = float(PropsSI("Tmin", FLUID)) + 1.0
    Tcrit = float(PropsSI("Tcrit", FLUID)) - 1.0
    return Tmin, Tcrit

def _clip_T(T):
    T_lo, T_hi = _T_bounds()
    return float(max(min(T, T_hi), T_lo))

def _psat(T):
    return float(PropsSI("P", "T", _clip_T(T), "Q", 0, FLUID))

EPS_PSAT = 1e-3  # 0.1%

def rho_liq_PT(T, P):
    T = _clip_T(T)
    Ps = _psat(T)
    if P <= Ps*(1.0 + EPS_PSAT):
        return float(PropsSI("D","T",T,"Q",0,FLUID))  # lado líquido em sat
    return float(PropsSI("D","T",T,"P",P,FLUID))

def Cp_liq_PT(T, P):
    T = _clip_T(T)
    Ps = _psat(T)
    if P <= Ps*(1.0 + EPS_PSAT):
        return float(PropsSI("Cpmass","T",T,"Q",0,FLUID))
    return float(PropsSI("Cpmass","T",T,"P",P,FLUID))


def state_DT(T, rho):
    T = _clip_T(T); rho = max(float(rho), 1e-9)
    p = PropsSI("P", "D", rho, "T", T, FLUID)
    h = PropsSI("H", "D", rho, "T", T, FLUID)
    s = PropsSI("S", "D", rho, "T", T, FLUID)
    return p, h, s

# ---------------- escoamento ---------------- #

def m_dot_incompressible(Cd, Ac, rho1, P1, P2):
    dP = max(P1-P2, 0.0);  return Cd*Ac*sqrt(2.0*rho1*dP)

def m_dot_HEM(Cd, Ac, rho2, h1, h2):
    dh = max(h1-h2, 0.0);  return Cd*Ac*rho2*sqrt(2.0*dh)

def _Pvap_from_T(T):
    try:    return float(PropsSI("P","T",_clip_T(T),"Q",0,FLUID))
    except: return float(PropsSI("P","T",_clip_T(T),"Q",0.5,FLUID))

def kappa(P1, P2, Pv1):
    num=max(P1-P2,0.0); den=max(Pv1-P2,1e-6); return sqrt(num/den)

def flash_weight(P1, T, P2, Pv1, dP_flash=3e5):
    # 0 => líquido seguro; 1 => flash dominante
    dP_sub = max(P1 - Pv1, 0.0)
    w = 1.0 - np.clip(dP_sub/(dP_flash+1e-6), 0.0, 1.0)
    w2 = np.clip((Pv1 - P2)/max(P1-P2,1e-6), 0.0, 1.0)
    return max(w, w2)

def m_dot_NHNE(Cd, Ac, P1, P2, rho1, rho2, h1, h2, Pv1, T=None):
    k = kappa(P1, P2, Pv1)
    w_hem = 1.0/(1.0+k)
    if T is not None:
        w_hem = max(w_hem, flash_weight(P1, T, P2, Pv1))
    w_inc = 1.0 - w_hem
    return (w_inc*m_dot_incompressible(Cd,Ac,rho1,P1,P2)
          +  w_hem*m_dot_HEM(Cd,Ac,rho2,h1,h2))

def mdot_choked_cap(Cd, Ac, T, rho_guess):
    # tampão de estrangulamento bifásico simples
    # usa uma-velocidade a partir de CoolProp (mistura saturada)
    try:
        a = float(PropsSI("A","T",_clip_T(T),"Q",0.5,FLUID))
    except:
        a = float(PropsSI("A","T",_clip_T(T),"Q",0,FLUID))
    a = max(a, 1.0)
    rho = max(rho_guess, 50.0)  # evita valores irreais
    return Cd * Ac * rho * a

# ---------------- downstream solver ---------------- #

def downstream_state(P2, s1=None, h1=None, T_ref=None, rho_ref=None):
    assert (s1 is None) ^ (h1 is None)
    T_lo, T_hi = _T_bounds()
    T0 = _clip_T(T_ref) if T_ref is not None else 0.5*(T_lo+T_hi)
    r0 = max(float(rho_ref), 1e-3) if rho_ref is not None else 10.0
    x0 = np.array([T0, r0], float)

    def F(v):
        T = _clip_T(v[0]); rho = max(v[1], 1e-8)
        p = PropsSI("P","D",rho,"T",T,FLUID)
        s = PropsSI("S","D",rho,"T",T,FLUID)
        h = PropsSI("H","D",rho,"T",T,FLUID)
        return np.array([p-P2, (s-s1) if s1 is not None else (h-h1)], float)

    sol = root(F, x0)
    T2 = _clip_T(sol.x[0]); rho2 = max(sol.x[1], 1e-8)
    p2, h2, s2 = state_DT(T2, rho2)
    return T2, rho2, p2, h2, s2


# ---------------- válvula ---------------- #

def Cd_time(Cd_nom, t, t_open=0.15):
    if t <= 0.0: return 0.0
    if t >= t_open: return Cd_nom
    x = t/t_open
    return Cd_nom*0.5*(1-np.cos(np.pi*x))

# ---------------- dados ---------------- #

@dataclass
class Tank:
    V: float
    M0: float
    T0: float

@dataclass
class Line:
    V: float          # volume de linha [m3]
    T0: float         # K
    filled: bool      # começa cheia?
    Cd_fac: float=1.0 # multiplicador em perdas internas

@dataclass
class Downstream:
    V: float          # volume jusante [m3]
    T: float          # K (mantido)
    P0: float         # Pa inicial

@dataclass
class Injector:
    Cd: float
    D: float
    N: int

def M0_from_fill(V_total, fL, T0, P0):
    rhoL = rho_liq_PT(T0, P0)
    return rhoL*fL*V_total

# ---------------- simulação ---------------- #

def run_sim(tank: Tank, inj: Injector, line: Line, down: Downstream,
            tank_mode="pressurized", P_const=5.0e6, P2_fixed=None,
            dt=0.01, t_stop=30.0, t_open=0.15):

    Ac = inj.N * 0.25*pi*inj.D**2

    # tanque
    M_t = float(tank.M0)
    T_t = _clip_T(float(tank.T0))

    # linha
    M_l = 0.0 if not line.filled else rho_liq_PT(line.T0, P_const)*line.V
    H_l = 0.0
    T_l = float(line.T0)

    # downstream
    R_u = 8.314462618
    M_gas = (down.P0*down.V)/(R_u*down.T)  # moles; pressão variável
    P2 = down.P0 if P2_fixed is None else P2_fixed

    t=0.0
    hist = {k: [] for k in ["t","p1","p_line","p2","mdot_out"]}

    line_full = False
    step = 0

    while t <= t_stop and M_t > 1e-6:
        print(f't --...')
        # pressão do tanque
        if tank_mode=="pressurized":
            P1 = max(P_const, _psat(T_t)*(1+1e-5))
        elif tank_mode=="autopressurized":
            P1 = _psat(T_t)
        else:
            raise ValueError("tank_mode inválido")

        # propriedades tanque
        rhoL_t = rho_liq_PT(T_t, P1)
        p_t, h_t, s_t = state_DT(T_t, rhoL_t)
        Pv1 = _Pvap_from_T(T_t)

        # pressão na linha (aprox: se não cheia, segue P1; se cheia, calcula por compressibilidade)
                # queda mínima através da válvula de entrada durante o enchimento
        # --- pressão na linha e critério de enchimento ---
        rhoL_line_ref = rho_liq_PT(T_l if line_full else T_t, P1)
        fill_mass = rhoL_line_ref * line.V

        if not line_full:
            dP_in_fill = 1.0e5           # 1 bar para gerar ΔP de enchimento
            P_line = max(P2, P1 - dP_in_fill)
        else:
            dP_minor  = 0.005 * P1       # 0.5% de perdas quando cheia
            P_line = max(P2, P1 - dP_minor)

        # --- fluxo tanque->linha ---
        Cd_in = Cd_time(inj.Cd*line.Cd_fac, t, t_open=t_open)
        T2i, rho2i, _, h2i, _ = downstream_state(P_line, s1=s_t, T_ref=T_t, rho_ref=rhoL_t)
        mdot_in = m_dot_NHNE(Cd_in, Ac, P1, P_line, rhoL_t, rho2i, h_t, h2i, Pv1, T=T_t)
        mdot_in = min(mdot_in, mdot_choked_cap(Cd_in, Ac, T_t, rho2i))

        # --- atualizar massa na linha e travar ao volume ---
        M_l_new = min(M_l + mdot_in*dt, fill_mass)
        if (not line_full) and (M_l_new >= 0.999*fill_mass):
            line_full = True


        # --- Escoamento jusante (só se linha estiver cheia) --- #
        # --- escoamento linha->jusante ---
        if line_full:
            rhoL_l = rho_liq_PT(T_l, P_line)
            p_l, h_l, s_l = state_DT(T_l, rhoL_l)
            Cd_out = Cd_time(inj.Cd, t, t_open=t_open)
            T2o, rho2o, _, h2o, _ = downstream_state(P2, s1=s_l, T_ref=T_l, rho_ref=rhoL_l)
            mdot_out = m_dot_NHNE(Cd_out, Ac, P_line, P2, rhoL_l, rho2o, h_l, h2o, _Pvap_from_T(T_l), T=T_l)
            mdot_out = min(mdot_out, mdot_choked_cap(Cd_out, Ac, T_l, rho2o))
        else:
            mdot_out = 0.0




        # balanços
        # tanque
        dM_t = -mdot_in*dt
        M_t_new = max(M_t + dM_t, 1e-9)
        H_t = M_t*h_t
        H_t_new = H_t - h_t*mdot_in*dt
        h_t_new = H_t_new/M_t_new
        Cp_t = Cp_liq_PT(T_t, P1)
        T_t_new = _clip_T(T_t + (h_t_new - h_t)/Cp_t)

        # linha
        if line_full:
            # energia na linha
            if M_l < 1e-9: h_l_old = h2i
            else:
                rhoL_l = rho_liq_PT(T_l, P_line)
                h_l_old = PropsSI("H","T",T_l,"D",rhoL_l,FLUID)
            H_l = M_l*h_l_old
            H_l_new = H_l + h_t*mdot_in*dt - h_l_old*mdot_out*dt
            M_l_new = max(M_l + (mdot_in - mdot_out)*dt, 1e-9)
            h_l_new = H_l_new/M_l_new
            Cp_l = Cp_liq_PT(T_l, P_line)
            T_l_new = _clip_T(T_l + (h_l_new - h_l_old)/Cp_l)
        else:
            T_l_new = T_l

        # downstream pressão dinâmica se não for fixa
        if P2_fixed is None:
            # assume gás ideal a T_down constante
            M_gas += (mdot_out*dt)/PropsSI("molar_mass",FLUID)  # para moles
            P2 = (M_gas*R_u*down.T)/down.V
        print(f"t={t:6.3f}s | P1={P1/1e5:7.2f}bar P_line={P_line/1e5:7.2f}bar P2={P2/1e5:6.2f}bar "
      f"| md_in={mdot_in:7.3f} md_out={mdot_out:7.3f} | M_line={M_l_new:7.3f} "
      f"| full={line_full}")
        step += 1

        # registo
        hist["t"].append(t)
        hist["p1"].append(P1)
        hist["p_line"].append(P_line)
        hist["p2"].append(P2)
        hist["mdot_out"].append(mdot_out)

        # avanço
        M_t, T_t = M_t_new, T_t_new
        M_l, T_l = M_l_new, T_l_new
        t += dt

        # fim por pressão
        if P1 <= P2: break

    for k in hist: hist[k]=np.array(hist[k],float)
    return hist, t

# ---------------- exemplo ---------------- #

if __name__ == "__main__":
    # d
    V_tank = 0.009
    fL = 0.90
    T0 = 283.15
    P0 = 4.4e6  # Pa
    M0 = M0_from_fill(V_tank, fL, T0, P0)

    tank = Tank(V=V_tank, M0=M0, T0=T0)
    line = Line(V=2.0e-4, T0=T0, filled=False, Cd_fac=1.0)  # 200 ml de linha
    down = Downstream(V=1.0e-3, T=293.15, P0=1.0e5)        # 1 litro jusante
    inj  = Injector(Cd=0.67, D=0.0015, N=12)

    # escolha do modo de tanque
    tank_mode = "pressurized"   # não autopressurized
    P_const   = 5.0e6           # ~50 bar
    P2_fixed  = 1.0e5           # 1 bar
               # define valor em Pa para manter P2 fixo

    P2_fixed = 1.0e5  # 1 bar fixo (backpressure constante)

    hist, t_end = run_sim(tank, inj, line, down,
                          tank_mode=tank_mode, P_const=P_const, P2_fixed=P2_fixed,
                          dt=0.01, t_stop=5.0, t_open=0.15)

    print(f"Modo: {tank_mode} | Tempo sim: {t_end:.2f} s")
    print(f"mdot médio: {hist['mdot_out'].mean():.3f} kg/s")
    print(f"P1 inicial: {hist['p1'][0]/1e5:.1f} bar → final: {hist['p1'][-1]/1e5:.1f} bar")
    print(f"P2 inicial: {hist['p2'][0]/1e5:.2f} bar → final: {hist['p2'][-1]/1e5:.2f} bar")

    # gráficos
    fig, ax = plt.subplots(2,1, figsize=(7,6), sharex=True)
    ax[0].plot(hist["t"], hist["mdot_out"], "r")
    ax[0].set_ylabel("Mass flow [kg/s]")
    ax[0].set_title("Mass Flow vs Time")

    ax[1].plot(hist["t"], hist["p1"]/1e5, label="P1 tank")
    ax[1].plot(hist["t"], hist["p_line"]/1e5, label="P_line")
    ax[1].plot(hist["t"], hist["p2"]/1e5, label="P2 down")
    ax[1].set_xlabel("Time [s]"); ax[1].set_ylabel("Pressure [bar]")
    ax[1].legend(); ax[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()

    with open("sim_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        # cabeçalho
        writer.writerow(["t [s]", "P1 [Pa]", "P_line [Pa]", "P2 [Pa]", "mdot_out [kg/s]"])
        # linhas
        for i in range(len(hist["t"])):
            writer.writerow([
                hist["t"][i],
                hist["p1"][i],
                hist["p_line"][i],
                hist["p2"][i],
                hist["mdot_out"][i]
            ])

        print("Resultados guardados em sim_results.csv")