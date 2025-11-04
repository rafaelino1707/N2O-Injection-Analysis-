# === NHNE mass flow vs Tank Pressure and Temperature ===
# Requirements: pip install CoolProp numpy matplotlib
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from math import pi, sqrt

FLUID = "NitrousOxide"

# --- Fixed geometry ---
N_holes = 12
D_hole  = 1.5e-3       # m
L_over_D = 6.7
K_in, K_out = 0.5, 1.0
Pa_per_bar = 1e5
P_cham_bar = 30.0
P2 = P_cham_bar * Pa_per_bar

# --- Helpers ---
def re_from_Gmu(G, D, mu):
    return max(G*D/max(mu,1e-9), 1.0)

def f_blasius(Re):
    return 64.0/Re if Re < 2100 else 0.316*Re**-0.25

def Cd_geom(G, D, mu, L_over_D):
    Re = re_from_Gmu(G, D, mu)
    fD = f_blasius(Re)
    Ktot = K_in + 4*fD*(L_over_D) + K_out
    return 1.0 / np.sqrt(max(Ktot, 1e-9))

def G_HEM(P1, T1, P2, h1):
    try:
        T_sat2 = PropsSI("T","P",P2,"Q",0,FLUID)
        hf2    = PropsSI("H","P",P2,"Q",0,FLUID)
        hg2    = PropsSI("H","P",P2,"Q",1,FLUID)
        rhol2  = PropsSI("D","P",P2,"Q",0,FLUID)
        rhog2  = PropsSI("D","P",P2,"Q",1,FLUID)
    except Exception:
        return 0.0
    xs = np.linspace(0,1,501)
    Gbest=0.0
    for x in xs:
        hm = (1-x)*hf2 + x*hg2
        if h1>hm:
            rm = 1.0/((1-x)/rhol2 + x/rhog2)
            G = rm*np.sqrt(2*(h1-hm))
            if G>Gbest: Gbest=G
    return Gbest

def mdot_NHNE_total(P1, T1):
    P_sat = PropsSI("P","T",T1,"Q",0,FLUID)
    rho_l = PropsSI("D","T",T1,"P",max(P1,1.01*P_sat),FLUID)
    mu_l  = 0.0003
    h1    = PropsSI("H","T",T1,"P",max(P1,1.01*P_sat),FLUID)

    dP = max(P1 - P2, 0.0)
    G = np.sqrt(2*rho_l*dP)
    Cd = Cd_geom(G, D_hole, mu_l, L_over_D)
    G_spi = Cd*np.sqrt(2*rho_l*dP)
    G_hem = G_HEM(P1, T1, P2, h1)
    kappa = np.sqrt(max(P1-P2,0)/max(P_sat-P2,1.0))
    w_HEM = 1/(1+kappa)
    G_nhne = (1-w_HEM)*G_spi + w_HEM*G_hem
    A_hole = 0.25*pi*D_hole**2
    return N_holes*A_hole*G_nhne

# --- Sweep ranges ---
T_range = np.linspace(270, 295, 6)  # 270–295 K (~ -3 to +22°C)
P_range = np.linspace(25, 50, 11)   # bar
m_matrix = np.zeros((len(T_range), len(P_range)))

for i,T1 in enumerate(T_range):
    for j,P_bar in enumerate(P_range):
        P1 = P_bar*Pa_per_bar
        m_matrix[i,j] = mdot_NHNE_total(P1,T1)

# --- Plot for each T (left y-axis = mdot, right y-axis = T) ---
fig, ax1 = plt.subplots(figsize=(7,4))
colors = plt.cm.plasma(np.linspace(0,1,len(T_range)))

for i,T1 in enumerate(T_range):
    ax1.plot(P_range, m_matrix[i,:], color=colors[i], label=f"{T1:.0f} K")

ax1.set_xlabel("Tank Pressure [bar]")
ax1.set_ylabel("Total NHNE mass flow [kg/s]", color="tab:blue")
ax1.tick_params(axis='y', labelcolor="tab:blue")
ax1.grid(True, which='both', ls=':')

# secondary y-axis just to show temperature scale (visual reference)
ax2 = ax1.twinx()
ax2.set_ylabel("Tank Temperature [K]", color="tab:red")
ax2.set_ylim(T_range[0]-2, T_range[-1]+2)
ax2.tick_params(axis='y', labelcolor="tab:red")

ax1.legend(title="Tank T [K]", fontsize=8)
plt.title("NHNE mass flow vs Tank Pressure and Temperature (N₂O)")
plt.tight_layout()
plt.show()
