# --- Heatmap version ---
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI
from math import pi

FLUID = "NitrousOxide"
N_holes, D_hole, Cd = 12, 1.5e-3, 0.67
A_hole = 0.25*pi*D_hole**2
P_cham_bar, Pa_per_bar = 30.0, 1e5
P2 = P_cham_bar * Pa_per_bar

def mdot_NHNE_total(P1, T1):
    P_sat = PropsSI("P","T",T1,"Q",0,FLUID)
    rho_l = PropsSI("D","T",T1,"P",max(P1,1.01*P_sat),FLUID)
    dP = max(P1 - P2, 0.0)
    h1 = PropsSI("H","T",T1,"P",max(P1,1.01*P_sat),FLUID)
    s1 = PropsSI("S","T",T1,"P",max(P1,1.01*P_sat),FLUID)
    try:
        T2 = PropsSI("T","P",P2,"S",s1,FLUID)
        h2 = PropsSI("H","P",P2,"T",T2,FLUID)
    except Exception:
        T2 = PropsSI("T","P",P2,"Q",0,FLUID)
        hL = PropsSI("H","T",T2,"Q",0,FLUID)
        hV = PropsSI("H","T",T2,"Q",1,FLUID)
        h2 = 0.5*(hL + hV)
    rho2 = PropsSI("D","P",P2,"T",T2,FLUID)
    dh = max(h1 - h2, 0.0)
    m_inc = Cd*A_hole*np.sqrt(2.0*rho_l*dP)
    m_hem = Cd*A_hole*rho2*np.sqrt(2.0*dh)
    kappa = np.sqrt(max(P1-P2,0.0)/max(P_sat-P2,1.0))
    W = 1.0/(1.0+kappa)
    m_hole = (1.0-W)*m_inc + W*m_hem
    return N_holes*m_hole

# Grid (P,T)
P_range_bar = np.linspace(25,50,50)
T_range_K = np.linspace(270,295,40)
P_grid, T_grid = np.meshgrid(P_range_bar, T_range_K)

m_grid = np.zeros_like(P_grid)
for i in range(P_grid.shape[0]):
    for j in range(P_grid.shape[1]):
        m_grid[i,j] = mdot_NHNE_total(P_grid[i,j]*Pa_per_bar, T_grid[i,j])

# --- Plot heatmap ---
fig, ax = plt.subplots(figsize=(7,5))
c = ax.contourf(P_grid, T_grid, m_grid, levels=30, cmap="plasma")
cb = plt.colorbar(c, ax=ax)
cb.set_label(r"Total $\dot m_{\mathrm{NHNE}}$ [kg/s]")

ax.set_xlabel("Tank Pressure [bar]")
ax.set_ylabel("Tank Temperature [K]")
plt.title("NHNE mass flow map (N₂O)")
plt.tight_layout()
plt.show()
