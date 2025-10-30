import math
import numpy as np
import matplotlib.pyplot as plt
from CoolProp.CoolProp import PropsSI

def m_ox_NHNE(Pt_bar, Pc_bar, T_C, n_holes, D_m,
              Cd=0.67, mode="isentropic", fluid="NitrousOxide"):
    """
    NHNE mass-flow: mdot = (1-W)*mdot_inc + W*mdot_HEM,  W = 1/(1+kappa)
    kappa = sqrt((P1-P2)/(Pv1-P2)), com Pv1 = Psat(T1)
    Upstream: assume estado saturado ao T1 (conservador p/ N2O em tanque bifásico).
    Downstream: resolve (P2, S=s1) se isentropic, ou (P2, H=h1) se adiabatic.
    """
    Pt = Pt_bar*1e5
    Pc = Pc_bar*1e5
    if Pc >= Pt:  # sem queda de pressão
        return 0.0

    T1 = T_C + 273.15

    # --- propriedades a montante (saturado a T1)
    # se T1 estiver fora da cúpula, CoolProp lança; neste caso, deverias fechar mistura com LSQ
    rhoL = PropsSI("D","T",T1,"Q",0,fluid)
    rhoV = PropsSI("D","T",T1,"Q",1,fluid)
    hL   = PropsSI("H","T",T1,"Q",0,fluid)
    hV   = PropsSI("H","T",T1,"Q",1,fluid)
    sL   = PropsSI("S","T",T1,"Q",0,fluid)
    sV   = PropsSI("S","T",T1,"Q",1,fluid)
    # sem qualidade conhecida: usa líquido como proxy para incompressível
    rho1 = rhoL
    h1   = hL
    s1   = sL

    # --- jusante
    if mode.lower().startswith("isen"):
        # isentropic: S2 = s1
        T2  = PropsSI("T","P",Pc,"S",s1,fluid)
        rho2= PropsSI("D","P",Pc,"S",s1,fluid)
        h2  = PropsSI("H","P",Pc,"S",s1,fluid)
    elif mode.lower().startswith("adiab"):
        # adiabatic expansion: resolve T2 tal que H(T2,P2)=H1  → mas depois recalcula h2
        T2  = PropsSI("T","P",Pc,"H",h1,fluid)
        rho2= PropsSI("D","P",Pc,"T",T2,fluid)
        h2  = PropsSI("H","P",Pc,"T",T2,fluid)


    # --- NHNE
    Pv1   = PropsSI("P","T",T1,"Q",0,fluid)  # pressão de vapor a montante
    kappa = math.sqrt(max(Pt-Pc, 0.0) / max(Pv1-Pc, 1.0))
    W     = 1.0/(1.0 + kappa)                # peso do HEM (Solomon)

    A_hole = math.pi*(D_m*0.5)**2
    A_inj  = n_holes*A_hole

    mdot_inc = Cd * A_inj * math.sqrt(2.0*rho1*max(Pt-Pc,0.0))
    dh       = max(h1 - h2, 0.0)
    mdot_HEM = Cd * A_inj * rho2 * math.sqrt(2.0*dh)

    mdot = (1.0 - W)*mdot_inc + W*mdot_HEM
    return mdot

# Exemplo de varrimento
if __name__ == "__main__":
    Pt_bar = 40.0
    T_C    = 10.0
    D_m    = 1.5e-3
    Cd     = 0.67
    Ns     = [i for i in range(1,15)]

    Pc_vec = np.linspace(1.0, Pt_bar-0.1, 60)
    plt.figure(figsize=(10,5))
    for N in Ns:
        m = [m_ox_NHNE(Pt_bar, Pc, T_C, N, D_m, Cd, mode="isentropic", fluid="NitrousOxide")
             for Pc in Pc_vec]
        plt.plot(Pc_vec, m, label=f"N={N}")
    plt.xlabel("Chamber Pressure [bar]")
    plt.ylabel("Mass Flow [kg/s]")
    plt.title(f"NHNE mdot vs Pc (D={D_m*1e3:.1f} mm, Cd={Cd})")
    plt.grid(True); plt.legend(); plt.show()
