# === NHNE mass-flow loss vs L/D ==========================================
# Requirements: pip install CoolProp numpy
from CoolProp.CoolProp import PropsSI
import numpy as np
from math import pi, sqrt

FLUID = "NitrousOxide"

# ---------------- User Inputs (SI) ----------------
m_dot_target = 1.5            # kg/s (apenas para contexto; não é usado no cálculo da perda)
P_tank_bar   = 44.0           # bar (montante)
T_tank_K     = 273.15         # K
P_cham_bar   = 30.0           # bar (jusante)

# Geometria (fixa N e D; L muda para varrer L/D):
N_holes      = 12
D_hole       = 1.5e-3         # m
L_over_D_now = 10.0/1.5       # L/D atual (p.ex., 10 mm / 1.5 mm -> 6.67)
L_over_D_ref = 2.0            # L/D "ideal" de referência (edita aqui)

# Perdas localizadas típicas (ajusta se tiveres medições)
K_in  = 0.5                   # entrada com aresta/chanfrado leve
K_out = 1.0                   # saída livre

# ---------------- Convenience ----------------
Pa_per_bar = 1e5
P1 = P_tank_bar * Pa_per_bar
P2 = P_cham_bar * Pa_per_bar
A_hole = 0.25*pi*D_hole**2
A_tot  = N_holes * A_hole

# Propriedades a montante (líquido comprimido; reais)
P_sat = PropsSI("P", "T", T_tank_K, "Q", 0, FLUID)
rho_l = PropsSI("D", "T", T_tank_K, "P", max(P1, 1.01*P_sat), FLUID)
mu_l  = 0.6e-3
h1    = PropsSI("H", "T", T_tank_K, "P", max(P1, 1.01*P_sat), FLUID)

# Saturadas a P2 (para HEM)
T_sat2 = PropsSI("T", "P", P2, "Q", 0, FLUID)
hf2    = PropsSI("H", "P", P2, "Q", 0, FLUID)
hg2    = PropsSI("H", "P", P2, "Q", 1, FLUID)
rhol2  = PropsSI("D", "P", P2, "Q", 0, FLUID)
rhog2  = PropsSI("D", "P", P2, "Q", 1, FLUID)

# Parâmetro NHNE (Dyer/Solomon)
kappa = np.sqrt((P1 - P2) / max(P_sat - P2, 1.0))

# ---------------- Helpers ----------------
def sqrt_pos(x): return np.sqrt(max(x, 0.0))

def re_from_Gmu(G, D, mu):
    # Re_local = G*D/mu  (pois G = rho*v e Re = rho*v*D/mu)
    return max(G*D/max(mu,1e-9), 1.0)

def darcy_f_blasius(Re):
    # laminar ou turbulento liso; uso contínuo e robusto
    if Re < 2100.0:
        return 64.0/Re
    else:
        return 0.316 * (Re**-0.25)

def Cd_geom_from_LD(G, L_over_D):
    # fecha f com Re baseado em G e viscosidade de montante (robusto e simples)
    Re = re_from_Gmu(G, D_hole, mu_l)
    fD = darcy_f_blasius(Re)
    Ktot = K_in + 4.0*fD*(L_over_D) + K_out
    return 1.0 / sqrt(max(Ktot, 1e-9))

def G_HEM(P1, T1, P2):
    # Max G sobre qualidade x a P2 (mistura saturada)
    def hmix(x):  return (1.0-x)*hf2 + x*hg2
    def rhomix(x): return 1.0 / ((1.0-x)/rhol2 + x/rhog2)
    xs = np.linspace(0.0, 1.0, 1001)
    Gbest = 0.0
    for x in xs:
        hm = hmix(x)
        if h1 > hm:
            rm = rhomix(x)
            G = rm * sqrt(2.0*(h1 - hm))
            if G > Gbest:
                Gbest = G
    return Gbest

G_HEM_cap = G_HEM(P1, T_tank_K, P2)  # independente de L/D

def G_SPI_from_LD(L_over_D, tol=1e-6, itmax=60):
    """
    Resolve o ramo hidráulico compressível:
    Δp = K_in * (G^2 / (2 ρ*)) + ∫ (4 f/D) * (G^2 / (2 ρ(x))) dx + K_out * (G^2 / (2 ρ**))
    Aproximação robusta: usa ρ de montante para normalização e f(Re(G)).
    Como G aparece dos dois lados, faz-se iteração fixa em G.
    """
    dP = max(P1 - P2, 0.0)
    # chute inicial: orifício ideal com Cd=0.7 (apenas para arrancar iteração)
    G = 0.7 * sqrt(2.0 * rho_l * dP)

    for _ in range(itmax):
        Cd = Cd_geom_from_LD(G, L_over_D)
        # Com esse Cd_geom, o "SPI efetivo" vira G = Cd * sqrt(2 rho_l dP) / (ajuste por L/D já embutido no Cd)
        # Mas para não cair na forma incompressível explícita, fechamos por equilíbrio:
        # Δp ≈ (K_in + 4 f L/D + K_out)*(G^2/(2 ρ_ref)), com ρ_ref≈ρ_l
        Re   = re_from_Gmu(G, D_hole, mu_l)
        fD   = darcy_f_blasius(Re)
        Ktot = K_in + 4.0*fD*(L_over_D) + K_out
        Gnew = sqrt( (2.0 * rho_l * dP) / max(Ktot,1e-9) )
        # amortecimento leve
        G = 0.5*G + 0.5*Gnew
        if abs(Gnew - G)/max(G,1e-9) < tol:
            break
    # Aplica Cd_geom ao SPI "ideal" para coerência (mesma normalização usada acima)
    return Cd_geom_from_LD(G, L_over_D) * sqrt(2.0 * rho_l * dP)

def G_NHNE_from_LD(L_over_D):
    G_spi = G_SPI_from_LD(L_over_D)
    w_HEM = 1.0/(1.0 + max(kappa, 1e-9))
    return (1.0 - w_HEM)*G_spi + w_HEM*G_HEM_cap

def mdot_total_from_LD(L_over_D):
    G = G_NHNE_from_LD(L_over_D)
    return N_holes * A_hole * G

def percent_loss(LD_now, LD_ref):
    m_now = mdot_total_from_LD(LD_now)
    m_ref = mdot_total_from_LD(LD_ref)
    return 100.0 * (1.0 - m_now / max(m_ref, 1e-12)), m_now, m_ref

# ---------------- Run & print ----------------
loss_pct, m_now, m_ref = percent_loss(L_over_D_now, L_over_D_ref)

print(f"L/D now = {L_over_D_now:.3f}  | L/D_ref = {L_over_D_ref:.3f}")
print(f"mdot_NHNE(now) = {m_now:.6f} kg/s")
print(f"mdot_NHNE(ref) = {m_ref:.6f} kg/s")
print(f"% loss due to higher L/D = {loss_pct:.2f}%")
