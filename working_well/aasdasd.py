from CoolProp.CoolProp import PropsSI

FLUID = "NitrousOxide"
T = 273.15 + 2  # K
P = 40e5        # Pa

rho = PropsSI("D", "T", T, "P", P, FLUID)
print("ρ =", rho, "kg/m³")
