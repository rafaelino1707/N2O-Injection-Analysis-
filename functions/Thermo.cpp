#include "Thermo.h"
#include <CoolProp.h>
#include <algorithm>
#include <cmath>

using namespace CoolProp;

namespace Thermo {

// ======== Configuração base ========

static const std::string FLUID = "NitrousOxide";

static double T_lo()
{
    return PropsSI("Tmin", FLUID.c_str()) + 1.0;
}

static double T_hi()
{
    return PropsSI("Tcrit", FLUID.c_str()) - 1.0;
}

// ======== Funções públicas ========

// Retorna estado termodinâmico geral (qualquer fase)
State from_rhoT(double rho, double T)
{
    T = std::clamp(T, T_lo(), T_hi());
    if (rho <= 0.0) rho = 1e-8;

    State s;
    s.rho = rho;
    s.T   = T;
    s.p   = PropsSI("P", "D", rho, "T", T, FLUID.c_str());
    s.h   = PropsSI("H", "D", rho, "T", T, FLUID.c_str());
    s.s   = PropsSI("S", "D", rho, "T", T, FLUID.c_str());
    return s;
}

// Pressão de saturação a T (usada para o fator kappa)
double Psat_from_T(double T)
{
    T = std::clamp(T, T_lo(), T_hi());
    try {
        return PropsSI("P", "T", T, "Q", 0, FLUID.c_str());
    } catch (...) {
        // fallback seguro
        return PropsSI("P", "T", T, "Q", 0.5, FLUID.c_str());
    }
}


} // namespace Thermo
