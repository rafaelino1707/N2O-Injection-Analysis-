#include "Solver.h"
#include <CoolProp.h>
#include <algorithm>
#include <cmath>
using namespace CoolProp;

namespace Solver {

static const std::string FLUID = "NitrousOxide";

static double T_lo() { return PropsSI("Tmin", FLUID.c_str()) + 1.0; }
static double T_hi() { return PropsSI("Tcrit", FLUID.c_str()) - 1.0; }

double T_from_h_rho(double h_target, double rho)
{
    double a = T_lo(), b = T_hi();
    if (rho <= 0.0) rho = 1e-8;

    auto f = [&](double T) {
        T = std::clamp(T, a, b);
        return PropsSI("H", "D", rho, "T", T, FLUID.c_str()) - h_target;
    };

    double fa = f(a), fb = f(b);
    if (std::abs(fa) < 1e-6) return a;
    if (std::abs(fb) < 1e-6) return b;

    for (int i = 0; i < 80; ++i) {
        double m = 0.5 * (a + b);
        double fm = f(m);
        if (fm == 0.0 || std::abs(b - a) < 1e-6) return m;
        if (fa * fm < 0.0) { b = m; fb = fm; } else { a = m; fa = fm; }
    }
    return 0.5 * (a + b);
}

} // namespace Solver
