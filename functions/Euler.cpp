#include "Euler.h"
#include "Thermo.h"
#include "Solver.h"
#include "Flow.h"
#include "Nozzle.h"

#include <cmath>   // M_PI
#include <algorithm>

namespace Integrator {

static constexpr double M_EPS = 1e-6;

[[nodiscard]] static inline double area_total(const Injector& inj) {
    return static_cast<double>(inj.N) * 0.25 * M_PI * inj.D * inj.D;
}

bool euler_step(Tank& tank,
                const Injector& inj,
                double dt,
                bool isentropic,
                StepOut& out)
{
    // Estado a montante (tanque)
    const double rho1 = tank.M / tank.V;
    const auto up = Thermo::from_rhoT(rho1, tank.T);
    if (up.p <= inj.P2 || tank.M <= M_EPS) return false;

    // Estado a jusante (garganta) a P2
    const auto dn = isentropic
        ? solve_isentropic(inj.P2, up)
        : solve_adiabatic (inj.P2, up);

    // Pv para kappa
    const double Pv1 = Thermo::Psat_from_T(tank.T);

    // Caudal NHNE
    const double Ac = area_total(inj);
    const double mdot = Flow::m_NHNE(inj.Cd, Ac,
                                     up.p, inj.P2,
                                     up.rho, dn.rho2,
                                     up.h,  dn.h2,
                                     Pv1);

    // Integração de Euler (massa e energia)
    const double M_new = tank.M - mdot * dt;
    if (M_new <= M_EPS) return false;

    const double U_old = tank.M * up.h;
    const double U_new = U_old - mdot * up.h * dt;

    tank.M = M_new;
    const double rho_new = tank.M / tank.V;
    const double h_new   = U_new / tank.M;
    tank.T = Solver::T_from_h_rho(h_new, rho_new);

    // Saída do passo
    out.t    += dt;
    out.mdot  = mdot;
    out.p     = up.p;
    out.T     = tank.T;
    out.M     = tank.M;

    return true;
}

} // namespace Integrator
