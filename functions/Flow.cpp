#include "Flow.h"
#include <algorithm>
#include <cmath>

namespace Flow {

static constexpr double EPS = 1e-6;

double m_incompressible(double Cd, double Ac,
                        double rho1, double P1, double P2)
{
    const double dP = std::max(P1 - P2, 0.0);
    if (dP <= 0.0 || Cd <= 0.0 || Ac <= 0.0 || rho1 <= 0.0) return 0.0;
    return Cd * Ac * std::sqrt(2.0 * rho1 * dP);
}

double m_HEM(double Cd, double Ac,
             double rho2, double h1, double h2)
{
    const double dh = std::max(h1 - h2, 0.0);
    if (dh <= 0.0 || Cd <= 0.0 || Ac <= 0.0 || rho2 <= 0.0) return 0.0;
    return Cd * Ac * rho2 * std::sqrt(2.0 * dh);
}

double kappa(double P1, double P2, double Pv1)
{
    const double num = std::max(P1 - P2, 0.0);
    const double den = std::max(Pv1 - P2, EPS);
    if (den <= 0.0) return 0.0;
    return std::sqrt(num / den);
}

double m_NHNE(double Cd, double Ac, double P1, double P2,
              double rho1, double rho2, double h1, double h2, double Pv1)
{
    const double k    = kappa(P1, P2, Pv1);
    const double wHEM = 1.0 / (1.0 + k);
    const double wINC = 1.0 - wHEM;

    const double mINC = m_incompressible(Cd, Ac, rho1, P1, P2);
    const double mHEMv= m_HEM(Cd, Ac, rho2, h1, h2);

    return wINC * mINC + wHEM * mHEMv;
}

} 
