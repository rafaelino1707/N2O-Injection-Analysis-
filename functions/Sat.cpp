#include "Sat.h"
#include "EOS.h"
#include <cmath>
#include <algorithm>

namespace Sat {

// Ex.: ln(psat/pc) = sum a_k * theta^{b_k},  theta = 1 - T/Tc
// Preencher a_k, b_k com os coeficientes publicados para N2O
static const double a[] = { /* TODO */ };
static const double b[] = { /* TODO */ };
static constexpr int    N = /* nº de termos */;

double psat_T(double T)
{
    const double Tc = EOS::Tc();
    const double pc = /* TODO: pressão crítica do N2O em Pa */;
    double theta = 1.0 - T / Tc;
    theta = std::clamp(theta, 0.0, 1.0);
    double sum = 0.0;
    for (int i = 0; i < N; ++i) sum += a[i] * std::pow(theta, b[i]);
    return pc * std::exp(sum);
}

double rhol_T(double /*T*/) { /* opcional */ return 0.0; }
double rhov_T(double /*T*/) { /* opcional */ return 0.0; }

} // namespace Sat
