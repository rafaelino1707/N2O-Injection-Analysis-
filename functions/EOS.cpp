#include "EOS.h"
#include <algorithm>
#include <cmath>

namespace EOS {

// ======== COEFICIENTES DO N2O AQUI ========
// Preencher R, Tc, rhoc e os arrays de termos da phi^0 e phi^r
static constexpr double R_spec = /* TODO */;
static constexpr double Tc_v   = /* TODO */;
static constexpr double rhoc_v = /* TODO */;

double R()    { return R_spec; }
double Tc()   { return Tc_v; }
double rhoc() { return rhoc_v; }

// Avaliação de phi(δ,τ) e derivadas necessárias
// δ = rho/rhoc, τ = Tc/T
struct Phi {
    double phi0, phir;
    double phi0_tau, phir_tau;
    double phir_delta;
    // acrescenta segundas derivadas se precisares
};

static Phi eval_phi(double delta, double tau)
{
    Phi P{};
    // TODO: somatórios de termos ideal (phi0) e residual (phir)
    // P.phi0 = ...
    // P.phir = ...
    // P.phi0_tau = ...
    // P.phir_tau = ...
    // P.phir_delta = ...
    return P;
}

EOS::State props_rhoT(double rho, double T)
{
    // reduzir
    const double delta = rho / rhoc_v;
    const double tau   = Tc_v / T;

    const Phi P = eval_phi(delta, tau);

    // p = rho*R*T*(1 + delta*phir_delta)
    const double p = rho * R_spec * T * (1.0 + delta * P.phir_delta);

    // u = R*T * tau*(phi0_tau + phir_tau)
    const double u = R_spec * T * tau * (P.phi0_tau + P.phir_tau);

    // s = R * [ tau*(phi0_tau + phir_tau) - (phi0 + phir) ]
    const double s = R_spec * ( tau*(P.phi0_tau + P.phir_tau) - (P.phi0 + P.phir) );

    // h = u + p/rho
    const double h = u + p / rho;

    return {p, h, s};
}

EOS::Derivs derivs_rhoT(double /*rho*/, double /*T*/)
{
    // OPCIONAL: implementar se fores usar Newton robusto no nozzle
    return {};
}

} // namespace EOS
