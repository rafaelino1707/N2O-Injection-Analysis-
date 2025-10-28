#pragma once
#include "Tank.h"
#include "Injector.h"

namespace Integrator {

// Saída de um passo
struct StepOut {
    double t;     // s
    double mdot;  // kg/s
    double p;     // Pa (pressão no tanque no início do passo)
    double T;     // K  (temperatura do tanque após o passo)
    double M;     // kg (massa do tanque após o passo)
};

// Um passo de Euler explícito.
// Retorna false quando o escoamento deve parar (p <= P2 ou M ~ 0).
[[nodiscard]] bool euler_step(Tank& tank,
                              const Injector& inj,
                              double dt,
                              bool isentropic,
                              StepOut& out);

} // namespace Integrator
