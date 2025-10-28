#include <iostream>
#include <iomanip>
#include "Tank.h"
#include "Injector.h"
#include "Euler.h"

int main() {
    // ===================== CONFIGURAÇÃO INICIAL =====================
    Tank tank{
        0.0499,   // V [m³]
        12.76,    // M0 [kg]
        274.25    // T0 [K]
    };

    Injector inj{
        6,              // N = nº de furos
        0.178 * 0.0254, // D = 0.178" -> m
        0.80,           // Cd
        101325.0        // P2 = pressão jusante [Pa]
    };

    const double dt = 0.05;    // passo de integração [s]
    const double t_stop = 30.0; // tempo máximo [s]
    const bool isentropic = true; // true → modelo isentrópico, false → adiabático

    // ===================== VARIÁVEIS DE SIMULAÇÃO =====================
    Integrator::StepOut step{0, 0, 0, tank.T, tank.M};

    // Cabeçalho
    std::cout << std::fixed << std::setprecision(4);
    std::cout << " t[s]\tP_tank[MPa]\tM[kg]\tT[K]\tmdot[kg/s]\n";

    // ===================== LOOP DE INTEGRAÇÃO =====================
    while (step.t < t_stop) {
        bool ok = Integrator::euler_step(tank, inj, dt, isentropic, step);
        if (!ok) break;

        std::cout << step.t << "\t"
                  << step.p / 1e6 << "\t"
                  << step.M << "\t"
                  << step.T << "\t"
                  << step.mdot << "\n";
    }

    // ===================== RESULTADOS FINAIS =====================
    std::cout << "\nFinal tank pressure: " << step.p / 1e6 << " MPa\n";
    std::cout << "Remaining mass: " << step.M << " kg\n";
    std::cout << "Final temperature: " << step.T << " K\n";
}
