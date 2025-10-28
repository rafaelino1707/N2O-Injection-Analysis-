#pragma once

namespace EOS {

// Constantes do fluido (preencher com os valores do N2O)
double R();    // J/(kg·K)
double Tc();   // K
double rhoc(); // kg/m^3

// Estado termodinâmico mínimo necessário
struct State {
    double p; // Pa
    double h; // J/kg
    double s; // J/(kg·K)
};

// Propriedades a partir de (rho,T) via Helmholtz
State props_rhoT(double rho, double T);

// === OPCIONAL: se fores usar Newton no nozzle ===
// derivadas (ex.: dp/dT|rho, dp/drho|T) para jacobianos
struct Derivs {
    double dp_dT_rho;
    double dp_drho_T;
    double dh_dT_rho;
    double dh_drho_T;
    double ds_dT_rho;
    double ds_drho_T;
};
Derivs derivs_rhoT(double rho, double T);

} // namespace EOS
