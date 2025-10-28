#pragma once

namespace Thermo {

struct State {
    double p;   // Pa
    double h;   // J/kg
    double s;   // J/kg·K
    double rho; // kg/m³
    double T;   // K
};

// State
State from_rhoT(double rho, double T);

// Saturation Pressure
double Psat_from_T(double T);


} 
