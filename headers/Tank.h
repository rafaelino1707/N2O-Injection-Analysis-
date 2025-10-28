#pragma once
#include <stdexcept>

// Tank State
struct Tank {
    double V;  // m^3
    double M;  // kg
    double T;  // K  

    Tank(double V_, double M0, double T0) : V(V_), M(M0), T(T0) {
        if (V <= 0.0) throw std::invalid_argument("Tank.V must be > 0");
        if (M <  0.0) throw std::invalid_argument("Tank.M must be >= 0");
        if (T <= 0.0) throw std::invalid_argument("Tank.T must be > 0");
    }
};
