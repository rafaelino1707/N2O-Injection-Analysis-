#pragma once
#include <stdexcept>

// Estado do tanque (volume fixo; massa e temperatura evoluem)
struct Injector {
    double D;  // m
    double N;  // Holes Number
    double Cd;   
    double P2; // Pa

};
