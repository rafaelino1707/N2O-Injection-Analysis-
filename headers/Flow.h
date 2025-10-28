#pragma once

namespace Flow {

double m_incompressible(double Cd, double Ac, double rho1, double P1, double P2);
double m_HEM(double Cd, double Ac, double rho2, double h1, double h2);
double kappa(double P1, double P2, double Pv1);
double m_NHNE(double Cd, double Ac, double P1, double P2,
              double rho1, double rho2, double h1, double h2, double Pv1);
              
}
