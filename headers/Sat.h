#pragma once

namespace Sat {
// p_sat(T) para o N2O (Wagner-like)
double psat_T(double T);
// OPCIONAL: densidades em saturação se precisares
double rhol_T(double T);
double rhov_T(double T);
} // namespace Sat
