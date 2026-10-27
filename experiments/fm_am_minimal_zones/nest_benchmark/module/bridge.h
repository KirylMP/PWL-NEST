#pragma once
#include <cstddef>
struct PwaConfig {
    int regions;
    const double* thresholds;
    const double* coeff;
    double p,e,n,k,ref;
    bool affine;
};
// Advances an autonomous PWA oscillator driven by exponential IP3 in seconds.
// Returns zero on success; negative status if state leaves physical domain.
int pwa_advance(const PwaConfig&, double& C, double& h, double& I,
                double I0, double tau_s, double dt_s);
