#pragma once
struct PwaConfig {
    int regions;
    const double* thresholds;
    const double* coeff;
    double p,e,n,k,ref;
    bool affine;
    bool direct_calcium;
};
// State z,h,I is propagated in seconds. All functions return zero on success.
int pwa_prepare(const PwaConfig&, double I0, double tau_s, double dt_s, double sic_threshold);
int pwa_to_z(const PwaConfig&, double C, double& z);
int pwa_to_c(const PwaConfig&, double z, double& hint);
int pwa_advance_z(const PwaConfig&, double& z, double& h, double& I,
                  double I0, double tau_s, double dt_s);
int pwa_sic(const PwaConfig&, double z, double threshold, double scale,
            double& hint, double& value);
