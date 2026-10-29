# FM2 rational calcium maps

Two monotone maps, shifted to Phi(0.5)=0:

- fractional: Phi(C) = C/(1+s C) - 0.5/(1+0.5 s).
- rational: Phi(C) = (1-w)(C-0.5) + w [C/(1+s C) - 0.5/(1+0.5 s)].

s=exp(log_s)>0 and w=sigmoid(logit_weight) in (0,1). The first model has one map parameter and the second has two. Both have analytic inverses. The hidden coordinate is unrestricted. Physical calcium thresholds, equilibrium transition inputs and calcium rest target remain fixed. Initial coefficients come from mixed-map candidates, but are *only starting guesses*: the map is different and those coefficients do not describe the same model.

Uses the same training targets and peak-weighted period/waveform objective as fm2_amplitude_refine, followed by independent DOP853 validation of the best candidates and previous mixed model on the same inputs. No fold or global cycle count. Failed initializations are explicitly reported. The optimizations are finite and do not prove impossibility of either model. Each process selects its family via RATIONAL_FAMILY; validating a saved candidate independently requires setting this to its coordinate_family.
