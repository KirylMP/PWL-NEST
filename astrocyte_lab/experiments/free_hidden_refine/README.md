# Refine no-fold models with unrestricted hidden q

Seed from a previous `no_fold_refine` run. The hidden coordinate q is allowed to take any finite real value; there is no q range penalty in the optimizer, no q bounds in the RK4 engine, and no q bounds in the independent DOP853 validation. Physical calcium remains in its configured interval; numerical finiteness is required. Rest calcium, both physical thresholds and equilibrium crossings remain fixed. The value q at I=0 continues to use the previous convention to set the hidden coordinate origin. This normalization is not a bound on q along trajectories. Fold and unstable-cycle constraints and analyses are absent.

Each top candidate from the previous run is refined by Powell followed by finite-difference least-squares, using stable-cycle waveform and period training loss. The best finite evaluation with all training cycles is retained. Both the new winner and the previous winner are revalidated with the same unrestricted-q DOP853 procedure. `refined_vs_previous.json` compares holdout and training errors. q minima/maxima on the phase-sampled cycle appear in validation.json.

These models are not yet benchmarked in NEST. Existing NEST code and assumptions about h_IP3R need separate review for hidden q outside 0–1. Cycle enumeration, basin coverage and stability of every possible trajectory are not established.
