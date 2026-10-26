# FM v5 source calibration: adjacent near-zero interval

The supplied failed `epsilon_calibration.json` contains 46 training I values.
Only I=.580960283961 fails. Its counter error is about 6.3e-6 at the first
profile setting and 1.27e-5 at the second, below the .025 tolerance throughout.
The blocker is an unresolved orbit-scan warning, not smoothing error. I=1.14
is simply the last source-inventory progress line.

Local reproduction found one stable LR orbit at u=.6600228039552, with period
30.4322583543 s. The adjacent interval [.65,.66] was flagged
`near_zero_without_bracket` because G(.66)=3.45514e-5. The root lies just across
that grid node, inside the next interval. The old code cleared warnings only
when their interval contained a root. Repeated uniform refinement could retain
the same grid node and the same adjacent warning.

The correction keeps an adjacent warning unresolved unless a neighboring closed
orbit is confirmed in contracting integration time. Three additional interior
returns must have the original displacement sign, remain clear of the numerical
closure tolerance, and all five sampled multipliers must be below 1-1e-4 in
that time direction. Failed returns, sign changes and expanding or near-neutral
samples retain the warning. Successful checks are saved with their actual
samples as `resolved_near_zero_intervals` in the source inventory and raw scans.
This is an explicit finite-grid check, not an interval proof of monotonicity or
global orbit completeness.

Calibration now distinguishes unresolved orbit/domain scans from epsilon
convergence. It refines unresolved profiles without repeating the epsilon loop;
it still fails if the scan remains unresolved. Reports include actual counts,
warning details and record counts. The final epsilon field no longer reports an
untested value after the last failed iteration. Source-cache signatures change
to prevent reuse of inventories made with the old warning policy.

The scientific losses, weights and numerical acceptance tolerance are unchanged.
Tests cover the exact failing LR point, the adjacent-root case, retained warnings
for sign changes/expansion/failed returns, and failure of genuinely unresolved
calibration. A full recalibration uses the same 46 I values from the supplied
failure report.

## Verification

All 42 tests passed. The direct regression uses the exact failing I; additional
mocked cases check both integration directions and retain warnings on failed
returns, sign changes and expanding samples.

Full recalibration on the supplied 46-point grid passed on the first attempt,
at epsilon .001 with 65 profile nodes and unchanged count tolerance .025. The
largest count/convergence error across the grid was .0083267041. At the failing
point, the observed counts were [1,0], smoothed counts were
[1.0000031000,.0000031000], and no unresolved warning remained. The confirmed
neighboring root was u=.6600228039700 with period 30.4322583323 s.

The 65-node source scan resolves interval [.64,.66], retaining its five checked
points and log multipliers (approximately -118). The former automatic 129-node
retry split that into the falsely flagged [.65,.66] interval. The source profile
no longer needs that retry at this I. This verification covers source
initialization; it does not replace a new full optimization of the candidate.
