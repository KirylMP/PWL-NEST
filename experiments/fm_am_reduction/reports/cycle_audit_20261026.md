# FM cycle audit, 2026-10-26

Inputs: `latest_candidates_check.zip`, `selected2_check/selected.json` and
`candidate4_check/candidate.json`. No model coefficients were changed or fitted.
The original run configuration was not included; counter reconstruction uses
the saved v3 defaults. This is separate from direct orbit closure verification.

## Confirmed validation defect

For LR at I=87/81, the old forward-only shooting accepted a supposed repelling
cycle with normalized closing residual |G| approximately .01. A scalar bracket
of width 1e-8 in u does not guarantee a closed orbit when the return multiplier
is millions. Different sections produced different approximate periods/extrema,
and these approximations could be counted as different cycles.

At the source equilibrium section, corrected contracting-time shooting gives:

| LR orbit | Period (s) | Forward multiplier | Closing residual in h |
| --- | ---: | ---: | ---: |
| stable | 17.2325120647 | 1.7274e-18 | 1.11e-16 |
| unstable | 14.9288345812 | 3.1437e6 | 1.11e-14 |

The unstable orbit is computed backward; its multiplier is converted to forward
time. An independent RK45 return check passes. The validation now requires
closure, compares common oriented intersections when merging orbits, keeps
stability types separate, and includes candidate equilibrium sections.

## Independent checks of selected (2)

RK45 integration from the supplied orbit seeds, using our physical PWA field
and coordinate transform, confirms the following closed orbits. This checks
the supplied orbits; by itself it does not establish completeness.

| I | Stability | Period (s) | Absolute closing residual in h |
| --- | --- | ---: | ---: |
| .78 | stable, inner | 11.8560864372 | 3.77e-11 |
| .78 | stable, outer | 17.7100457227 | 9.78e-11 |
| .78 | unstable | 16.5368008846 | 2.37e-11 |
| .8024691358 | stable | 17.6841709131 | 9.58e-11 |
| .8024691358 | unstable | 18.5369113097 | 1.89e-10 |

The corrected multi-section scan independently finds the following counts on
65 section nodes, max return time 120 s, rtol=3e-10, atol=3e-12, max step .25 s:

| I | LR (stable, unstable) | selected (2) | candidate (4) |
| --- | --- | --- | --- |
| .78 | (1,0) | (2,1) | (0,0) |
| .8024691358 | (1,0) | (1,1) | (0,0) |
| 87/81 | (1,1) | (0,0) | (0,0) |

All six model/I comparisons have zero unresolved root intervals. All observed
cycles also cross the source training section in these examples. The added
candidate equilibrium sections improve the general coverage policy; they are
not the cause of the counter discrepancy established here. These finite-grid
observations do not establish global completeness or replace an 81-I scan.

## Counter and training grid

The implemented Gaussian variation counter and softplus extra-cycle penalty
match protocol section 5.4. Existing derivative and normalization checks pass.
The six original I nodes are .55, .65, .8, 1, 1.1, 1.14. None lies in the
reported three-cycle interval (.742, .798).

At I=.75, selected (2), epsilon=.02:

| Section grid | Smoothed stable count | Smoothed unstable count | Weighted extra-cycle contribution |
| --- | ---: | ---: | ---: |
| original 17 nodes | 2.000000570 | 1.000000570 | 30.00006835 |
| refined union, 81 nodes | 2.000003361 | 1.000003361 | 30.00040333 |

This point would be penalized on either section grid. It has zero contribution
to the original objective because it is absent from the original I grid.
More restarts on the same grid cannot fix that coverage omission.

There is a second, independently confirmed discretization problem. At I=.8,
the original 17-u grid reports (.5163311,.5163311). On a refined grid restricted
to the **same** u interval [.001,.98], the counter reports (1.0000026,1.0000026).
The largest sampled positive G rises from .0008189 on the coarse grid to .12116
on the refinement. The coarse grid missed the return map's positive maximum;
this is a resolution issue, not an algebraic error in the variation formula.

Candidate (4) has the same issue at I=.75: coarse counts are
(1.25677398,.25677398), while refinement gives (1.99952020,.99952020).
The weighted extra-cycle contribution changes from approximately 9e-24 to
29.94245160 at this point. Its omission in I and poor u resolution can both hide
the extra-cycle regime.

The small saved extra-cycle loss .0115318766 is also consistent with the
reference softplus subtraction: at I=1 and 1.1 LR has two cycles, the candidate
has none, and 2*30*(.02*ln(2))^2=.0115308723 before the finite-xi background.
Thus this positive value need not indicate that a third cycle was detected.

Reconstruction on the original six-I/17-u grid reproduces the saved selected
loss components, independently of the archived scalar values:

| Component | Saved | Recomputed |
| --- | ---: | ---: |
| weighted cycle error | 22.5026699849817 | 22.5026699849176 |
| weighted extra-cycle error | .0115318766050650 | .0115318766050650 |

Refining only u on the original six I values changes these to 25.0000290464
and .01730431391. These remain sampled errors on six I values and still omit
the interior three-cycle band. Candidate (4) is inadmissible under the same
return-coverage rule at I=1.14; its missing returns are not scored as zero cycles.

New v4 FM runs default to 65 u nodes and add I=.75,.78. This multiplies the
number of return launches by approximately five compared with v3. The
unaveraged extra-cycle term is explicitly summed over eight I nodes, with the
same coefficient 30. Existing models and historical scores remain unchanged.
New full runs validate on 129 section nodes, independently of the 65 training
nodes. Further candidate-dependent grid refinement may still be necessary.

The refined grid has a different finite-xi positive background; the LR counts
are recomputed on the same grid and are not rounded. The extra-cycle term is
summed without division by the number of I nodes, as required by the protocol.
Adding I nodes therefore changes its effective balance and must be recorded.

## Reproduction

`scripts/audit_cycle_counts.py` saves both grids' raw maps, all configured
epsilons, coverage checks against observed closed orbits, original-grid and
refined-grid score summaries, logs and figures. Supply the original experiment's
`protocol.json` when available; otherwise the audit marks the default assumption.
It does not refit or select candidates. See the experiment README for commands.

All cycle searches use finite grids. Unresolved intervals and degenerate roots
must be inspected; these checks do not prove global cycle completeness.
