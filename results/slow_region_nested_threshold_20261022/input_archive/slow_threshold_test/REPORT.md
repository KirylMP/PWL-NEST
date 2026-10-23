# Nested slow-region hinge: FM pilot

Model: four-region calcium PWA with one hidden state r, as in `validated.json`, `augmented_results.json`. New hinge appended without changing old coefficients:

`q = max(0, theta - (u + beta*v))`, `d(u,v,r)/dt += k*q`, with `u=(C-Cref)/scale`, `v=(h-HR)/HS`. This is continuous PWA and adds exactly one new plane (the zero-amplitude starting point exactly reproduces baseline).

Best limited-search tilt candidate: `theta=0.01170887741, beta=-0.01106572539, k=[1.1100854297957472, -1.993109225974594, -1.0167659803442974]`. In physical coordinates, threshold plane `C + (-0.0798034)*(h-HR) = 0.03963989` uM (with Cref=0.02812613673817445, HR=0.8371177795140701). The activation is on the `u+beta*v < theta` side.

## Independent DOP853 late periods (seconds)

| I | LR | Baseline | C hinge | r hinge | Tilt hinge |
|---:|---:|---:|---:|---:|---:|
| 0.53 | 46 | 40.0 | 42.0 | 41.6 | 44.6 |
| 0.55 | 35.75 | 35.2 | 37.0 | 36.8 | 37.0 |
| 0.7 | 22.75 | 23.3 | 24.2 | 24.4 | 23.4 |
| 1.0 | 17.5 | 17.2 | 17.5 | 17.6 | 17.2 |
| 1.12 | 17.75 | 17.0 | 17.2 | 17.2 | 17.0 |

At I=1.13 all four had no late sustained cycles from the tested initial condition. All candidates have late oscillations at the five oscillatory inputs.

**Limits:** only limited differential evolution (C,r 14 generations; tilt 6 generations); only one starting condition (C=1.3,h=0.8,r=u) for fixed-I runs; DOP853 rtol=2e-9 atol=2e-11 max_step=.12, horizon 620 s. No continuation of bifurcations or proof of attractors; no dynamic-I or SIC/NEST tests. Max observed C=1.345 uM at I=1.12 (some trajectories above 1.3), which matches the preexisting issue and has not been fixed. The reference Li–Rinzel period numbers are from earlier experiments, not independently recomputed in this package.

`search_slow.py` and `tilt_short.py` reproduce fitting; `validate_slow.py` reproduces DOP853 checks; `search_results.json` and `validated_slow.json` contain complete saved parameters and metrics.
