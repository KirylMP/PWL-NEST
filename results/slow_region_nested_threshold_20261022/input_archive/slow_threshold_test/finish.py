import json,numpy as np,matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
r=json.load(open('validated_slow.json'));I=[.53,.55,.7,1.,1.12];lr=[46,35.75,22.75,17.5,17.75]
fig,ax=plt.subplots(figsize=(8,5))
ax.plot(I,lr,'ko-',label='Li–Rinzel reference')
for name in ['baseline','C','r','tilt']:
 d=r[name]['DOP853'];ax.plot(I,[v['period'] for v in d[:5]],marker='o',label=name)
ax.set(xlabel='Constant IP3 (uM)',ylabel='Late period (s)',title='Nested slow-threshold extension: independent DOP853');ax.grid(alpha=.25);ax.legend();fig.tight_layout();fig.savefig('periods.png',dpi=170)
th=r['tilt']['theta'];be=r['tilt']['beta'];ks=r['tilt']['k'];cref=.02812613673817445;scale=.9833352639826316;hr=.8371177795140701;hs=hr*(1-hr)
report=f'''# Nested slow-region hinge: FM pilot

Model: four-region calcium PWA with one hidden state r, as in `validated.json`, `augmented_results.json`. New hinge appended without changing old coefficients:

`q = max(0, theta - (u + beta*v))`, `d(u,v,r)/dt += k*q`, with `u=(C-Cref)/scale`, `v=(h-HR)/HS`. This is continuous PWA and adds exactly one new plane (the zero-amplitude starting point exactly reproduces baseline).

Best limited-search tilt candidate: `theta={th:.10g}, beta={be:.10g}, k={ks}`. In physical coordinates, threshold plane `C + ({be*scale/hs:.6g})*(h-HR) = {cref+scale*th:.7g}` uM (with Cref={cref}, HR={hr}). The activation is on the `u+beta*v < theta` side.

## Independent DOP853 late periods (seconds)

| I | LR | Baseline | C hinge | r hinge | Tilt hinge |
|---:|---:|---:|---:|---:|---:|
'''
for idx,Ix in enumerate(I):
 report+=f"| {Ix} | {lr[idx]} | "+' | '.join(str(r[n]['DOP853'][idx]['period']) for n in ['baseline','C','r','tilt'])+' |\n'
report+='\nAt I=1.13 all four had no late sustained cycles from the tested initial condition. All candidates have late oscillations at the five oscillatory inputs.\n\n'
report+='**Limits:** only limited differential evolution (C,r 14 generations; tilt 6 generations); only one starting condition (C=1.3,h=0.8,r=u) for fixed-I runs; DOP853 rtol=2e-9 atol=2e-11 max_step=.12, horizon 620 s. No continuation of bifurcations or proof of attractors; no dynamic-I or SIC/NEST tests. Max observed C=1.345 uM at I=1.12 (some trajectories above 1.3), which matches the preexisting issue and has not been fixed. The reference Li–Rinzel period numbers are from earlier experiments, not independently recomputed in this package.\n\n`search_slow.py` and `tilt_short.py` reproduce fitting; `validate_slow.py` reproduces DOP853 checks; `search_results.json` and `validated_slow.json` contain complete saved parameters and metrics.\n'
open('REPORT.md','w').write(report)
