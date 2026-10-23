import json,os,numpy as np,matplotlib.pyplot as plt,zipfile
root='/mnt/data/riccati_slow_pilot';d=json.load(open(root+'/results.json'))
I=np.array([x['I'] for x in d['baseline'][:5]]);base=np.array([x['period'] for x in d['baseline'][:5]]);new=np.array([x['period'] for x in d['DOP853'][:5]]);lr=np.array([46,35.75,22.75,17.5,17.75])
fig,ax=plt.subplots(figsize=(8,4.6));ax.plot(I,lr,'o-',label='Li–Rinzel reference');ax.plot(I,base,'s--',label='4-region PWA baseline');ax.plot(I,new,'d-',label='modal Riccati correction (DOP853)');ax.set(xlabel='Constant IP3 (µM)',ylabel='Late-cycle period (s)',title='FM period: one quadratic slow-mode correction');ax.legend();ax.grid(alpha=.25);fig.tight_layout();fig.savefig(root+'/periods.png',dpi=160);plt.close(fig)
report=f'''# Modal Riccati correction: reproducible FM pilot

Base: four-region PWA with one hidden state, files from `slow_threshold_test`.
No changes to its existing 3 calcium thresholds, hinge coefficients, or 5 hidden-feedback coefficients.

## Model

Let `y=(u,v,r)`, where `u=(C-Cref)/S`, `v=(h-HR)/HS`. The base four-region vector field is `F0(y,I)`. The low-C Jacobian has a real slow eigenvalue `lambda={d['slow_eigenvalue']:.10g} s^-1`, left eigenvector `l={np.round(d['slow_left'],8)}`, right eigenvector `vslow={np.round(d['slow_right'],8)}` with `l @ vslow=1`.

We add **one** quadratic hinge:

`F(y,I)=F0(y,I)+k*vslow*max(0,l@y-theta)**2`

`k={d['k']:.10f}, theta={d['theta']:.10f}`.

Two fitted parameters; no extra state, no extra linear PWA region (but one nonlinear switching surface). Continuous and C1 across the new surface, provided the baseline is continuous.

When `u` is strictly in the first original C-region and `s=l@y>theta`, `s'=a+b*s+c*s²` with `a=l@b(I)+k*theta²`, `b=lambda-2*k*theta`, `c=k`. Orthogonal modes evolve linearly: `p'=A0*p+(Id-vslow*l) b(I)`. Thus this **part of the regional flow** has an exact formula via two small matrix exponentials. When `s<=theta`, it returns to the affine flow. **The entire switched system is not one globally elementary flow**: original C-thresholds and the new s-threshold require event detection. The modal Riccati formula applies only up to the earliest crossing.

## Constant I, late periods, seconds

| I | LR target | Base 4-region PWA | Modal quadratic, independently DOP853 | Ca minimum | Ca maximum |
|---|---:|---:|---:|---:|---:|
'''
for a,b,c,t in zip(I,base,new,d['DOP853']):report+=f"| {a:.2f} | {lr[list(I).index(a)]:.2f} | {b:.2f} | {c:.2f} | {t['cmin']:.4f} | {t['cmax']:.4f} |\n"
report+='\nAt I=1.13 the tested trajectory converges to rest rather than sustaining a late large cycle. 620s runs, DOP853 rtol=2e-9, atol=2e-11, max_step=.12, from (C,h,r)=(1.3,.8,u0).\n\n## Activation fraction of late observations\n\n| I | Active % | Active outside first C-region % | In first C-region % |\n|---|---:|---:|---:|\n'
for a in d['activation_statistics']:report+=f"| {a['I']:.2f} | {100*a['active_fraction']:.1f} | {100*a['active_outside_low_region_fraction']:.1f} | {100*a['slow_region_fraction']:.1f} |\n"
report+='''
## Formula check

`check_exact.py` compares exact Riccati + transverse affine propagation with DOP853 over short intervals. Errors stay around 1e-15 **if both the original C-region and the active/inactive quadratic regime remain unchanged**. When the C-threshold is crossed, applying the original region formula beyond the event gives an error; this is expected. This is not a complete validated event-driven implementation or performance benchmark.

## Limitations

- The calcium maximum reaches about 1.345 µM in high-IP3 trajectories, above the prior 1.3-µM guard. This is an inherited failure, not resolved here.
- No independently optimized peak shape, dynamic-IP3 impulses, SIC/NEST speed benchmark, bifurcation continuation, multistart basin study, or global proof of the limit cycle.
- k and theta fitted using the five listed constant-IP3 periods and a rest requirement at I=1.13; these are *training*, not held-out measurements.
- The quadratic term is globally applied when s>theta, including short excursions outside the first C-region (~4-5% of late time for I=.53,.55); **there the simple first-region Riccati decomposition does not apply**. One can integrate as nonlinear ODE or construct a new compatible regional exact solver; no global cheap exact flow is claimed.

Files: `experiment.py` (fit, RK4 and independent DOP853), `check_exact.py` (local formula check), `results.json`, `periods.png`, `run.log`, and a copy of `validated.json`, `augmented_results.json` and `search_slow.py`/`run.py` from the baseline.
'''
open(root+'/REPORT.md','w').write(report)
import shutil
for name in ['validated.json','augmented_results.json','search_slow.py','run.py']:
 shutil.copy('/mnt/data/slow_threshold_test/'+name,root+'/'+name)
with zipfile.ZipFile('/mnt/data/riccati_slow_pilot.zip','w',zipfile.ZIP_DEFLATED) as z:
 for name in ['experiment.py','check_exact.py','finalize.py','REPORT.md','periods.png','results.json','run.log','validated.json','augmented_results.json','search_slow.py','run.py']:z.write(root+'/'+name,arcname=name)
print('done',os.stat('/mnt/data/riccati_slow_pilot.zip').st_size)
