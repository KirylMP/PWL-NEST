"""Run the selected-model NEST accuracy and speed audit."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import numpy as np
import scipy
from accuracy_speed_core import ROOT, KER, load_case, reference, error_metrics, lr_rhs
from accuracy_speed_nest import diagnostic

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cases', nargs='+', choices=sorted(KER), default=['AM','AFM','FM'])
    p.add_argument('--inputs', nargs='+', type=float, default=[.2,.6,1.])
    p.add_argument('--steps-ms', nargs='+', type=float, default=[.05,.1,.2])
    p.add_argument('--interval-ms', type=float, default=50.)
    p.add_argument('--duration-s', type=float, default=300.)
    p.add_argument('--nodes', type=int, default=32)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmup', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    a=p.parse_args()
    if not (0 <= min(a.inputs) <= max(a.inputs) <= 1.3 and min(a.steps_ms)>0 and
            a.duration_s>0 and a.interval_ms>0 and min(a.nodes,a.repeats)>0 and a.warmup>=0):
        p.error('Invalid settings')
    if any(abs(value/dt-round(value/dt))>1e-7 for dt in a.steps_ms
           for value in (a.duration_s*1000.,a.interval_ms)):
        p.error('Duration and interval must be integer multiples of every step')
    import nest
    from benchmark_constant_ip3 import run
    module = ROOT / '.build/pwl_astro_module.so'
    if not module.is_file(): raise FileNotFoundError(module)
    a.output.mkdir(parents=True,exist_ok=False)
    meta={'nest_version':nest.__version__, 'numpy':np.__version__, 'scipy':scipy.__version__,
          'platform':platform.platform(),'processor':platform.processor(),
          'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),
          'settings':vars(a)|{'output':str(a.output)},
          'scope':'constant IP3; each model compared to own ODE; no network; one thread',
          'reference':'DOP853; fine rtol=2e-12 atol=2e-14 max_step=0.025 s; coarse rtol=1e-10 atol=1e-12 max_step=0.05 s',
          'warning':'LR GSL absolute tolerance hardcoded 1e-3 in NEST v3.10; varying resolution is not a tolerance sweep; endpoint-only PWA threshold detection may miss leave-return excursions'}
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    failures=0
    with (a.output/'runs.jsonl').open('x') as f:
        for case in a.cases:
            data, sha=load_case(case)
            for I in a.inputs:
                for dt in a.steps_ms:
                    for kind in ('LR','PWA'):
                        record={'case':case,'I':I,'dt_ms':dt,'model':kind,'candidate_sha256':sha}
                        try:
                            t, actual, iperr=diagnostic(case,kind,data,I,dt,a.interval_ms,a.duration_s)
                            initial=np.array(data['rest_target'],float)
                            fine=reference(case,kind,data,I,initial,t,fine=True)
                            coarse=reference(case,kind,data,I,initial,t,fine=False)
                            record.update(samples=len(t),first_time_s=float(t[0]),last_time_s=float(t[-1]),
                                          error=error_metrics(actual,fine),
                                          reference_disagreement=error_metrics(coarse,fine),ip3_max_abs=iperr,
                                          reference_C_range=[float(np.min(fine[0])),float(np.max(fine[0]))],
                                          observed_C_range=[float(np.min(actual[0])),float(np.max(actual[0]))],
                                          reference_boundary_free=(kind != 'LR' or bool(np.min(fine[0])>0 and np.max(fine[0])<2)))
                        except Exception as e:
                            record['error_diagnostic']=f'{type(e).__name__}: {e}'; failures+=1
                        f.write(json.dumps(record,allow_nan=False)+'\n');f.flush()
                        print(case,I,dt,kind,record.get('error'),record.get('error_diagnostic',''),flush=True)
                    # Use the same worker and no recording in timing, alternating order.
                    args=argparse.Namespace(dt_ms=dt,threads=1,nodes=a.nodes,duration_s=a.duration_s,
                                            interval_ms=a.interval_ms,transient_s=0.)
                    for n in range(-a.warmup,a.repeats):
                        for kind in (('LR','PWA') if n%2==0 else ('PWA','LR')):
                            rec={'case':case,'I':I,'dt_ms':dt,'model':kind,'repeat':n,
                                 'warmup':n<0,'candidate_sha256':sha}
                            try: rec['timing']=run(case,kind,data,I,args)
                            except Exception as e:
                                rec['error_timing']=f'{type(e).__name__}: {e}';failures+=1
                            f.write(json.dumps(rec,allow_nan=False)+'\n');f.flush()
    print('Saved',a.output,'failures',failures,flush=True)
    if failures: raise SystemExit(1)


if __name__=='__main__':main()
