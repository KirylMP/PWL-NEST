"""Replay the article's shared input with and without the lower guard."""
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from fit_lower_fixed_fold import rhs
from lower_calcium_guard import add_guard
from random_selected_variable_input import SEGMENTS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'paper/data/random_selected'
manifest = json.loads((OUT/'manifest.json').read_text())['models']

def simulate(model):
    state = np.asarray(model['rest_target'], float)
    times, values, output, zero_up, minima = [], [], [], [], []
    for start, end, I in SEGMENTS:
        def spike(t, y): return y[0] - .2
        spike.direction = 1
        def zero(t, y): return y[0]
        zero.direction = 1
        def turning(t, y): return rhs(model, I, y)[0]
        turning.direction = 1
        grid = np.linspace(start, end, round((end-start)*10)+1)
        sol = solve_ivp(lambda t,y: rhs(model,I,y), (start,end), state,
                        method='DOP853', rtol=1e-10, atol=1e-12, max_step=.03,
                        t_eval=grid, events=(spike,zero,turning), dense_output=True)
        if not sol.success: raise RuntimeError(sol.message)
        times.extend(sol.t if not times else sol.t[1:]); values.extend(sol.y.T if not values else sol.y.T[1:])
        output.extend(sol.t_events[0].tolist()); zero_up.extend(sol.t_events[1].tolist())
        candidates = [start,end,*sol.t_events[2]]
        minima.extend((float(sol.sol(t)[0]),float(t)) for t in candidates)
        state=sol.y[:,-1]
    return {'times':np.asarray(times),'values':np.asarray(values),'events':output,'zero_up':zero_up,'minima':minima}

def main():
    result={}; traces={}
    for case, meta in manifest.items():
        raw=(ROOT/meta['candidate_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=meta['candidate_sha256']: raise ValueError('candidate hash mismatch')
        base=json.loads(raw); states={}
        for name,model in [('three',base),('four',add_guard(base,cut=.025,gain=100.))]:
            z=simulate(model); minimum,when=min(z['minima']); after=[x for x in z['zero_up'] if x>320 and x>when]
            eq=root(lambda y: rhs(model,.2,y),z['values'][-1]).x
            if np.linalg.norm(rhs(model,.2,eq))>1e-8: raise RuntimeError('equilibrium solve failed')
            mask=z['times']>=320
            dist=np.abs(z['values'][mask,0]-eq[0]); bad=np.flatnonzero(dist > .01*abs(eq[0]))
            recovery_1pct=(float(z['times'][mask][bad[-1]+1]-320) if len(bad) and bad[-1]+1<len(dist) else (0.0 if not len(bad) else None))
            states[name]={'equilibrium_C_uM':float(eq[0]),'recovery_to_1pct_s_sampled':recovery_1pct,'min_C_uM':minimum,'time_of_min_s':when,'negative_recovery_after_320_s': after[0]-320 if minimum<0 and after else None,
                          'event_count_total':len(z['events']),'event_times_s':z['events'],
                          'event_count_after_320':sum(320<=t<=400 for t in z['events'])}
            traces[case+'_'+name+'_C']=z['values'][:,0]
        result[case]={'sha256':meta['candidate_sha256'],'models':states}
        traces['time_s']=z['times']
        print(case,states,flush=True)
    np.savez_compressed(OUT/'guard_variable_input.npz',**traces)
    (OUT/'guard_variable_input.json').write_text(json.dumps({'protocol':SEGMENTS,'solver':'DOP853 rtol=1e-10 atol=1e-12 max_step=0.03 s','initial':'same selected PWA rest state','events':'continuous upward crossings of C=0.2 µM; not NEST tick-quantized','recovery':'zero crossing: from t=320 s to upward C=0 after a negative minimum; 1pct: first 0.1s sample after t=320 staying within 1% of equilibrium C until t=400; null if not attained','results':result},indent=2)+'\n')
if __name__ == '__main__': main()
