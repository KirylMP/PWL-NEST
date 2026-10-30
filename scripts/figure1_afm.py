"""Extend the existing figure 1 caches with K_ER=0.075; preserve AM/FM data."""
import csv,json
import numpy as np
from scipy.integrate import solve_ivp
import paper_figure1 as f

def main():
    case='AFM'; eq, events=f.equilibria(case)
    np.savez_compressed(f.DATA/'AFM_equilibria.npz',**eq)
    hopfs=sorted((e for e in events if e['kind']=='trace_zero'),key=lambda e:e['I'])
    if len(hopfs)!=2: raise RuntimeError(f'Expected two Hopf candidates: {hopfs}')
    sols=f.continue_cycles(case,hopfs[-1])
    rows,orbits=zip(*(f.cycle_metrics(case,s) for s in sols))
    fold,fold_orbit=f.refine_cycle_fold(case,sols,rows)
    np.savez_compressed(f.DATA/'AFM_upper_cycles.npz',phase=f.PHASE,orbits=np.array(orbits))
    np.savez_compressed(f.DATA/'AFM_cycle_fold.npz',phase=f.PHASE,orbit=fold_orbit)
    previous=f.load_rows()
    with (f.DATA/'cycles.csv').open(newline='') as stream: original=list(csv.DictReader(stream))
    original=[r for r in original if r['case']!='AFM']
    with (f.DATA/'cycles.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(original[0]));writer.writeheader();writer.writerows(original)
        writer.writerows(dict(case=case,branch='upper',index=k,**row) for k,row in enumerate(rows))
    ip3=.5; ode=f.rhs(case,ip3)
    tr=solve_ivp(ode,(0,900),[.7,.8],method='DOP853',rtol=2e-11,atol=2e-13,max_step=.15)
    def peak(t,y): return ode(t,y)[0]
    peak.direction=-1
    sol=solve_ivp(ode,(0,150),tr.y[:,-1],events=peak,method='DOP853',rtol=2e-11,atol=2e-13,max_step=.06,dense_output=True)
    if len(sol.t_events[0])<2: raise RuntimeError('No stable AFM phase portrait cycle')
    t0,t1=sol.t_events[0][-2:];orbit=sol.sol(t0+f.PHASE*(t1-t0))
    np.savez_compressed(f.DATA/'AFM_phase_plane.npz',phase=f.PHASE,orbit=orbit,I=ip3,period_s=t1-t0)
    meta=json.loads((f.DATA/'metadata.json').read_text());meta['parameters']['KER_AFM']=.075
    meta['events']=[e for e in meta['events'] if e['case']!='AFM']+[dict(case=case,**e) for e in events if e['kind'] in ('trace_zero','fold')]+[fold]
    shown=[r for r in rows if r['displayed']]
    meta['validation']['AFM_upper']=dict(computed=len(rows),displayed=len(shown),max_displayed_closure=max(r['ivp_closure'] for r in shown),max_displayed_waveform_error=max(r['ivp_waveform_error'] for r in shown),max_bvp_residual=max(r['bvp_residual'] for r in rows))
    (f.DATA/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    f.plot()

if __name__=='__main__': main()
