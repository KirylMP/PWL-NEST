"""Paired 3-/4-zone NEST benchmark with calcium-spike output and no SIC."""
import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from benchmark_ca_events import load_case
from lower_calcium_guard import add_guard

ROOT=Path(__file__).resolve().parents[1]
def field(d,I,y):
    C,q=y;v=np.r_[1.,C,I,np.maximum(C-np.asarray(d['physical_thresholds_C']),0.)]
    return [d['a']*(q-np.dot(d['alpha'],v)),d['b']*(np.dot(d['beta'],v)-q)]

def reset_start(d):
    state=np.asarray(d['rest_target'],float)
    for I in (.2,1.2):
        s=solve_ivp(lambda t,y:field(d,I,y),(0.,80.),state,method='DOP853',rtol=1e-10,atol=1e-12,max_step=.05)
        if not s.success:raise RuntimeError('cannot prepare reset initial state')
        state=s.y[:,-1]
    return state.tolist()

def encode(d,I,C0,q0,threshold):
    N,H=np.array(d['alpha']),np.array(d['beta']);cuts=np.array(d['physical_thresholds_C'])
    a,b,cref,href=float(d['a']),float(d['b']),1e-6,.5;scale=href*(1-href);rows=[]
    for j in range(len(cuts)+1):
        n=N[1]+N[3:3+j].sum();h=H[1]+H[3:3+j].sum()
        nc=N[0]-N[3:3+j]@cuts[:j];hc=H[0]-H[3:3+j]@cuts[:j]
        rows.extend([a*(href-nc-n*cref),-a*n,a*scale,-a*N[2],
                     b*(hc+h*cref-href)/scale,b*h/scale,-b,b*H[2]/scale])
    return dict(thresholds=(cuts-cref).tolist(),coefficients=rows,input_thresholds=[],
                c_coordinate=0,c_reference=cref,c_scale=1.,h_coordinate=0,h_reference=href,
                u=float(C0-cref),v=float((q0-href)/scale),IP3=I,IP3_0=I,
                tau_IP3=20000.,delta_IP3=0.,ca_event_threshold=threshold)

def run(d,I,initial,a,diagnostic):
    import nest
    nest.ResetKernel();nest.SetKernelStatus({'resolution':a.dt_ms,'local_num_threads':1,'print_time':False})
    if 'astrocyte_pwa_guard_bench' not in nest.node_models:nest.Install(str(ROOT/'.build/pwa_guard_bench_module'))
    t=time.perf_counter();nodes=nest.Create('astrocyte_pwa_guard_bench',n=1 if diagnostic else a.nodes,params=encode(d,I,*initial,a.spike_threshold))
    rec=nest.Create('spike_recorder');nest.Connect(nodes,rec);meter=None
    if diagnostic:
        meter=nest.Create('multimeter',params={'interval':a.interval_ms,'record_from':['Ca_astro','IP3']});nest.Connect(meter,nodes)
    build=time.perf_counter()-t;t=time.perf_counter();nest.Prepare();prepare=time.perf_counter()-t
    try:
        t=time.perf_counter();nest.Run(a.duration_s*1000.);elapsed=time.perf_counter()-t
    finally:
        t=time.perf_counter();nest.Cleanup();cleanup=time.perf_counter()-t
    result={'build_s':build,'prepare_s':prepare,'run_s':elapsed,'cleanup_s':cleanup,
            'guard_crossings':int(np.sum(nodes.get('guard_crossings'))),
            'all_zone_crossings':int(np.sum(nodes.get('region_crossings'))),
            'affine_flows':int(np.sum(nodes.get('affine_flows'))),
            'events_emitted':int(np.sum(nodes.get('ca_event_count'))),'events_recorded':int(rec.get('n_events'))}
    if diagnostic:
        ev=meter.get('events');C=np.asarray(ev['Ca_astro']);t=np.asarray(ev['times'])/1000.
        result['min_C']=float(C.min());result['max_C']=float(C.max());result['last_C']=float(C[-1]);
        result['trace']={'time_s':t.tolist(),'C':C.tolist()}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',nargs='+',choices=['AM','AFM','FM'],default=['AM','AFM','FM'])
    p.add_argument('--inputs',nargs='+',type=float,default=[.2,.35,.45,.55,.65,.75,.9]);p.add_argument('--cut',type=float,default=.025);p.add_argument('--gain',type=float,default=100.)
    p.add_argument('--spike-threshold',type=float,default=.2);p.add_argument('--nodes',type=int,default=32);p.add_argument('--duration-s',type=float,default=120.)
    p.add_argument('--dt-ms',type=float,default=.1);p.add_argument('--interval-ms',type=float,default=50.);p.add_argument('--repeats',type=int,default=3);p.add_argument('--warmup',type=int,default=1)
    p.add_argument('--skip-reset',action='store_true');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if min(a.nodes,a.repeats)<1 or a.warmup<0 or min(a.inputs)<0 or max(a.inputs)>1.3 or min(a.cut,a.gain,a.dt_ms,a.duration_s,a.interval_ms)<=0:p.error('invalid settings')
    for x in (a.duration_s*1000,a.interval_ms):
        if abs(x/a.dt_ms-round(x/a.dt_ms))>1e-7:p.error('duration and recording interval must align with NEST step')
    import nest
    module=ROOT/'.build/pwa_guard_bench_module.so'
    if not module.is_file():raise RuntimeError('run scripts/build_guard_benchmark.py')
    a.output.mkdir(parents=True,exist_ok=True);output=a.output/'runs.jsonl'
    if output.exists():raise FileExistsError('choose a new output directory')
    metadata={'nest_version':nest.__version__,'platform':platform.platform(),'processor':platform.processor(),
              'module_sha256':hashlib.sha256(module.read_bytes()).hexdigest(),
              'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
              'scope':'same binary, same node type, physical-coordinate PWA, spike recorder; no LR, no SIC; finite observations',
              'caveats':['guard not globally positivity-preserving','time includes event recorder and counters','reset state from unguarded original model']}
    (a.output/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
    with output.open('x') as f:
        for case in a.cases:
            base,sha=load_case(case,'random_selected');guard=add_guard(base,a.cut,a.gain)
            conditions=[('constant',float(I),base['rest_target']) for I in a.inputs]
            if not a.skip_reset:conditions.append(('reset_from_1.2',0.,reset_start(base)))
            for kind,I,initial in conditions:
                models={'three':base,'four':guard};diagnostics={}
                for name,d in models.items():diagnostics[name]=run(d,I,initial,a,True)
                for rep in range(-a.warmup,a.repeats):
                    for name in (('three','four') if rep%2==0 else ('four','three')):
                        result=run(models[name],I,initial,a,False)
                        item={'case':case,'kind':kind,'I':I,'initial':initial,'model':name,'repeat':rep,'warmup':rep<0,
                              'source_sha256':sha,'result':result,'diagnostic':diagnostics[name] if rep==0 else None}
                        f.write(json.dumps(item,allow_nan=False)+'\n');f.flush()
                print(case,kind,I,'guarded diagnostic crossings',diagnostics['four']['guard_crossings'],
                      'min C:',diagnostics['three']['min_C'],diagnostics['four']['min_C'],flush=True)
    print('Saved',output)
if __name__=='__main__':main()
