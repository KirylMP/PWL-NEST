"""Check unchanged fitted cycles and reset protocol for a fixed lower guard."""
import argparse
import json
from pathlib import Path
from benchmark_ca_events import load_case
from ca_reset_stress import stress
from lower_calcium_guard import add_guard

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cut',type=float,default=.025);p.add_argument('--gain',type=float,default=100.);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out={}
for case,seed in [('AM',465),('AFM',800),('FM',474)]:
    base,sha=load_case(case,'random_selected');guard=add_guard(base,a.cut,a.gain)
    report=json.loads((ROOT/f'results/random_64/{case}/seed_{seed}/fit/report.json').read_text())
    minima=[z['Cmin'] for z in report['best']['cycles']]
    minima += [z['cycle']['candidates'][0]['Cmin'] for z in report['holdout'] if z['cycle']['candidates']]
    minima += [report['best']['fold']['Cmin_sampled'],base['rest_target'][0]]
    if min(minima)<=a.cut:raise ValueError('a selected equilibrium/cycle may reach the guard zone')
    r=stress(guard,rtol=1e-10,atol=1e-12,max_step=.03)
    out[case]={'source_sha256':sha,'rest_C':base['rest_target'][0],
               'minimum_saved_C_among_checked_states':min(minima),'stress':r,
               'new_zone_trace':guard['trace'][0],'new_zone_determinant':guard['determinant'][0]}
    print(case,'smallest saved C',min(minima),'reset min C',r['min_C'],flush=True)
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(json.dumps({'cut':a.cut,'gain':a.gain,'scope':'finite saved-orbit and reset checks; no arbitrary-state positivity guarantee','results':out},indent=2,allow_nan=False)+'\n')
