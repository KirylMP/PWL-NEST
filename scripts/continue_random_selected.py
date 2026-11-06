"""Continue the four-segment neutral orbits; finite numerical portions only."""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from continue_physical_fixed import Orbit, continue_branch

ROOT = Path(__file__).resolve().parents[1]
def task(task):
    case, direction, steps = task
    path = ROOT/'paper/data/random_selected'/case
    d = json.loads((path/'candidate.json').read_text())
    r = json.loads((path/'report.json').read_text())
    fold = r['best']['fold']; y = np.array([*fold['times'], fold['I']]); orb = Orbit(d)
    try:
        branch = continue_branch(orb, y, direction, steps, require_physical=False, max_I=1.3, initial_step=.006, max_step=.015)
        for p in branch['points']:
            p.pop('physical',None)
            p['sampled_positive_C'] = bool(p['Cmin']>0)
        return {'case':case,'direction':direction,'branch':branch}
    except (ValueError, RuntimeError, OverflowError, np.linalg.LinAlgError) as e:
        return {'case':case,'direction':direction,'error':str(e)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--steps',type=int,default=120);args=parser.parse_args()
    if args.steps<1:parser.error('--steps must be positive')
    tasks = [(c,d,args.steps) for c in ('AM','AFM','FM') for d in (-1,1)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        rows=list(pool.map(task,tasks))
    out=ROOT/'paper/data/random_selected/continued_both.json'
    out.write_text(json.dumps({'scope':'finite continuation of four-segment itinerary; no completeness claim','results':rows},indent=2,allow_nan=False)+'\n')
    for row in rows: print(row['case'],row['direction'],len(row.get('branch',{}).get('points',[])),row.get('error',row.get('branch',{}).get('reason')),flush=True)
if __name__=='__main__':main()
