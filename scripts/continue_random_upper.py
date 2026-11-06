"""Find lower grazing and continue the upper-only itinerary for selected fields."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from physical_fold_routes import Route
from continue_upper_only_selected import grazing,mid_minimum,continuation

ROOT=Path(__file__).resolve().parents[1]
def task(args):
    case,steps=args
    p=ROOT/'paper/data/random_selected'/case;model=json.loads((p/'candidate.json').read_text());fold=json.loads((p/'report.json').read_text())['best']['fold'];t=fold['times'];I=fold['I'];route=Route(model,'upper')
    try:
        g=grazing(route,[t[1],t[0]+t[2],I]);branches=[continuation(route,g,sign,steps) for sign in (-1,1)]
        return {'case':case,'grazing':{'I':float(g[2]),'times':g[:2].tolist(),'Cmin':mid_minimum(route,g[:2],g[2])},'branches':branches}
    except (ValueError,OverflowError,FloatingPointError) as e:return {'case':case,'error':str(e),'branches':[]}
def main():
    p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=140);p.add_argument('--fm-steps',type=int,default=420);args=p.parse_args()
    if min(args.steps,args.fm_steps)<1:p.error('step budgets must be positive')
    with ProcessPoolExecutor(max_workers=3) as pool:rows=list(pool.map(task,[(c,args.fm_steps if c=='FM' else args.steps) for c in ('AM','AFM','FM')]))
    path=ROOT/'paper/data/random_selected/continued_upper.json';path.write_text(json.dumps({'scope':'finite numerical upper-only route; sampled checks; no completeness claim','results':rows},indent=2,allow_nan=False)+'\n')
    fm=next(x for x in rows if x['case']=='FM');negative=next((b for b in fm['branches'] if b['direction']==-1),None)
    if negative is not None:(ROOT/'paper/data/random_selected/continued_upper_fm_extended.json').write_text(json.dumps({'case':'FM','branch':negative},indent=2,allow_nan=False)+'\n')
    for x in rows:print(x['case'],x.get('grazing'),x.get('error'),[(z['direction'],len(z['points']),z['reason']) for z in x['branches']],flush=True)
if __name__=='__main__':main()
