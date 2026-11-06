"""Evaluate one trial seed with return shooting bootstrapped from IVP.

Not optimization; bootstrap is a finite convergence-dependent initializer.
"""
import argparse,json
from pathlib import Path
from return_cycle_metrics import solve_cycle
from cycle_bootstrap import bootstrap
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--report',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-nonphysical',action='store_true');a=p.parse_args();d=json.loads(a.candidate.read_text());fold=json.loads(a.report.read_text())['fold'];base=json.loads((ROOT/'paper/data/physical_models/fixed_fold'/a.case/'parameters.json').read_text());out=[];a.output.parent.mkdir(parents=True,exist_ok=True)
    for row in base['training_cycles']:
        seeds=bootstrap(d,row['I']);result=solve_cycle(d,row['I'],fold['times'],extra_seeds=seeds,require_positive=not a.allow_nonphysical);record={'I':row['I'],'target':{k:row[k] for k in ('target_Cmin','target_Cmax','target_period')},'cycle':result};out.append(record);a.output.write_text(json.dumps({'case':a.case,'scope':'initial seed only; bootstrap IVP; finite route search','results':out},indent=2)+'\n');print(a.case,row['I'],[(s['route'],s['Cmin'],s['Cmax'],s['period']) for s in result['candidates']],flush=True)

if __name__=='__main__':main()
