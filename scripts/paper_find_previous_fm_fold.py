"""Finite candidate-fold search for the previous no-fold-target mixed FM."""
import argparse,concurrent.futures,json,os,sys
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'paper/figures/figure6_data'

def worker(task):
    model_root,arguments=task
    sys.path.insert(0,str(model_root/'astrocyte_lab/scripts'))
    sys.path.insert(0,str(model_root/'astrocyte_lab/experiments/paper_fold_audit'))
    from audit import solve
    return solve(arguments)

def main():
    p=argparse.ArgumentParser();p.add_argument('--model-root',type=Path,default=ROOT.parent/'PWL-NEST');p.add_argument('--starts',type=int,default=20);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    d=json.loads((DATA/'no_fold/FM/parameters.json').read_text());reference={'I':1.1231503496462252,'period':19.10456189314679}
    tasks=[(a.model_root.resolve(),('fm',d,reference,first,n,9000+100*first+n)) for first in (0,1) for n in range(a.starts)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:results=list(pool.map(worker,tasks))
    (DATA/'no_fold_FM_fold_search.json').write_text(json.dumps(dict(scope='Finite multistart candidate fold search; root and local checks do not establish global completeness.',results=results),indent=2)+'\n')
    print('candidate folds',sum(r['status']=='candidate_fold' for r in results),'of',len(results),flush=True)
if __name__=='__main__':main()
