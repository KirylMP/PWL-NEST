"""Summarize completed refits without ranking unresolved/missing cycles as zero."""
import argparse
import json
from pathlib import Path


def error(rows, key, model):
    pairs = [(r[model][key], r['target']['target_'+key]) for r in rows
             if r.get(model) is not None and r.get('target', {}).get('target_'+key) is not None]
    if not pairs: return 'NA'
    return f'{sum(abs(a-b) for a,b in pairs)/len(pairs):.5g}'


def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);args=p.parse_args()
    paths=sorted(args.root.rglob('report.json'))
    if not paths: raise SystemExit(f'No completed reports below {args.root}')
    print('case arm seed train_loss/base holdout_cycles Cmin_MAE Cmax_MAE period_MAE fold_residual logM closure optimizer')
    for path in paths:
        r=json.loads(path.read_text()); h=r['holdout']; n=sum(x.get('candidate') is not None for x in h)
        fold=r['fold']; closure=fold.get('independent',{}).get('closure')
        print(r['case'],r['arm'],r['seed'],f"{r['best_squared_residual']:.4g}/{r['baseline_squared_residual']:.4g}",
              f'{n}/{len(h)}',*(error(h,k,'candidate') for k in ('Cmin','Cmax','period')),
              f"{fold['matching_max']:.2g}",f"{fold['logM']:.2g}",
              f'{closure:.2g}' if closure is not None else 'FAILED',r['optimizer']['success'])


if __name__=='__main__': main()
