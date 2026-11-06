"""Summarize hard-constrained AM reset optimization starts."""
import argparse
from collections import Counter
import json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args();count=Counter();rows=[]
for path in sorted(a.directory.glob('seed_*/status.json')):
    s=json.loads(path.read_text());count[s['status']]+=1
    if s['status']=='completed':
        h=s['holdout'];rows.append((s['best_loss'],s['seed'],s['source_seed'],s['min_C'],sum(bool(x['cycle']['candidates']) for x in h),len(h),s['optimizer_success'],s['message']))
print('Counts:',dict(count));print('loss seed source min_C holdout optimizer_success message')
for loss,seed,source,C,n,total,success,msg in sorted(rows):print(round(loss,6),seed,source,round(C,8),f'{n}/{total}',success,repr(msg))
