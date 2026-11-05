"""Summarize per-start statuses without conflating failure with nonexistence."""
import argparse,json
from collections import Counter
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
print('case seed status loss holdout/total min_C trace_discriminant optimizer')
counts=Counter()
for status in sorted(a.directory.glob('*/seed_*/status.json')):
    d=json.loads(status.read_text());case=d['case'];seed=d['seed'];state=d['status'];counts[(case,state)]+=1;report=status.parent/'fit/report.json'
    if report.exists():
        r=json.loads(report.read_text());b=r['best'];hold=r['holdout'];print(case,seed,state,round(b['loss'],6),f"{sum(bool(x['cycle']['candidates']) for x in hold)}/{len(hold)}",round(min(x['Cmin'] for x in b['cycles']),6),[round(x,6) for x in b['discriminant']],repr(r['message']))
    else:print(case,seed,state,'- - - -',d.get('detail',''))
print('Counts:',dict(counts))
print('Failed starts or unresolved shooting do not prove nonexistence.')
