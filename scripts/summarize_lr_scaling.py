"""Pair repeated LR/PWA runtimes for each case, size and seed."""
import argparse,csv,json,statistics
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
groups=defaultdict(lambda:defaultdict(dict));diagnostics={}
with (a.directory/'runs.jsonl').open() as f:
 for line in f:
  r=json.loads(line);key=(r['case'],r['count'],r['seed'])
  if r.get('diagnostic'):diagnostics[key,r['kind']]=r['result'];continue
  if not r['warmup']:groups[key][r['kind']][r['repeat']]=r['result']
rows=[]
for (case,count,seed),by_model in sorted(groups.items()):
 row={'case':case,'astrocytes':count,'seed':seed}
 for model in ('LR','PWA'):
  samples=list(by_model[model].values());times=[x['run_s'] for x in samples if 'run_s' in x]
  row[model+'_success']=len(times);row[model+'_median_s']=statistics.median(times) if times else None
  row[model+'_errors']=' | '.join(x['error'] for x in samples if 'error' in x)
  diag=diagnostics.get(((case,count,seed),model),{});row[model+'_diagnostic_error']=diag.get('error','')
  for k,v in diag.get('diagnostic',{}).items():row[model+'_'+k]=v
  graphs={x['selected_edges_sha256'] for x in samples if 'selected_edges_sha256' in x}
  row[model+'_graph_sha256']=';'.join(sorted(graphs))
 row['graphs_match']=bool(row['LR_graph_sha256'] and row['LR_graph_sha256']==row['PWA_graph_sha256'])
 x,y=row['LR_median_s'],row['PWA_median_s'];row['speedup']=x/y if x is not None and y and row['graphs_match'] else None
 rows.append(row)
fields=sorted({key for row in rows for key in row},key=lambda k:(k not in ('case','astrocytes','seed'),k))
with (a.directory/'summary.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('Saved',a.directory/'summary.csv','groups',len(rows),'unpaired or failed',sum(bool(r['speedup'] is None or r['LR_errors'] or r['PWA_errors'] or r['LR_diagnostic_error'] or r['PWA_diagnostic_error']) for r in rows))
