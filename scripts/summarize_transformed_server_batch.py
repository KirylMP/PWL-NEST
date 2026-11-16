"""Display current fold-search and fitting progress, including unfinished runs."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();root=a.output
 groups=defaultdict(Counter);best={}
 for f in (root/'search').glob('*/*/status.json'):
  try:j=json.loads(f.read_text())
  except (OSError,ValueError):continue
  key=(j['case'],j['kind']);groups[key]['attempts']+=1
  if j.get('distinct'):groups[key]['distinct_folds']+=1
  elif j.get('fold_valid'):groups[key]['duplicates_or_pending']+=1
  else:groups[key]['rejected']+=1
 for f in (root/'fits').glob('*/*/status.json'):
  try:j=json.loads(f.read_text())
  except (OSError,ValueError):continue
  key=(j['case'],j['kind']);groups[key]['fit_'+j['status']]+=1
  if j['status']=='completed' and (key not in best or j['training_loss']<best[key]['training_loss']):best[key]=j
 print('case kind attempts distinct_folds rejected fit_completed fit_other best_seed training_loss holdout_Cmax_MAE holdout_period_MAE')
 for key in sorted(groups):
  g=groups[key];b=best.get(key,{});other=sum(n for k,n in g.items() if k.startswith('fit_') and k!='fit_completed')
  print(*key,g['attempts'],g['distinct_folds'],g['rejected'],g['fit_completed'],other,b.get('seed','-'),b.get('training_loss','-'),b.get('holdout_MAE_Cmax','-'),b.get('holdout_MAE_period','-'))
 print('\nFold-parameter local fixability: parameter case kind candidates full_rank fraction median_sigma_min selected_as_pivot')
 for folder in sorted((root/'search').glob('*_*')):
  case,kind=folder.name.rsplit('_',1);diagnostics=[];selected=[]
  for f in folder.glob('attempt_*/status.json'):
   try:j=json.loads(f.read_text())
   except (OSError,ValueError):continue
   if j.get('fold_valid'):
    selected.append(j.get('selected_pivot'))
    diagnostics.extend(j.get('pivot_diagnostics',[]))
  by_name=defaultdict(list)
  for d in diagnostics:by_name[d['name']].append(d)
  for name,rows in sorted(by_name.items()):
   print(name,case,kind,len(rows),sum(r['locally_solvable'] for r in rows),f'{sum(r["locally_solvable"] for r in rows)/len(rows):.3f}',f'{np.median([r["relative_sigma_min"] for r in rows]):.3g}',sum(x==name for x in selected))
 print('\nLocal full rank is finite-difference evidence at the sampled fold, not proof of global solvability; completed means four holdouts resolved, not that the fit is accurate.')
if __name__=='__main__':main()
