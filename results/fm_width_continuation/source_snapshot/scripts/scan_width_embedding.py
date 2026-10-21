"""Training-only wide scan of input embedding and input gain, with exact rest."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from pwl_nest.calcium_coordinates import CalciumTransformPWA
from pwl_nest.peak_shape import peak_shape_check
from fit_anchored_shapes import ShapeObjective,parameterization,shape_checks,ERRORS
from overnight_coordinate_grid import atomic

parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
args=parser.parse_args();out=Path(args.output);config=json.loads((out/'config.json').read_text())
source=CalciumTransformPWA.load(config['source_model']);objective=ShapeObjective(out,'FM',source)
z,lo,hi,active,m,make=parameterization(source,objective.spec,'control');n=source.latent.region_count
ci=int(np.flatnonzero(active==2*n+4)[0]);gain=int(np.flatnonzero(active==2*n+1)[0]);rows=[];ranked=[];start=time.monotonic()
for shift_ci in np.linspace(-.4,.8,13):
 for shift_gain in np.linspace(-.15,.15,13):
  p=z.copy();p[ci]+=shift_ci;p[gain]+=shift_gain
  row=dict(delta_cI=float(shift_ci),delta_log_qb=float(shift_gain))
  if np.any(p<lo) or np.any(p>hi):continue
  try:
   model=make(p);_,_,paths=objective.paths(model,objective.spec['train'])
   checks=[]
   for spec in objective.spec['train']:
    checks.append(peak_shape_check(*paths[spec['name']][:2],*objective.refs[spec['name']][:2]))
   phase=objective.phase_check();width=max(c['max_width_relative_error'] for c in checks)
   height=max(c['max_height_relative_error'] for c in checks);matches=all(c['counts_match'] for c in checks)
   eligible=bool(all(c['passes'] for c in checks) and phase['passes'])
   merit=width+height+5*(not matches)+10*(not phase['passes'])
   row.update(status='complete',width=width,height=height,counts_match=matches,phase=phase,eligible_dynamic=eligible)
   ranked.append((not eligible,merit,p,row))
  except ERRORS as exc:row.update(status='failed',error=str(exc))
  rows.append(row)
ranked.sort(key=lambda r:r[:2]);directory=out/'embedding_scan';directory.mkdir(exist_ok=True);verified=[]
for k,(_,_,p,row) in enumerate(ranked[:5]):
 model=make(p);r=objective(model);check=shape_checks(objective,model);model.save(directory/f'seed_{k}.json')
 record=dict(rank=k,coarse=row,loss=float(r@r),checks=check,model=str((directory/f'seed_{k}.json').relative_to(out)))
 verified.append(record)
atomic(out/'embedding_scan.json',dict(source_model=config['source_model'],scope='training only, no heldout readout',
 grid=dict(delta_cI=[-.4,.8,13],delta_log_qb=[-.15,.15,13]),rows=rows,full_objective_checks=verified,
 seconds=time.monotonic()-start))
print(json.dumps(dict(complete=sum(r['status']=='complete' for r in rows),dynamic_eligible=sum(r.get('eligible_dynamic',False) for r in rows),
 best=[dict(rank=r['rank'],loss=r['loss'],eligible=r['checks']['eligible'],width=r['checks'].get('max_width_relative_error')) for r in verified])),flush=True)
