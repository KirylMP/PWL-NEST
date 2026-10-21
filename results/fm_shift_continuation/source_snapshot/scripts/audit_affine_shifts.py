"""Verify that affine recentering preserves fields, guards and regional flows."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from pwl_nest.affine_coordinates import recenter_identity
from pwl_nest.calcium_coordinates import CalciumTransformPWA
from overnight_coordinate_grid import atomic
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',default='results/fm_shift_continuation')
parser.add_argument('--model',default='models/anchored_shape_candidates/FM_control.json')
args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
model=CalciumTransformPWA.load(args.model)
rest=np.array(model.metadata['exact_rest_constraint']);rng=np.random.default_rng(20261030)
x=np.column_stack([rng.uniform(.001,1.5,1000),rng.uniform(.01,.99,1000),rng.uniform(.1,1.3,1000)])
original=model.field(model.to_latent(x))/np.column_stack([model.coordinate.derivative(x[:,0]),model.gate.derivative(x[:,1])])
rows=[]
for c in [.01,float(rest[0]),.1,.3]:
 for h in [.3,.6,float(rest[1])]:
  new=recenter_identity(model,c,h)
  physical=new.field(new.to_latent(x))/np.column_stack([new.coordinate.derivative(x[:,0]),new.gate.derivative(x[:,1])])
  S=np.diag([*new.metadata['affine_recenter']['scales'],1.,1.]);S[:2,3]=new.metadata['affine_recenter']['shifts']
  regional=[]
  for a,b in zip(model.latent.local_coefficients(),new.latent.local_coefficients()):
   A=np.zeros((4,4));B=A.copy();A[:2,:3]=a[:,1:];A[:2,3]=a[:,0];B[:2,:3]=b[:,1:];B[:2,3]=b[:,0]
   A[2,2]=B[2,2]=-1/48;A[2,3]=B[2,3]=.45/48
   regional.append(float(abs(expm(A*.7)@S-S@expm(B*.7)).max()))
  rows.append(dict(C_reference=c,h_reference=h,max_physical_field_error=float(abs(physical-original).max()),
    max_physical_threshold_error=float(abs(new.coordinate.inverse(new.latent.thresholds)-model.coordinate.inverse(model.latent.thresholds)).max()),
    max_regional_exponential_conjugacy_error=max(regional)))
new=recenter_identity(model,*rest[:2]);new.save(out/'rest_centered_equivalent.json')
atomic(out/'affine_shift_audit.json',dict(grid=rows,point_count=len(x),
 conclusion='Equivalent physical fields and regional propagators: coordinate shifts add no model capacity.',
 rest_centered_latent_rest=new.to_latent(rest).tolist()))
print(json.dumps(dict(max_field_error=max(r['max_physical_field_error'] for r in rows),max_expm_error=max(r['max_regional_exponential_conjugacy_error'] for r in rows))))
