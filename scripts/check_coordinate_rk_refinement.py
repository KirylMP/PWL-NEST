"""Refine only helpers whose existing DOP comparison exceeds 1e-3 uM."""
import json
import numpy as np
from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA,NativeCoordinateDynamic

OUT=ROOT/'results/coordinate_families';records=[];native=NativeCoordinateDynamic()
for case in ['AM','FM']:
    variants=json.loads((OUT/f'{case}_selection.json').read_text())['variants']
    for variant in variants:
        data=json.loads((OUT/variant/f'{case}.json').read_text())
        protos={p['name']:p for p in data['training_protocols']}
        for r in data['results']:
            check=r.get('RK_check',{})
            if check.get('max_abs_C_vs_DOP',0)<=.001:continue
            model=CalciumTransformPWA.load(ROOT/f'models/coordinate_families/{variant}/{case}.json')
            p=protos[r['name']]
            with np.load(OUT/variant/f'{case}_{r["name"]}_train.npz') as tr:
                grid=np.linspace(0,p['duration'],361)
                truth=np.interp(grid,tr['t'],tr['candidate'][:,0])
            errors={}
            for step in [.001,.0002]:
                y=native.run(model,p['initial'],p['duration'],p['tau'],p['baseline'],maxdt=step)
                if y is None:raise AssertionError('Refined RK failed')
                errors[str(step)]=float(abs(model.coordinate.inverse(y[:,0])-truth).max())
            if errors['0.0002']>1e-5:raise AssertionError('RK did not converge to DOP')
            records.append(dict(case=case,variant=variant,protocol=r['name'],original_check=check,refined_errors=errors))
(OUT/'RK_refinement.json').write_text(json.dumps(dict(trigger_uM=.001,records=records),indent=2)+'\n')
print(json.dumps(records,indent=2))
