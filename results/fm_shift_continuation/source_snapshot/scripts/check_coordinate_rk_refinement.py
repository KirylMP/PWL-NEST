"""Refine only helpers whose existing DOP comparison exceeds 1e-3 uM."""
import argparse,json
import numpy as np
from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA,NativeCoordinateDynamic,simulate_coordinates

parser=argparse.ArgumentParser();parser.add_argument('--experiment',choices=['coordinate_families','extended_pwa'],default='coordinate_families');args=parser.parse_args()
OUT=ROOT/'results'/args.experiment;records=[];native=NativeCoordinateDynamic()
for case in ['AM','FM']:
    variants=json.loads((OUT/f'{case}_selection.json').read_text())['variants']
    for variant in variants:
        data=json.loads((OUT/variant/f'{case}.json').read_text())
        protos={p['name']:p for p in data['training_protocols']}
        for r in data['results']:
            check=r.get('RK_check',{})
            if check.get('max_abs_C_vs_DOP',0)<=.001:continue
            model=CalciumTransformPWA.load(ROOT/f'models/{args.experiment}/{variant}/{case}.json')
            p=protos[r['name']]
            with np.load(OUT/variant/f'{case}_{r["name"]}_train.npz') as tr:
                grid=np.linspace(0,p['duration'],361)
                truth=np.interp(grid,tr['t'],tr['candidate'][:,0])
            errors={};curves={}
            for step in [.001,.0002,.00005]:
                y=native.run(model,p['initial'],p['duration'],p['tau'],p['baseline'],maxdt=step)
                if y is None:raise AssertionError('Refined RK failed')
                curves[str(step)]=model.coordinate.inverse(y[:,0])
                errors[str(step)]=float(abs(curves[str(step)]-truth).max())
                if step<=.0002 and errors[str(step)]<=1e-5:break
            reference_refinement=None;original_errors=errors.copy()
            if list(errors.values())[-1]>1e-5:
                kwargs={k:v for k,v in p.items() if k!='name'}
                t,x,meta=simulate_coordinates(model,**kwargs,dt=.1,rtol=2e-11,atol=2e-13,max_step=.03)
                refined=np.interp(grid,t,x[:,0])
                errors={step:float(abs(curve-refined).max()) for step,curve in curves.items()}
                reference_refinement=dict(rtol=2e-11,atol=2e-13,max_step=.03,
                    C_difference_at_training_samples=float(abs(refined-truth).max()),
                    IP3_error=meta['ip3_max_abs_error'])
                destination=OUT/'numerical_reference';destination.mkdir(exist_ok=True)
                np.savez_compressed(destination/f'{case}_{variant}_{r["name"]}.npz',t=t,candidate=x,latent=meta['latent'])
            tolerance=1e-4 if args.experiment=='extended_pwa' else 1e-5
            if list(errors.values())[-1]>tolerance:raise AssertionError(f'RK did not agree within {tolerance}: {case}/{variant}/{r["name"]}: {errors}')
            records.append(dict(case=case,variant=variant,protocol=r['name'],original_check=check,
                errors_vs_original_DOP=original_errors,refined_errors=errors,reference_refinement=reference_refinement,
                agreement_tolerance_uM=tolerance))
(OUT/'RK_refinement.json').write_text(json.dumps(dict(trigger_uM=.001,records=records),indent=2)+'\n')
print(json.dumps(records,indent=2))
