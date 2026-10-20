"""Refine the RK fitting helper for the selected, independently verified models."""
import json
import numpy as np
from pwl_nest.calcium_coordinates import CalciumTransformPWA,NativeCoordinateDynamic,simulate_coordinates
from fit_positive_coordinates import OUT,MODELS,is_fixed,CalibrationResidual
from fit_calcium_coordinates import dump
from pwl_nest.protocols import simulate


def run():
    records=[];native=NativeCoordinateDynamic()
    for case in ['AM','FM']:
        selected=json.loads((OUT/f'{case}_selection.json').read_text())['selected']
        protocols=json.loads((OUT/f'{case}_protocols.json').read_text())['calibration']
        model=CalciumTransformPWA.load(MODELS/f'selected_{case}.json')
        for p in protocols:
            if is_fixed(p):continue
            with np.load(OUT/selected/f'{case}_{p["name"]}.npz') as trace:
                grid=np.linspace(0,p['duration'],721)
                truth=np.column_stack([np.interp(grid,trace['t'],trace['candidate'][:,d]) for d in [0,1]])
            errors={};curves={}
            for step in [.001,.0002]:
                y=native.run(model,p['initial'],p['duration'],p['tau'],p['baseline'],outputs=721,maxdt=step)
                if y is None:raise AssertionError('Refined RK failed')
                x=np.column_stack([model.coordinate.inverse(y[:,0]),model.gate.inverse(y[:,1])])
                curves[str(step)]=x
                errors[str(step)]=np.max(abs(x-truth),axis=0).tolist()
            reference_refinement=None;original_errors=errors.copy()
            if max(errors['0.0002'])>1e-4:
                atol=np.array([min(2e-14,float(model.coordinate.derivative(.005))*2e-14),
                    min(2e-14,float(model.gate.derivative(.03))*2e-14),2e-14])
                t,path,meta=simulate_coordinates(model,**{k:v for k,v in p.items() if k!='name'},
                    dt=.25,rtol=2e-12,atol=atol,max_step=.03)
                refined=np.column_stack([np.interp(grid,t,path[:,d]) for d in [0,1]])
                reference_refinement=dict(max_C_h_change=np.max(abs(refined-truth),axis=0).tolist(),
                    rtol=2e-12,atol=atol.tolist(),max_step=.03)
                destination=OUT/'numerical_reference';destination.mkdir(exist_ok=True)
                np.savez_compressed(destination/f'{case}_{selected}_{p["name"]}.npz',t=t,candidate=path,latent=meta['latent'])
                for step in [.00005,.00001]:
                    y=native.run(model,p['initial'],p['duration'],p['tau'],p['baseline'],outputs=721,maxdt=step)
                    if y is None:raise AssertionError('Refined RK failed')
                    curves[str(step)]=np.column_stack([model.coordinate.inverse(y[:,0]),model.gate.inverse(y[:,1])])
                    if np.max(abs(curves[str(step)]-refined))<=1e-4:break
                errors={step:np.max(abs(curve-refined),axis=0).tolist() for step,curve in curves.items()}
            if max(list(errors.values())[-1])>1e-4:raise AssertionError(f'Refined RK disagrees with DOP: {case}/{p["name"]}/{errors}')
            records.append(dict(case=case,variant=selected,protocol=p['name'],max_C_h_errors=errors,
                errors_vs_original_reference=original_errors,reference_refinement=reference_refinement))
    refined_scores={}
    for case in ['AM','FM']:
        selection=json.loads((OUT/f'{case}_selection.json').read_text());selected=selection['selected']
        protocols=json.loads((OUT/f'{case}_protocols.json').read_text())['calibration']
        model=CalciumTransformPWA.load(MODELS/f'selected_{case}.json');refs={};replay={}
        for p in protocols:
            refs[p['name']]=simulate(case,**{k:v for k,v in p.items() if k!='name'},dt=.2 if is_fixed(p) else .1)
            refined=OUT/'numerical_reference'/f'{case}_{selected}_{p["name"]}.npz'
            source=refined if refined.exists() else OUT/selected/f'{case}_{p["name"]}.npz'
            with np.load(source) as data:replay[p['name']]=(data['t'],data['candidate'],{})
        residual=CalibrationResidual(case,protocols,refs,model)(model,replay=replay)
        score=float(residual@residual);refined_scores[case]=score
        selection['refined_selected_calibration_loss']=score
        dump(OUT/f'{case}_selection.json',selection)
        model.metadata['refined_calibration_loss']=score
        model.save(MODELS/f'selected_{case}.json');model.save(MODELS/selected/f'{case}.json')
        report_path=OUT/selected/f'{case}.json';report=json.loads(report_path.read_text())
        report.update(refined_calibration_loss=score,metadata=model.metadata);dump(report_path,report)
    dump(OUT/'selected_RK_refinement.json',dict(agreement_tolerance=1e-4,records=records,
        refined_selected_calibration_scores=refined_scores,selection_uses_fresh_tests=False))
    print(json.dumps(records,indent=2))


if __name__=='__main__':run()
