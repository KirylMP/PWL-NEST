"""Re-evaluate preserved previous sqrt on selected common physical protocols."""
import json
import numpy as np
from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA,simulate_coordinates
from pwl_nest.protocols import simulate
from fit_strict import stats
from scipy.signal import find_peaks

OUT=ROOT/'results/coordinate_families';report={}
for case in ['AM','FM']:
    model_path=ROOT/f'models/calcium_powers/sqrt/{case}.json'
    model=CalciumTransformPWA.load(model_path)
    data=json.loads((OUT/f'sqrt/{case}.json').read_text())
    names=['decay_tau_20.0','decay_tau_100.0','decay_tau_50.0','fixed_0.638' if case=='AM' else 'fixed_1.13']
    protos={p['name']:p for p in data['training_protocols']+data['historical_diagnostics']}
    records=[]
    for name in names:
        kwargs={k:v for k,v in protos[name].items() if k!='name'}
        kwargs['dt']=.2 if name.startswith('fixed') else .1
        t,truth,_=simulate(case,**kwargs);tt,x,meta=simulate_coordinates(model,**kwargs)
        np.testing.assert_allclose(t,tt,atol=1e-12)
        r=dict(name=name,rmse=np.sqrt(np.mean((x-truth)**2,axis=0)).tolist(),physical=meta['physical'])
        if name.startswith('fixed'):r.update(reference=stats(t,truth),candidate=stats(t,x))
        else:r['peaks']=dict(reference=len(find_peaks(truth[:,0],prominence=.02)[0]),candidate=len(find_peaks(x[:,0],prominence=.02)[0]))
        if name=='decay_tau_50.0':np.savez_compressed(OUT/f'{case}_incumbent_tau50.npz',t=t,lr=truth,candidate=x)
        records.append(r)
    report[case]=dict(model=str(model_path.relative_to(ROOT)),
        reported_previous_training_loss=model.metadata['training_data_loss'],
        note='training loss copied from previous fit, not recomputed; listed trajectories rerun with current code',results=records)
(OUT/'incumbents.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
