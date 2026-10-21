"""Select only on the common TRAIN residual recomputed using saved DOP paths.

The independent validation grid contains all 361 fitting samples exactly.
Static native statistics and the original soft source-I anchors are unchanged.
No historical diagnostic metric participates in this numerical audit.
"""
import json
import numpy as np
from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA
from fit_calcium_coordinates import training_data,dump
from fit_movable_thresholds import CommonResidual

OUT=ROOT/'results/extended_pwa';MODELS=ROOT/'models/extended_pwa'


class DOPTrainingReplay:
    def __init__(self,case,variant,training):
        self.entries=[]
        for p in training:
            if not p['name'].startswith('decay'):continue
            refined=OUT/'numerical_reference'/f'{case}_{variant}_{p["name"]}.npz'
            path=refined if refined.exists() else OUT/variant/f'{case}_{p["name"]}_train.npz'
            with np.load(path) as trace:
                grid=np.linspace(0,p['duration'],361)
                distance=abs(trace['t'][:,None]-grid).min(axis=0)
                if distance.max()>1e-10:raise AssertionError('Saved DOP grid misses fitting samples')
                y=np.column_stack([np.interp(grid,trace['t'],trace['latent'][:,d]) for d in [0,1]])
            self.entries.append((p,y))

    def run(self,model,initial,duration,tau,baseline,**kwargs):
        matches=[y for p,y in self.entries if p['duration']==duration and p['tau']==tau and p['baseline']==baseline and np.array_equal(p['initial'],initial)]
        if len(matches)!=1:raise AssertionError('Missing unique training path')
        return matches[0]


report={}
for case in ['AM','FM']:
    selection=json.loads((OUT/f'{case}_selection.json').read_text());data=training_data(case)
    seed=CalciumTransformPWA.load(MODELS/'baseline'/f'{case}.json');objective=CommonResidual(data,seed)
    verified={};coarse=selection.get('optimizer_training_scores',selection['training_scores']).copy()
    for variant in selection['variants']:
        model=CalciumTransformPWA.load(MODELS/variant/f'{case}.json')
        objective.dynamic=DOPTrainingReplay(case,variant,data[0]);r=objective(model);score=float(r@r)
        verified[variant]=score
        if variant=='baseline':model.metadata['active_fit_parameters']=0
        model.metadata.update(class_name='PWA_in_transformed_calcium_and_gate',
            verified_training_data_loss=score,verified_training_method='native static + independent DOP853 training paths + original source-I anchors')
        model.save(MODELS/variant/f'{case}.json')
        result_path=OUT/variant/f'{case}.json';result=json.loads(result_path.read_text());result['metadata']=model.metadata
        dump(result_path,result)
    best=min(verified,key=verified.get);model=CalciumTransformPWA.load(MODELS/best/f'{case}.json');model.save(MODELS/f'selected_{case}.json')
    selection.update(optimizer_training_scores=coarse,verified_training_scores=verified,training_scores=verified.copy(),
        selected_before_numerical_audit=selection.get('selected_before_numerical_audit',selection['selected']),
        selected=best,selection='minimum common training residual verified by independent DOP paths; no diagnostics')
    dump(OUT/f'{case}_selection.json',selection)
    report[case]=dict(optimizer_training_scores=coarse,verified_training_scores=verified,selected=best,
        numerical_audit='dynamic training samples replaced by independent DOP853; no reoptimization or diagnostic selection')
dump(OUT/'verified_training.json',report);print(json.dumps(report,indent=2))
