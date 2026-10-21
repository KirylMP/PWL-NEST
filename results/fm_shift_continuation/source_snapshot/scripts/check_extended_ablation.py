"""Frozen-coefficient training ablation; no reoptimization or selection."""
import copy,json
import numpy as np
from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA
from pwl_nest.calcium_strips import CalciumStrips
from fit_calcium_coordinates import training_data,dump
from fit_movable_thresholds import CommonResidual

OUT=ROOT/'results/extended_pwa';report={}
for case in ['AM','FM']:
    baseline=CalciumTransformPWA.load(ROOT/f'models/extended_pwa/baseline/{case}.json')
    data=training_data(case);objective=CommonResidual(data,baseline);records=[]
    for variant in ['input_one','input_two','tilted']:
        model=CalciumTransformPWA.load(ROOT/f'models/extended_pwa/{variant}/{case}.json')
        metadata=copy.deepcopy(model.metadata);metadata['physical_thresholds_C']=baseline.metadata['physical_thresholds_C']
        latent=CalciumStrips(case,model.coordinate.forward(metadata['physical_thresholds_C']),model.latent.matrix.copy(),metadata)
        removed=CalciumTransformPWA(latent,model.coordinate,model.h_reference,model.h_kind)
        full=objective(model);ablated=objective(removed)
        delta_C=model.coordinate.inverse(model.thresholds_at_input(1.3))-model.coordinate.inverse(model.thresholds_at_input(0.))
        records.append(dict(variant=variant,original_helper_loss=float(full@full),
            remove_geometry_keep_coefficients_loss=float(ablated@ablated),
            max_threshold_change_C_over_input_domain=float(abs(delta_C).max()),
            input_hinge_weights=model.input_weights.tolist(),threshold_slopes=model.threshold_slopes.tolist()))
    report[case]=records
dump(OUT/'geometry_ablation.json',dict(note='Frozen learned coefficients, all new geometry removed; original RK4 training helper, no refit, no selection',results=report))
print(json.dumps(report,indent=2))
