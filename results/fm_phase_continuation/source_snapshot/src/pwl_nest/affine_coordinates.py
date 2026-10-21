"""Exact affine recentering of plain calcium-strip models.

Moving coordinate origins is a conjugacy, not an additional model parameter.
All vector-field coefficients, switching surfaces and readout move together.
"""
import copy
import numpy as np
from .calcium_coordinates import CalciumCoordinate,GateCoordinate


def recenter_identity(model,calcium_reference,h_reference,*,calcium_scale=None):
    """Represent the same physical field with different C/h coordinate origins.

    The gate coordinate also changes its normalization when h_reference changes.
    Mixed/oblique/input-hinge models are deliberately outside this helper's scope.
    """
    if (model.coordinate.kind!='identity' or model.h_kind!='identity'
            or model.mix_linear or model.mix_quadratic or len(model.input_thresholds)
            or np.any(model.threshold_slopes)):
        raise ValueError('Require unmixed identity coordinates and calcium-only strips')
    result=copy.deepcopy(model)
    coordinate=CalciumCoordinate('identity',calcium_reference,
        model.coordinate.scale if calcium_scale is None else calcium_scale)
    GateCoordinate('identity',h_reference)  # Validate before using its scale.
    result.coordinate=coordinate;result.h_reference=h_reference;result.__post_init__()
    old_scale=model.h_reference*(1-model.h_reference)
    new_scale=h_reference*(1-h_reference)
    scales=np.array([coordinate.scale/model.coordinate.scale,new_scale/old_scale])
    shifts=np.array([(calcium_reference-model.coordinate.reference)/model.coordinate.scale,
                     (h_reference-model.h_reference)/old_scale])
    original=model.latent.matrix
    matrix=original.copy()
    matrix[:,0]=(original[:,0]+original[:,1:3]@shifts)/scales
    matrix[:,1:3]=original[:,1:3]*scales[None,:]/scales[:,None]
    matrix[:,3]=original[:,3]/scales
    matrix[:,4:]=original[:,4:]*scales[0]/scales[:,None]
    result.latent.matrix=matrix
    result.latent.thresholds=(model.latent.thresholds-shifts[0])/scales[0]
    result.__post_init__()
    # Keep the spectral parameter serialization consistent with the new field.
    n=result.latent.region_count
    if 'parameters' in result.metadata:
        parameters=np.asarray(result.metadata['parameters'],dtype=float).copy()
        _,_,q,qb=result.latent.spectral_coefficients()
        parameters[2*n]=q[0]/qb;parameters[2*n+1]=np.log(qb)
        parameters[2*n+2]=np.log(matrix[0,2]);parameters[2*n+3]=np.log(-matrix[1,2])
        parameters[2*n+4]=matrix[0,3];parameters[2*n+5]=matrix[0,0]
        result.metadata['parameters']=parameters.tolist()
    for key in ['raw_fit_vector','threshold_shifts']:
        result.metadata.pop(key,None)
    result.metadata.update(physical_thresholds_C=coordinate.inverse(result.latent.thresholds).tolist(),
        affine_recenter=dict(physical_model_unchanged=True,old_C_reference=model.coordinate.reference,
            old_h_reference=model.h_reference,new_C_reference=calcium_reference,new_h_reference=h_reference,
            relation='old_latent = scales * new_latent + shifts',scales=scales.tolist(),shifts=shifts.tolist()))
    return result
