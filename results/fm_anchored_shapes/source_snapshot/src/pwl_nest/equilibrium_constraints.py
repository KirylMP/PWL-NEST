"""Exact equilibrium constraints for continuous coordinate PWA models."""
import copy
import numpy as np


def anchor_equilibrium(model, physical_rest):
    """Return an independent model with F(T(rest))=0 by deriving two offsets.

    Only the two constant columns change. All regional state matrices, input
    columns, switching surfaces and continuity relations are preserved. Stability
    must be checked separately: shifting offsets alone does not enforce it.
    """
    anchored=copy.deepcopy(model)
    physical_rest=np.asarray(physical_rest,dtype=float)
    latent_rest=anchored.to_latent(physical_rest)
    anchored.latent.matrix[:,0]-=anchored.field(latent_rest)
    # Re-encode the two now-derived entries for compatible model serialization.
    n=anchored.latent.region_count
    if 'parameters' in anchored.metadata:
        parameters=np.array(anchored.metadata['parameters'],dtype=float)
        if parameters.shape!=(2*n+6,):raise ValueError('Unexpected coefficient parameter vector')
        B,D=anchored.latent.matrix[:,2]
        qb=anchored.latent.spectral_coefficients()[3]
        parameters[2*n]=(B*anchored.latent.matrix[1,0]-D*anchored.latent.matrix[0,0])/qb
        parameters[2*n+5]=anchored.latent.matrix[0,0]
        anchored.metadata['parameters']=parameters.tolist()
    anchored.metadata.update(exact_rest_constraint=physical_rest.tolist(),
        derived_coefficient_parameters=['input_offset','first_row_constant'],
        rest_constraint='two constants derived from F(T(LR_rest))=0; stability checked independently')
    return anchored
