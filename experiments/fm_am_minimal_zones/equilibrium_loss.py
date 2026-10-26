"""Match actual physical equilibrium sets, including missing and extra roots."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def matched_cost(reference, candidate, scale=.05, penalty=25.):
    reference=np.asarray(reference,dtype=float).reshape(-1,2)
    candidate=np.asarray(candidate,dtype=float).reshape(-1,2)
    nr,nc=len(reference),len(candidate)
    if nr+nc==0:return 0.
    costs=np.zeros((nr+nc,nr+nc))
    costs[:nr,:nc]=np.sum(((reference[:,None,:]-candidate[None,:,:])/scale)**2,axis=2)
    costs[:nr,nc:]=penalty;costs[nr:,:nc]=penalty
    a,b=linear_sum_assignment(costs)
    return float(costs[a,b].sum())


def model_cost(model, source, enumerate_model):
    total=0.;count=0
    for I,reference in source:
        candidates=enumerate_model(float(I),model)
        xy=[[c['C'],c['z'][1]] for c in candidates]
        total+=matched_cost(reference,xy);count+=len(reference)
    return total/max(count,1)
