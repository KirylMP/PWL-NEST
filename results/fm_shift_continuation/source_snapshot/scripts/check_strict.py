"""Independent numerical structural checks of the persisted strict models."""
import json
import numpy as np
from scipy.integrate import solve_ivp
from pwl_nest.models import ROOT
from pwl_nest.strict_pwa import StrictPWA,regional_flow,vertices


def check(model,seed=981):
    rng = np.random.default_rng(seed)
    # Roundoff-sensitive affine trace comparisons on internal mesh planes.
    continuity = 0.
    flow_error = 0.
    affine_error = 0.
    ip3_error = 0.
    for _ in range(120):
        cells = [int(rng.integers(len(a)-1)) for a in model.axes]
        lo = np.array([a[j] for a,j in zip(model.axes,cells)])
        width = np.array([a[j+1]-a[j] for a,j in zip(model.axes,cells)])
        frac = rng.uniform(.15,.85,3)
        point = lo+frac*width
        tau = float(rng.choice([2.,20.,100.]))
        baseline = .2
        a,b,coords = model.region(point,tau,baseline)
        affine_error = max(affine_error,float(np.max(abs(a@point+b-model.rhs(tau,baseline)(0,point)))))
        # A very short propagation is checked to stay in the initial region.
        dt = 1e-5
        target = regional_flow(a,b,point,dt)
        bary = np.linalg.solve(np.c_[coords,np.ones(4)].T,np.r_[target,1.])
        if bary.min() < -1e-10:
            raise AssertionError('Chosen short analytic check crosses a region')
        numerical = solve_ivp(model.rhs(tau,baseline,stage_extrapolation=True),(0,dt),point,
                              rtol=1e-11,atol=1e-13)
        if not numerical.success:
            raise AssertionError(numerical.message)
        flow_error = max(flow_error,float(np.max(abs(target-numerical.y[:,-1]))))
        ip3_error = max(ip3_error,abs(target[2]-(baseline+(point[2]-baseline)*np.exp(-dt/tau))))
        d = int(rng.integers(3))
        # Rectangular shared plane, from each neighbouring cell.
        if cells[d] > 0:
            face = point.copy()
            face[d] = model.axes[d][cells[d]]
            eps = np.zeros(3);eps[d] = min(np.diff(model.axes[d]))*1e-8
            left_a,left_b,_ = model.region(face-eps,tau,baseline)
            right_a,right_b,_ = model.region(face+eps,tau,baseline)
            continuity = max(continuity,float(np.max(abs(left_a@face+left_b-right_a@face-right_b))))
        # Interior diagonal faces (equal normalized coordinates).
        d,e = rng.choice(3,2,replace=False)
        face = point.copy()
        face[d] = lo[d]+.55*width[d]
        face[e] = lo[e]+.55*width[e]
        eps = np.zeros(3);eps[d],eps[e] = width[d]*1e-8,-width[e]*1e-8
        left_a,left_b,_ = model.region(face-eps,tau,baseline)
        right_a,right_b,_ = model.region(face+eps,tau,baseline)
        continuity = max(continuity,float(np.max(abs(left_a@face+left_b-right_a@face-right_b))))
    nodes = vertices(model.axes)
    inward = []
    for d in [0,1]:
        low = model.values[nodes[:,d]==model.axes[d][0],d].min()
        high = model.values[nodes[:,d]==model.axes[d][-1],d].max()
        inward.append(dict(axis=d,min_lower_velocity=float(low),max_upper_velocity=float(high)))
        if low < -1e-13 or high > 1e-13:
            raise AssertionError('Non-inward fitted boundary')
    result = dict(regions=model.region_count,nodes=len(model.values),
                  affine_max_abs_error=affine_error,continuity_max_abs_error=continuity,
                  regional_flow_max_abs_error=flow_error,IP3_max_abs_error=ip3_error,
                  random_interior_checks=120,inward_boundary=inward)
    if max(affine_error,continuity)>1e-10 or flow_error>1e-10 or ip3_error>1e-13:
        raise AssertionError(result)
    return result

if __name__=='__main__':
    report = {case:check(StrictPWA.load(ROOT/f'models/strict_pwa/{case}.npz')) for case in ['AM','FM']}
    (ROOT/'results/strict_pwa/structure.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
