"""Continuous simplicial PWA in physical (C,h,I); no state transformations.

Every rectilinear cell has the same conforming Freudenthal triangulation.
Four shared vertex values define the entire affine calcium/receptor field.
The IP3 row is exact, independently of the approximation or relaxation time.
"""
from dataclasses import dataclass
from pathlib import Path
import bisect
import itertools
import json
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.sparse import coo_matrix, eye, vstack
from scipy.sparse.linalg import lsmr

C_MAX = 2 / 1.185
PERMUTATIONS = list(itertools.permutations(range(3)))


def lr_field(points, case):
    """Vectorized nonlinear calcium/receptor field, last dimension (C,h,I)."""
    c, h, ip3 = np.moveaxis(np.asarray(points), -1, 0)
    ker = {'AM': .1, 'FM': .05}[case]
    q = 1.049*(ip3+.13)/(ip3+.9434)
    dc = (6*(ip3/(ip3+.13))**3*(c/(c+.08234))**3*h**3+.11)*(2-1.185*c)
    dc -= .9*c*c/(c*c+ker*ker)
    dh = .2*(q-(q+c)*h)
    return np.stack((dc, dh), axis=-1)


def equilibrium_h(c, ip3):
    q = 1.049*(ip3+.13)/(ip3+.9434)
    return q/(q+c)


def spectral_events(case):
    """Source LR events only, NOT claimed PWA bifurcations. Legacy lr_spectral heuristic."""
    def force(c, ip3):
        return lr_field([c, equilibrium_h(c, ip3), ip3], case)[0]
    def input_at(c):
        return brentq(lambda ip3: force(c, ip3), 0., 1.3, xtol=1e-13)
    def spectrum(c):
        ip3 = input_at(c)
        h = equilibrium_h(c, ip3)
        ker = .1 if case == 'AM' else .05
        q = 1.049*(ip3+.13)/(ip3+.9434)
        n, gain, d = c/(c+.08234), 6*(ip3/(ip3+.13))**3, 2-1.185*c
        fc = 3*gain*n*n*.08234/(c+.08234)**2*h**3*d
        fc -= (gain*n**3*h**3+.11)*1.185 + 1.8*c*ker**2/(c*c+ker**2)**2
        fh = 3*gain*n**3*h*h*d
        hc, hh = -.2*h, -.2*(q+c)
        tr, det = fc+hh, fc*hh-fh*hc
        return np.array([tr, det, tr*tr-4*det])
    lo = brentq(lambda c: force(c, 0.), 1e-8, C_MAX)
    hi = brentq(lambda c: force(c, 1.3), 1e-8, C_MAX)
    grid = np.linspace(lo+1e-9, hi-1e-9, 1001)
    values = np.array([spectrum(c) for c in grid])
    events = []
    for col, label in enumerate(['trace_zero', 'fold', 'node_focus']):
        for a, b, fa, fb in zip(grid[:-1], grid[1:], values[:-1,col], values[1:,col]):
            if fa*fb >= 0:
                continue
            c = brentq(lambda x: spectrum(x)[col], a, b, xtol=1e-13)
            ip3 = input_at(c)
            tr, det, disc = spectrum(c)
            if col == 0 and det <= 0:
                continue
            events.append(dict(kind=label, C=c, h=float(equilibrium_h(c,ip3)),
                               I=ip3, trace=tr, determinant=det, discriminant=disc))
    return events


def heuristic_axes(case, refine=False):
    events = spectral_events(case)
    # Resolve low-C saturation/pump scales; cover the complete invariant box.
    c = np.unique(np.r_[0., np.geomspace(.005,.15,10), np.linspace(.15,C_MAX,9),
                        [e['C'] for e in events]])
    h = np.unique(np.r_[np.linspace(0,1,11), [e['h'] for e in events]])
    ip3 = np.unique(np.r_[np.linspace(0,1.3,14), [e['I'] for e in events]])
    if refine:
        # Nested mesh; refinement is fixed before holdout simulations.
        c = np.sort(np.r_[c, (c[:-1]+c[1:])/2])
        h = np.sort(np.r_[h, (h[:-1]+h[1:])/2])
        ip3 = np.sort(np.r_[ip3, (ip3[:-1]+ip3[1:])/2])
    return (c,h,ip3), events


def vertices(axes):
    return np.stack(np.meshgrid(*axes,indexing='ij'),axis=-1).reshape(-1,3)


def interpolation_rows(axes, points):
    """Sparse continuous P1 basis; deterministic diagonal-face ownership."""
    points = np.atleast_2d(points)
    cells = np.column_stack([np.clip(np.searchsorted(a,points[:,d],side='right')-1,0,len(a)-2)
                             for d,a in enumerate(axes)])
    lower = np.column_stack([a[cells[:,d]] for d,a in enumerate(axes)])
    width = np.column_stack([np.diff(a)[cells[:,d]] for d,a in enumerate(axes)])
    u = (points-lower)/width
    if (u < -1e-10).any() or (u > 1+1e-10).any():
        raise ValueError('Point outside fitted physical box; no extrapolation')
    order = np.argsort(-u,axis=1,kind='stable')
    sorted_u = np.take_along_axis(u,order,axis=1)
    weights = np.column_stack([1-sorted_u[:,0], sorted_u[:,:-1]-sorted_u[:,1:], sorted_u[:,-1]])
    nodes = [cells.copy()]
    cursor = cells.copy()
    for d in range(3):
        cursor = cursor.copy()
        cursor[np.arange(len(points)),order[:,d]] += 1
        nodes.append(cursor)
    ids = np.column_stack([np.ravel_multi_index(n.T,tuple(map(len,axes))) for n in nodes])
    return coo_matrix((weights.ravel(),(np.repeat(np.arange(len(points)),4),ids.ravel())),
                      shape=(len(points),np.prod([len(a) for a in axes]))).tocsr()


@dataclass
class StrictPWA:
    case: str
    axes: tuple
    values: np.ndarray
    metadata: dict

    @property
    def region_count(self):
        return int(6*np.prod([len(a)-1 for a in self.axes]))

    def save(self, path):
        np.savez_compressed(path, case=self.case, C=self.axes[0], h=self.axes[1], I=self.axes[2],
                            values=self.values, metadata=json.dumps(self.metadata))

    @classmethod
    def load(cls, path):
        with np.load(path,allow_pickle=False) as z:
            return cls(str(z['case']),tuple(z[k].copy() for k in ['C','h','I']),
                       z['values'].copy(),json.loads(str(z['metadata'])))

    def field(self, points):
        return interpolation_rows(self.axes, points) @ self.values

    def rhs(self, tau, baseline, *, stage_extrapolation=False):
        if not (np.isfinite(tau) and tau > 0 and 0 <= baseline <= 1.3):
            raise ValueError('Require finite tau>0 and baseline in [0,1.3]')
        # Scalar lookup avoids sparse allocations during numerical validation.
        axes = [a.tolist() for a in self.axes]
        strides = [len(axes[1])*len(axes[2]),len(axes[2]),1]
        values = self.values.tolist()
        def rhs(time, x):
            cells, u = [], []
            for d in range(3):
                a, v = axes[d], x[d]
                # Explicit RK trial stages can leave an invariant box before a
                # step is rejected. Only the validation solver opts into an
                # affine extension for stages; its accepted outputs are checked.
                if not np.isfinite(v) or (not stage_extrapolation and
                                         (v < a[0]-1e-10 or v > a[-1]+1e-10)):
                    raise ValueError(f'State left fitted box on axis {d}: {v}')
                j = max(0,min(bisect.bisect_right(a,v)-1,len(a)-2))
                cells.append(j)
                u.append((v-a[j])/(a[j+1]-a[j]))
            p = sorted(range(3),key=lambda d:-u[d])
            weights = [1-u[p[0]],u[p[0]]-u[p[1]],u[p[1]]-u[p[2]],u[p[2]]]
            index = sum(j*s for j,s in zip(cells,strides))
            f0, f1 = weights[0]*values[index][0], weights[0]*values[index][1]
            for r in range(3):
                index += strides[p[r]]
                f0 += weights[r+1]*values[index][0]
                f1 += weights[r+1]*values[index][1]
            return [f0,f1,-(x[2]-baseline)/tau]
        return rhs

    def region(self, point, tau, baseline):
        """Return A,b,tetrahedron vertices; valid only while inside this tetrahedron."""
        if not np.isfinite(tau) or tau <= 0 or not 0 <= baseline <= 1.3:
            raise ValueError('Invalid IP3 dynamics')
        point = np.asarray(point)
        interpolation_rows(self.axes,point)  # box check
        cell = np.array([np.clip(np.searchsorted(a,point[d],side='right')-1,0,len(a)-2)
                         for d,a in enumerate(self.axes)])
        lower = np.array([a[cell[d]] for d,a in enumerate(self.axes)])
        width = np.array([a[cell[d]+1]-a[cell[d]] for d,a in enumerate(self.axes)])
        order = np.argsort(-(point-lower)/width,kind='stable')
        node = cell.copy()
        coords, ids = [], []
        for step in range(4):
            coords.append([a[node[d]] for d,a in enumerate(self.axes)])
            ids.append(np.ravel_multi_index(node,tuple(map(len,self.axes))))
            if step < 3:
                node = node.copy()
                node[order[step]] += 1
        coords = np.array(coords)
        affine = np.linalg.solve(np.c_[coords,np.ones(4)],self.values[ids])
        a, b = np.zeros((3,3)),np.zeros(3)
        a[:2], b[:2] = affine[:3].T,affine[3]
        a[2,2], b[2] = -1/tau,baseline/tau
        return a,b,coords


def regional_flow(a, b, state, duration):
    """Exact affine propagator, including singular A; caller must check region exit."""
    if duration < 0:
        raise ValueError('Negative duration')
    augmented = np.zeros((4,4))
    augmented[:3,:3], augmented[:3,3] = a,b
    return (expm(duration*augmented) @ np.r_[state,1.])[:3]


def fit_field(case, refine=False, seed=20261020, trajectory_points=None, ridge=.25):
    """Global shared-vertex least squares, exact LR boundary nodes fixed inward.

    Fit full (Cdot,hdot) at 3 stratified samples per box and optional LR training
    trajectories. Ridge to LR nodal values controls underdetermined regions.
    No holdout points are used. Dynamic row is never fitted.
    """
    axes, events = heuristic_axes(case,refine)
    nodes = vertices(axes)
    truth = lr_field(nodes,case)
    rng = np.random.default_rng(seed)
    lower = np.stack(np.meshgrid(*[a[:-1] for a in axes],indexing='ij'),axis=-1).reshape(-1,3)
    widths = np.stack(np.meshgrid(*[np.diff(a) for a in axes],indexing='ij'),axis=-1).reshape(-1,3)
    points = (lower[:,None,:]+widths[:,None,:]*rng.uniform(.05,.95,(len(lower),3,3))).reshape(-1,3)
    weight = np.ones(len(points))
    if trajectory_points is not None:
        points = np.vstack([points,trajectory_points])
        weight = np.r_[weight,np.full(len(trajectory_points),8.)]
    design = interpolation_rows(axes,points).multiply(np.sqrt(weight)[:,None]).tocsr()
    target = lr_field(points,case)*np.sqrt(weight)[:,None]
    fit = truth.copy()
    solvers = []
    for component in range(2):
        # Fix all box-boundary nodes, ensuring a globally inward field on faces.
        boundary = np.zeros(len(nodes),dtype=bool)
        for d,a in enumerate(axes):
            boundary |= (nodes[:,d]==a[0]) | (nodes[:,d]==a[-1])
        free = ~boundary
        mat = vstack([design[:,free],np.sqrt(ridge)*eye(int(free.sum()))],format='csr')
        residual = target[:,component]-design@truth[:,component]
        rhs = np.r_[residual,np.zeros(free.sum())]
        solution = lsmr(mat,rhs,atol=1e-9,btol=1e-9,maxiter=500)
        if solution[1] not in (1,2,4,5):
            raise RuntimeError(f'Least squares did not converge: {solution[1]}')
        fit[free,component] += solution[0]
        solvers.append(dict(stop=int(solution[1]),iterations=int(solution[2]),residual_norm=float(solution[3])))
    metadata = dict(format_version=1,coordinates=['C','h','I'],class_name='continuous_3D_simplicial_PWA',
                    case=case,refine=refine,seed=seed,ridge=ridge,events=events,
                    nodes=len(nodes),regions=int(6*np.prod([len(a)-1 for a in axes])),
                    free_parameters=int(2*free.sum()),field_training_samples=len(points),
                    training_trajectory_samples=0 if trajectory_points is None else len(trajectory_points),
                    fit_method='shared-vertex weighted field least squares; inward LR boundary nodes fixed',
                    solver=solvers,IP3='exact -(I-baseline)/tau; not fitted',
                    scope='initial candidate; full dynamical acceptance and minimality not established')
    return StrictPWA(case,axes,fit,metadata),StrictPWA(case,axes,truth,{**metadata,'fit_method':'LR vertex interpolation; no least-squares fit'})
