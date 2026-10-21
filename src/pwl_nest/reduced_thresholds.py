"""Continuous compact C strips with derived exact background equilibrium.

Coefficients and ordered thresholds are fitted directly. The old sequence of
spectral region labels is deliberately not imposed on a smaller model.
"""
import copy
import numpy as np
from scipy.optimize import lsq_linear
from .calcium_coordinates import CalciumTransformPWA
from .calcium_strips import CalciumStrips
from .thresholds import OrderedThresholds


class ReducedFamily:
    def __init__(self, source, thresholds_C, rest, points, lower, upper,
                 target='incumbent',keep_input=False):
        self.source=source
        self.rest=np.asarray(rest,dtype=float)
        self.reference=np.asarray(thresholds_C,dtype=float)
        self.mapping=OrderedThresholds(self.reference,lower,upper)
        self.knots=source.input_thresholds.copy() if keep_input else np.empty(0)
        self.m=len(self.reference);self.features=3+self.m+len(self.knots)
        self.rest_y=source.to_latent(self.rest)
        y=source.to_latent(points)
        theta=source.coordinate.forward(self.reference)
        design=self.design(y,theta)-self.design(self.rest_y,theta)
        truth=source.field(y)-source.field(self.rest_y) if target=='incumbent' else source.source_field(y)
        weights=np.column_stack([1/source.coordinate.derivative(points[:,0]),
                                 1/source.gate.derivative(points[:,1])])/[.2,.05]
        rows=[]
        for row in range(2):
            if target=='incumbent':
                fixed=[1]+list(range(3+self.m,self.features))
                values=np.r_[source.latent.matrix[row,2],source.input_weights[row] if keep_input else []]
                free=np.array([j for j in range(self.features) if j not in fixed])
                adjusted=truth[:,row]-design[:,fixed]@values
                solution=lsq_linear(design[:,free]*weights[:,row,None],adjusted*weights[:,row],
                    bounds=(-500.,500.),tol=1e-10,max_iter=100)
                coefficients=np.empty(self.features);coefficients[fixed]=values;coefficients[free]=solution.x
                rows.append(coefficients)
                continue
            lo=np.full(self.features,-500.);hi=np.full(self.features,500.)
            if row==0:lo[1]=1e-5
            else:hi[1]=-1e-5
            solution=lsq_linear(design*weights[:,row,None],truth[:,row]*weights[:,row],
                bounds=(lo,hi),tol=1e-10,max_iter=100)
            rows.append(solution.x)
        raw=np.asarray(rows).ravel()
        # Ensure the prescribed equilibrium is attracting at initialization.
        # This alters a seed only; the fitted region matrices stay unrestricted.
        raw=self.repair_rest(raw,theta)
        encoded=raw.copy()
        encoded[1]=np.log(raw[1]);encoded[self.features+1]=np.log(-raw[self.features+1])
        self.center=encoded;self.scale=np.maximum(1.,np.abs(encoded))
        self.size=len(raw)+self.m
        self.lower=np.r_[np.full(len(raw),-2.),np.full(self.m,-np.log(4))]
        self.upper=np.r_[np.full(len(raw),2.),np.full(self.m,np.log(4))]
        for index in [1,self.features+1]:
            self.lower[index]=max(self.lower[index],(-12-encoded[index])/self.scale[index])
            self.upper[index]=min(self.upper[index],(6-encoded[index])/self.scale[index])
        self.target=target

    def design(self,y,theta):
        y=np.asarray(y)
        return np.concatenate([y,np.maximum(0,y[...,0,None]-theta),
            np.maximum(0,y[...,2,None]-self.knots)],axis=-1)

    def repair_rest(self,raw,theta):
        rows=raw.reshape(2,self.features).copy()
        active=self.rest_y[0]>theta
        a=rows[:,:2].copy();a[:,0]+=rows[:,3:3+self.m][:,active].sum(axis=1)
        # Raise restoring determinant if projection created a saddle at rest.
        desired_trace=min(-.03,float(np.trace(a)))
        rows[0,0]+=desired_trace-np.trace(a)
        a[0,0]+=desired_trace-np.trace(a)
        determinant=float(np.linalg.det(a))
        if determinant<.002:
            rows[1,0]-=(.002-determinant)/rows[0,1]
        return rows.ravel()

    def make(self,z,check_rest=True):
        z=np.asarray(z,dtype=float)
        if z.shape!=(self.size,) or not np.isfinite(z).all():raise ValueError('Invalid reduced parameters')
        encoded=self.center+z[:len(self.center)]*self.scale
        rows=encoded.reshape(2,self.features).copy()
        rows[0,1]=np.exp(encoded[1]);rows[1,1]=-np.exp(encoded[self.features+1])
        physical=self.mapping.decode(z[len(self.center):])
        theta=self.source.coordinate.forward(physical)
        intercept=-rows@self.design(self.rest_y,theta)
        matrix=np.column_stack([intercept,rows[:,:3+self.m]])
        metadata=dict(class_name='reduced_continuous_transformed_C_strips',
            physical_thresholds_C=physical.tolist(),enforce_types=False,accepted=False,
            exact_rest_constraint=self.rest.tolist(),original_spectral_sequence_preserved=False,
            threshold_count=self.m,input_threshold_count=len(self.knots),
            regions=(self.m+1)*(len(self.knots)+1),active_fit_parameters=self.size,
            parameter_count=self.size,field_initializer=self.target,
            reference_thresholds_C=self.reference.tolist(),direct_fit_vector=z.tolist(),
            coordinates=['u','v','I'],IP3='exact -(I-baseline)/tau',calcium_clipping=False,
            scope='compact approximation; validate trajectories, SIC and speed separately')
        latent=CalciumStrips(self.source.case,theta,matrix,metadata)
        model=CalciumTransformPWA(latent,self.source.coordinate,self.source.h_reference,
            self.source.h_kind,self.knots,rows[:,3+self.m:])
        if check_rest:
            a,_,_=model.region(self.rest_y,20.,self.rest[2])
            if np.max(np.linalg.eigvals(a[:2,:2]).real)>=-1e-7:
                raise ValueError('Prescribed rest is not stable')
        return model
