#!/usr/bin/env python3
"""Check exact rest, event constraints and inherited types on perturbed seeds."""
import argparse
import json
import numpy as np
from model import Model


def check(model,x):
    m=model.unpack(x);C,h=model.rest;z=float(model.f.phi(C,m['shape']))
    field=m['co']@np.r_[1.,z,h,0.,np.maximum(z-m['padded'],0.)]
    assert np.max(abs(field))<1e-9,('rest field',field)
    for j,I in zip(model.event_indices,model.event_I):
        b=model.basis(model.cuts[j],I,m)
        assert abs(float(b@(m['N']-m['H'])))<1e-10,('fixed event',j,I)
    if model.case=='am':
        assert all((m['T'][r]>0 and m['D'][r]>m['T'][r]**2/4) if k==1 else (m['T'][r]<0 and m['D'][r]>0) for r,k in enumerate(model.kind))
    else:
        for r,k in enumerate(model.kind):
            T,D=m['T'][r],m['D'][r]
            assert (T<0 and 0<D<T*T/4) if k==0 else (D<0 if k==1 else (T>0 and D>T*T/4 if k==2 else T<0 and D>T*T/4))


def main():
    p=argparse.ArgumentParser();p.add_argument('case',choices=['am','fm']);p.add_argument('--parent');args=p.parse_args()
    model=Model(args.case);parent=args.parent or model.default_parent
    base=model.lift_parent(json.loads(open(parent).read()));check(model,base)
    rng=np.random.default_rng(53);valid=0
    for _ in range(80):
        x=base.copy();x[model.active]+=rng.normal(0,.001,len(model.active))
        try:x=model.constrain_events(x)
        except ValueError:continue
        check(model,x);valid+=1
    if valid<10:raise RuntimeError(f'Only {valid} feasible perturbations checked')
    print(f'{args.case}: rest, fixed events and inherited types verified for seed and {valid} perturbations')


if __name__=='__main__':main()
