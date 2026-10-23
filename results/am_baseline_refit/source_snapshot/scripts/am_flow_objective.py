"""Original RestObjective with full fixed-duration trajectories in its helper.

Residuals, reference samples and independent DOP replay remain unchanged.
Only the approximate numerical backend is replaced with cached affine flows.
"""
import numpy as np
from overnight_coordinate_grid import RestObjective
from pwl_nest.fit_flow import FlowDynamic


class FlowRestObjective(RestObjective):
    def __init__(self,out,case,seed,native_maxdt=.02):
        super().__init__(out,case,seed,native_maxdt=native_maxdt)
        self.dynamic=FlowDynamic()
        self.original_static=self.static
        self.static_length=len(self.original_static(seed))
        self.static=self.full_static

    def full_static(self,model,replay=None,parts=False):
        if replay is None:
            replay={}
            try:
                for p in self.spec['fixed']:
                    count=int(np.ceil(p['duration']/.2))+1
                    latent=self.dynamic.run(model,p['initial'],p['duration'],p['tau'],p['baseline'],
                        outputs=count,maxdt=self.native_maxdt)
                    if latent is None:raise ValueError('Affine fitting trajectory left domain')
                    t=np.linspace(0,p['duration'],count)
                    ip3=p['baseline']+(p['initial'][2]-p['baseline'])*np.exp(-t/p['tau'])
                    x=model.to_physical(np.column_stack([latent,ip3]))
                    replay[p['name']]=(t,x,{})
            except (ValueError,RuntimeError,FloatingPointError,OverflowError):
                penalty=np.full(self.static_length,100.)
                return (penalty,{'failed_full_static':float(penalty@penalty)}) if parts else penalty
        return self.original_static(model,replay=replay,parts=parts)
