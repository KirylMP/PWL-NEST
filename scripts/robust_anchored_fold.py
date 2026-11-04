"""Multiple deterministic starts for the anchored fold shooting problem."""
import copy
import numpy as np
from anchored_lower_fold import anchored_fold


class FoldSolver:
    def __init__(self,base,pivot,cut,reference_x,reference_fold):
        self.base=base;self.pivot=pivot;self.cut=cut
        self.reference_x=np.asarray(reference_x,float)
        self.reference_fold=copy.deepcopy(reference_fold)
        self.cache=[];self.attempts={'reference':0,'nearest':0,'original':0}
        self.successes={'reference':0,'nearest':0,'original':0}
        self.failures=0

    def solve(self,x,changes):
        x=np.asarray(x,float)
        attempts=[('reference',self.reference_fold)]
        if self.cache:
            near=min(self.cache,key=lambda item:float(np.linalg.norm(item[0]-x)))
            attempts.append(('nearest',near[1]))
        attempts.append(('original',None))
        reasons=[]
        for label,hint in attempts:
            self.attempts[label]+=1
            try:
                result=anchored_fold(self.base,changes,self.pivot,self.cut,0.,0.,initial_fold=hint)
                self.successes[label]+=1
                fold=result[3]
                self.cache.append((x.copy(),copy.deepcopy(fold)))
                if len(self.cache)>128:self.cache.pop(0)
                return result
            except (ValueError,np.linalg.LinAlgError,OverflowError,FloatingPointError) as exc:
                reasons.append(label+': '+str(exc))
        self.failures+=1
        raise ValueError('; '.join(reasons))

    def stats(self):
        return {'attempts':dict(self.attempts),'successes':dict(self.successes),'all_failed':self.failures,
                'note':'reference seed tried first; nearest cached and original fold are fallback; numerical failure does not prove nonexistence'}
