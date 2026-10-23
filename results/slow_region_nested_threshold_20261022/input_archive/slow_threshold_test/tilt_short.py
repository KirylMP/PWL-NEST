from search_slow import *
import json,time
res=json.load(open('search_results.json'))
bounds=[((.031-CREf)/S,(.042-CREf)/S),(-.025,.025),(-4.,4.),(-3.,3.),(-5.,5.)]
opt=differential_evolution(lambda x:score(x,2),bounds,popsize=4,maxiter=6,polish=False,seed=2325)
res['tilt']=summarize(opt.x,2)
open('search_results.json','w').write(json.dumps(res,indent=2))
print('tilt',opt.fun,opt.x,[(r['I'],r['T']) for r in res['tilt']['rows']])
