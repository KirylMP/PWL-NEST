"""Check exact delta-driven IP3 superposition against all saved jump traces."""
import json
import numpy as np
from pwl_nest.models import ROOT
from pwl_nest.impulsive_ip3 import exponential_ip3

OUT=ROOT/'results/coordinate_families';records=[]
for case in ['AM','FM']:
    selection=json.loads((OUT/f'{case}_selection.json').read_text())
    for variant in selection['variants']:
        data=json.loads((OUT/variant/f'{case}.json').read_text())
        protocols={p['name']:p for p in data['historical_diagnostics']}
        for r in data['results']:
            if not r['completed'] or not r.get('jumps'):continue
            p=protocols[r['name']]
            with np.load(OUT/variant/f'{case}_{r["name"]}_historical_diagnostic.npz') as tr:
                expected=exponential_ip3(tr['t'],p['initial'][2],p['tau'],p['baseline'],p['jumps'],side='left')
                error=float(abs(tr['candidate'][:,2]-expected).max())
                if error>2e-12:raise AssertionError('IP3 impulse superposition mismatch')
            jump_error=0.;continuity_error=0.
            for event in r['jumps']:
                for side,field in [('left','physical_before'),('right','physical_after')]:
                    exact=float(exponential_ip3(event['time'],p['initial'][2],p['tau'],p['baseline'],p['jumps'],side=side))
                    jump_error=max(jump_error,abs(event[field][2]-exact))
                continuity_error=max(continuity_error,float(abs(np.array(event['physical_after'][:2])-event['physical_before'][:2]).max()))
            if jump_error>2e-12 or continuity_error>1e-14:raise AssertionError('Impulse reset mismatch')
            records.append(dict(case=case,variant=variant,protocol=r['name'],trace_error=error,
                                jump_error=jump_error,C_h_jump_error=continuity_error))
(OUT/'impulsive_ip3.json').write_text(json.dumps(dict(records=records),indent=2)+'\n')
print(f'Checked {len(records)} impulse-driven traces')
