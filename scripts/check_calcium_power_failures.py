"""Recheck all reported domain/runaway exits with tighter integration."""
import argparse,json,re
from pwl_nest.models import ROOT
from pwl_nest.calcium_coordinates import CalciumTransformPWA,simulate_coordinates

parser=argparse.ArgumentParser()
parser.add_argument('--experiment',choices=['calcium_powers','coordinate_families'],default='calcium_powers')
args=parser.parse_args()
OUT=ROOT/'results'/args.experiment;report=[]
for case in ['AM','FM']:
    variants=json.loads((OUT/f'{case}_selection.json').read_text()).get('variants',['sqrt','square','cube'])
    for kind in variants:
        result=json.loads((OUT/kind/f'{case}.json').read_text())
        protocols={p['name']:p for p in result['training_protocols']+result['historical_diagnostics']+result['new_diagnostics']}
        model=CalciumTransformPWA.load(ROOT/f'models/{args.experiment}/{kind}/{case}.json')
        for record in result['results']:
            if record['completed']:continue
            p=protocols[record['name']];kwargs={k:v for k,v in p.items() if k!='name'}
            try:simulate_coordinates(model,**kwargs,dt=.01,max_step=.01,rtol=2e-11,atol=2e-13)
            except RuntimeError as exc:
                message=str(exc)
                if record['error'].split(': ')[-1]!=message.split(': ')[-1]:raise
                first=float(re.search(r't=([\d.e+-]+)',record['error'])[1])
                refined=float(re.search(r't=([\d.e+-]+)',message)[1])
                if abs(first-refined)>1e-6:raise AssertionError('Exit time failed refinement')
                report.append(dict(case=case,kind=kind,protocol=p,original_exit_time=first,
                                   refined_exit_time=refined,absolute_difference=abs(first-refined),error=message))
            else:raise AssertionError('Reported failed protocol succeeded at tighter tolerances')
(OUT/'failure_checks.json').write_text(json.dumps(dict(max_step=.01,rtol=2e-11,atol=2e-13,failures=report),indent=2)+'\n')
print(f'Confirmed {len(report)} domain/runaway exits')
