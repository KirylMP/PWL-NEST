"""Equal-budget comparison of C**.5, C**2 and C**3 with a common range."""
import argparse
from pwl_nest.models import ROOT
from fit_calcium_coordinates import training_data,run_kind,dump
from validate_calcium_coordinates import validate

KINDS=['sqrt','square','cube']
MODELS=ROOT/'models/calcium_powers'
OUT=ROOT/'results/calcium_powers'


def run(case,budget=120):
    data=training_data(case)
    pairs=[run_kind(case,kind,data,budget,normalize_span=True,models_root=MODELS,results_root=OUT) for kind in KINDS]
    selected,score=min(pairs,key=lambda pair:pair[1])
    selected.save(MODELS/f'selected_{case}.json')
    dump(OUT/f'{case}_selection.json',dict(selected=selected.coordinate.kind,
        training_scores={m.coordinate.kind:s for m,s in pairs},
        selection='training data only; sqrt/square/cube refitted with equal budgets and common coordinate span',
        normalization='phi(Cref)=0, phi(C_MAX)=C_MAX; no fitted normalization parameters'))
    validate(case,[m for m,s in pairs],data[0],data[1],output_root=OUT)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--case',choices=['AM','FM','both'],default='both')
    parser.add_argument('--budget',type=int,default=120)
    args=parser.parse_args()
    for case in (['AM','FM'] if args.case=='both' else [args.case]):run(case,args.budget)
