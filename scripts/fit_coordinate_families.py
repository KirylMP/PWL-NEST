"""Staged search: five C transforms with logit(h), then log/exp(h) for best C."""
import argparse
from pwl_nest.models import ROOT
from fit_calcium_coordinates import training_data,run_kind,dump
from validate_calcium_coordinates import validate

KINDS=['sqrt','exp','log','inverse','inverse_sqrt']
MODELS=ROOT/'models/coordinate_families';OUT=ROOT/'results/coordinate_families'


def run(case,budget=120):
    data=training_data(case)
    pairs=[run_kind(case,kind,data,budget,normalize_span=True,models_root=MODELS,results_root=OUT) for kind in KINDS]
    c_selected,c_score=min(pairs,key=lambda pair:pair[1])
    selected_kind=c_selected.coordinate.kind
    for h_kind in ['log','exp']:
        pairs.append(run_kind(case,selected_kind,data,budget,normalize_span=True,models_root=MODELS,results_root=OUT,h_kind=h_kind))
    selected,score=min(pairs,key=lambda pair:pair[1])
    selected.save(MODELS/f'selected_{case}.json')
    dump(OUT/f'{case}_selection.json',dict(selected=selected.variant,C_stage_selected=selected_kind,
        variants=[m.variant for m,s in pairs],training_scores={m.variant:s for m,s in pairs},
        selection='stage 1: best of five C transforms with logit h; stage 2: log/exp h at selected C; training only',
        normalization='C: phi(Cref)=0,phi(C_MAX)=C_MAX; h: psi(href)=0,psi_prime(href)=logit_prime(href)',
        budgets=dict(least_squares_per_start=budget,refined_starts=2,Powell_maxfev=2400)))
    validate(case,[m for m,s in pairs],data[0],data[1],output_root=OUT)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=['AM','FM','both'],default='both')
    parser.add_argument('--budget',type=int,default=120);args=parser.parse_args()
    for case in (['AM','FM'] if args.case=='both' else [args.case]):run(case,args.budget)
