#!/usr/bin/env python3
"""Summarize held-out cycle errors for the six local paper candidates."""
import csv, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=json.loads((ROOT/'paper/local_results/physical_no_fold_multistart/summary.json').read_text())
out=ROOT/'paper/local_results/cycle_errors.csv'
columns=['mode','coordinate','split','total','valid','mean_abs_Cmin','mean_abs_Cmax','mean_abs_period','mean_abs_shape_rmse']
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
    for mode in ('am','afm','fm'):
        for coord,label in [('physical','new_rank_01'),('mixed','frozen_mixed')]:
            for split in ('train','holdout'):
                stats=s['validation'][mode][label]['metrics'][split]
                w.writerow(dict(mode=mode,coordinate=coord,split=split,**{k:stats[k] for k in columns[3:]}))
print('Saved',out)
