#!/usr/bin/env python3
"""Summarize saved results without presenting unfinished stages as final."""
import csv,json,sys
from pathlib import Path
p=Path(__file__).resolve().parent/'results'
rows=[]
for variant in ('base','region1','region2','region3','all'):
    path=p/variant/'candidate.json'
    if not path.exists():
        rows.append(dict(variant=variant,stage='missing'));continue
    c=json.loads(path.read_text());parts=c.get('loss_components',{})
    row=dict(variant=variant,stage=c.get('stage'),thresholds=len(c['physical_thresholds_C']),loss=c.get('loss'),components=parts,training_settings=c.get('training_settings'),fold_constraint=c.get('fold_constraint'))
    rows.append(row)
p.mkdir(parents=True,exist_ok=True)
(p/'comparison.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
if any(row['stage']!='closed' for row in rows):sys.exit('Some variants do not have a closed-stage result; scores from different stages are not comparable.')

settings=[row.get('training_settings') for row in rows]
if any(s is None or s!=settings[0] for s in settings):sys.exit('Training settings differ; scores are not comparable.')
