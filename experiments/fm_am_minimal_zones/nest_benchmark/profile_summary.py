#!/usr/bin/env python3
"""Summarize normal and instrumented NEST runs; instrumentation is inclusive."""
import json
import statistics
import sys
from pathlib import Path


def load(root,mode,name):
    matches=list((root/mode).glob('nest_*/'+name+'.json'))
    if len(matches)!=1:raise ValueError(f'Expected exactly one {mode}/{name}.json; found {matches}')
    return json.loads(matches[0].read_text())


def main():
    root=Path(sys.argv[1]).resolve();out={}
    for name in ('one_thread','multi_thread'):
        normal=load(root,'normal',name);profiled=load(root,'instrumented',name)
        cases=[]
        for item in profiled['timings']:
            p=item['protocol']
            if p['model']=='astrocyte_lr_1994':continue
            samples=[q['pwa_profile'] for q in item['timing_breakdowns']]
            stats={k:statistics.median(s[k] for s in samples) for k in samples[0]}
            cases.append(dict(case=p['case'],connected_sic=p['connected_sic'],median_counters=stats,
                instrumented_wall_seconds=item['median_seconds']))
        out[name]=dict(nodes=normal['nodes'],threads=normal['threads'],resolution_ms=normal['resolution_ms'],
            baseline_speedups=normal['speedups'],profiled=cases)
    out['interpretation']='Nanoseconds are inclusive elapsed intervals summed over calls and threads; categories overlap and instrumentation changes performance. Use baseline_speedups for actual speed.'
    path=root/'profile_summary.json';path.write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out,indent=2));print(f'Saved {path}')


if __name__=='__main__':main()
