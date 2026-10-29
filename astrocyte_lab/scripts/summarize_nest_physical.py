#!/usr/bin/env python3
"""Collect completed NEST speed and accuracy results into a compact summary."""
import argparse
import json
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);a=p.parse_args()
    root=a.run.resolve();rows=[]
    for mode in ('identity','power'):
        for path in sorted((root/mode).glob('dt_*ms')):
            for benchmark in sorted((path/'benchmark').glob('nest_*')):
                for file,threads in [('one_thread.json',1),('multi_thread.json',None)]:
                    f=benchmark/file
                    if not f.is_file():continue
                    d=json.loads(f.read_text())
                    for s in d.get('speedups',[]):
                        case=s['case'];accuracy=d.get('approximation_vs_LR',{}).get(case,{}).get('errors',{});validation=d.get('validation',{}).get(case,{})
                        rows.append(dict(mode=mode,resolution_ms=d['resolution_ms'],threads=d['threads'],nodes=d['nodes'],case=case,sic=s['connected_sic'],speedup=s['speedup'],LR_seconds=s['LR_seconds'],PWA_seconds=s['PWA_seconds'],C_rmse=accuracy.get('Ca_astro',{}).get('rmse'),SIC_rmse=accuracy.get('I_SIC',{}).get('rmse'),independent_max_error_C_q_I=validation.get('max_abs_error_C_h_I'),source=str(f)))
            for f in (path/'network').glob('all_to_all_*/network_summary.json'):
                d=json.loads(f.read_text())
                for case,v in d.get('results',{}).items():
                    rows.append(dict(mode=mode,resolution_ms=d['settings']['dt'],threads=d['settings']['threads'],case=case,network='all_to_all',neurons=d['settings']['neurons'],astrocytes=d['settings']['neurons']*(d['settings']['neurons']-1),speedup=v['speedup'],LR_seconds=v['median_lr_s'],PWA_seconds=v['median_pwa_s'],source=str(f)))
    result=dict(run=str(root),rows=rows,note='Only completed JSON files. No timing row implies no completed measurement; inspect logs. Speedup = median LR / median PWA. NEST h_IP3R records q in power mode.')
    out=root/'summary.json';out.write_text(json.dumps(result,indent=2)+'\n')
    print('mode     dt(ms) threads case  scenario       speedup  LR(s)    PWA(s)   C_RMSE')
    for r in rows:
        scenario='network' if r.get('network') else ('SIC' if r['sic'] else 'no SIC')
        print(f'{r["mode"]:8s} {r["resolution_ms"]:6g} {r["threads"]:7d} {r["case"]:5s} {scenario:14s} {r["speedup"]:7.3f} {r["LR_seconds"]:8.3f} {r["PWA_seconds"]:8.3f} {str(round(r["C_rmse"],5)) if r.get("C_rmse") is not None else "-"}')
    print(f'Saved {out}; {len(rows)} completed rows')
    if not rows:raise SystemExit('No completed results; inspect benchmark logs.')

if __name__=='__main__':main()
