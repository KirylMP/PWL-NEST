#!/usr/bin/env python3
"""Read-only FM trial monitor; also works on already-running v5 workers."""
import argparse
import datetime
import json
import statistics
import time
from pathlib import Path
from trial_progress import duration,optimizer_plan


def read_json(path):
    try:return json.loads(path.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def tail_events(path,limit=262144):
    """Bounded read, tolerate an incomplete last line during worker writes."""
    try:
        with path.open('rb') as stream:
            size=stream.seek(0,2);offset=max(0,size-limit);stream.seek(offset)
            lines=stream.read().splitlines()
        if offset:lines=lines[1:]
        events=[]
        for line in lines:
            try:events.append(json.loads(line))
            except (json.JSONDecodeError,UnicodeDecodeError):continue
        return events
    except FileNotFoundError:return []


def trial_line(root,path,plans):
    label=path.name
    result=read_json(path/'result.json')
    if result is not None:
        return f"{label}: DONE elapsed={duration(result['elapsed_seconds'])} loss={result['parts'].get('full_total')}"
    failure=read_json(path/'failure.json')
    if failure is not None:return f"{label}: FAILED {failure.get('error')}"
    state=read_json(root/'logs'/f'{label}.progress.json')
    if state is not None:
        stamp=datetime.datetime.fromisoformat(state['timestamp_utc']).timestamp()
        line=(f"{label}: {state['status']} phase={state['phase']} elapsed={duration(state['trial_elapsed_seconds'])} "
              f"heartbeat_age={duration(time.time()-stamp)} activity={state['activity']}")
        if 'main_budget' in state:
            line+=f" eval={state['evaluation']}/{state['main_budget']}(+{state['recovery_budget']})"
        if state.get('details'):line+=' '+','.join(f'{k}={v}' for k,v in state['details'].items())
        if state.get('work'):
            w=state['work'];line+=f" work={w['completed']}/{w['total']} work_ETA~{duration(w['remaining_seconds'])}"
        if state.get('loss') is not None:line+=f" loss={state['loss']:.7g}"
        if 'main_budget' in state:
            line+=(f" stage_budget_ETA~{duration(state['stage_budget_remaining_seconds'])}"
                   f" optimizer_budget_ETA~{duration(state['optimizer_budget_remaining_seconds'])}")
        return line
    # Older v5 workers have no heartbeat. Progress here concerns the last
    # completed evaluation, not the current position inside a running ODE.
    audit=read_json(path/'training_audit_progress.json')
    if audit is not None:
        age=time.time()-(path/'training_audit_progress.json').stat().st_mtime
        return f"{label}: training_audit I={audit['completed']}/{audit['total']} file_age={duration(age)} (old worker)"
    events=tail_events(path/'iterations.jsonl')
    objectives=[q for q in events if q.get('event')=='objective']
    if not objectives:return f'{label}: waiting for first completed evaluation (old worker; no internal heartbeat)'
    latest=objectives[-1];stage=latest['stage'];count=latest['evaluation']
    same=[q for q in objectives if q['stage']==stage]
    deltas=[b['seconds']-a['seconds'] for a,b in zip(same[:-1],same[1:]) if b['seconds']>a['seconds']]
    pace=statistics.median(deltas[-8:]) if deltas else latest['seconds']/max(1,count)
    age=time.time()-(path/'iterations.jsonl').stat().st_mtime
    line=(f"{label}: last_logged_phase={stage} eval_completed={count} last_completed_loss={latest['loss']:.7g} "
          f"seconds/eval~{pace:.1f} file_age={duration(age)} (old worker)")
    index=next((j for j,p in enumerate(plans) if p['name']==stage),None)
    if index is not None:
        p=plans[index];remaining=max(0,p['budget']+p.get('recovery',0)-count)*pace
        line+=f" stage_budget_ETA~{duration(remaining)}"
        future=plans[index+1:]
        if all(q['kind']==p['kind'] for q in future):
            estimate=remaining+sum((q['budget']+q.get('recovery',0))*q['units']/p['units']*pace for q in future)
            line+=f" optimizer_budget_ETA~{duration(estimate)}"
    return line


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--interval',type=float,default=30.)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if not 1<=args.interval<=3600:parser.error('--interval must be within [1,3600] seconds')
    run=read_json(args.run_dir/'run.json');config=read_json(args.run_dir/'protocol.json')
    if run is None or config is None:parser.error('--run-dir must contain run.json and protocol.json')
    if config.get('cycle_fit',{}).get('section_policy')!='moving_equilibrium':parser.error('this monitor supports v5 FM runs')
    plans=optimizer_plan(config['cycle_fit'],config['cycle_smoothing']['epsilon_schedule'],run['max_nfev'])
    print('ETAs extrapolate optimizer budgets at current throughput; audits/analysis/validation are separate.',flush=True)
    print('They are not convergence forecasts. Old-worker file age does not locate the current ODE step.',flush=True)
    try:
        while True:
            paths=sorted((args.run_dir/'optimizer'/'trials').glob('*'))
            paths=[p for p in paths if p.is_dir()]
            print('\n'+datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
            print(f"Started optimized trials: {sum(p.name.startswith('trial_') for p in paths)}/{run['restarts']}",flush=True)
            for path in paths:print(trial_line(args.run_dir,path,plans),flush=True)
            if args.once:break
            time.sleep(args.interval)
    except KeyboardInterrupt:pass


if __name__=='__main__':main()
