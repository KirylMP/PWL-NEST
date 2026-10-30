"""Plot measured NEST network timings from the archived paper CSVs."""
import argparse
import csv
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/paper_mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'paper/figures'
COL={'physical':'#2669a2','mixed':'#d17a31'}
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'svg.fonttype':'none'})

def read(path):
    with path.open(newline='') as f:return list(csv.DictReader(f))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--local',action='store_true',help='Use locally recalculated network summaries')
    args=parser.parse_args()
    if args.local:
        network=json.loads((ROOT/'paper/local_results/nest_network/summary.json').read_text())['rows']
        ratio=json.loads((ROOT/'paper/local_results/nest_scaling/summary.json').read_text())['rows']
    else:
        network=read(ROOT/'paper/data/two_threshold_models/network_timings.csv')
        ratio=read(ROOT/'paper/data/astrocyte_neuron_ratio/timings.csv')
    fig,axes=plt.subplots(1,3,figsize=(12,4.2));fig.subplots_adjust(left=.06,right=.985,top=.83,bottom=.29,wspace=.38)
    ax=axes[0];cases=('am','fm','afm');offsets={('physical',10):-.23,('mixed',10):-.08,('physical',1000):.08,('mixed',1000):.23}
    for coordinate in ('physical','mixed'):
        for interval,marker in ((10,'o'),(1000,'s')):
            for j,case in enumerate(cases):
                matches=[r for r in network if r['mode']==case and r['coordinate']==coordinate and float(r['record_interval_ms'])==interval]
                if len(matches)!=1:raise ValueError(f'Expected one network row for {case}/{coordinate}/{interval}')
                r=matches[0];ax.scatter(j+offsets[(coordinate,interval)],float(r['lr_over_pwa']),marker=marker,s=35,color=COL[coordinate],zorder=3)
    ax.set_xticks(range(3),['AM','FM','AFM']);ax.set_ylabel('Speedup: LR time / PWA time');ax.set_title('a   10 neurons, 90 astrocytes',loc='left',fontsize=9);ax.grid(axis='y',alpha=.2)
    if not args.local: ax.set_ylim(1,3.6)
    coord_handles=[Line2D([],[],marker='o',ls='',color=COL[c],label=label) for c,label in [('physical','Physical C'),('mixed','Mixed C')]]
    interval_handles=[Line2D([],[],marker='o',ls='',color='.25',label='10 ms'),Line2D([],[],marker='s',ls='',color='.25',label='1000 ms')]
    legend=ax.legend(handles=coord_handles,title='PWA coordinate',loc='upper right' if args.local else 'upper left',fontsize=7,title_fontsize=7,frameon=False);ax.add_artist(legend)
    ax.legend(handles=interval_handles,title='Recording interval',loc='lower left' if args.local else 'lower right',fontsize=7,title_fontsize=7,frameon=False)
    ax=axes[1]
    for coordinate,label in [('physical','Physical C'),('mixed','Mixed C')]:
        rs=sorted([r for r in ratio if r['coordinate']==coordinate],key=lambda r:int(r['ratio']))
        ax.plot([int(r['ratio']) for r in rs],[float(r['speedup']) for r in rs],'-o',color=COL[coordinate],lw=1.6,ms=4,label=label)
    ax.axhline(1,color='.45',ls=':',lw=.8);ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],['1','2','4','8','16'])
    if not args.local: ax.set_ylim(1,5.8)
    ax.set(xlabel='Astrocytes per neuron',ylabel='Speedup: LR time / PWA time');ax.set_title('b   Scaling: 100-neuron FM network',loc='left',fontsize=9);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=7,loc='upper left')
    ax=axes[2]
    for coordinate,label,style in [('physical','Physical C','-'),('mixed','Mixed C','--')]:
        rs=sorted([r for r in ratio if r['coordinate']==coordinate],key=lambda r:int(r['ratio']));x=[int(r['ratio']) for r in rs]
        if not args.local or coordinate=='physical':
            ax.plot(x,[float(r['lr_median_s']) for r in rs],color='.45',ls=style,lw=1.4,marker='.',ms=4,label='LR baseline' if args.local else 'LR baseline ('+label+')')
        ax.plot(x,[float(r['pwa_median_s']) for r in rs],color=COL[coordinate],ls='-',lw=1.6,marker='o',ms=4,label='PWA '+label)
    ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16],['1','2','4','8','16']);ax.set_ylim(bottom=0)
    if not args.local: ax.set_ylim(0,27)
    ax.set(xlabel='Astrocytes per neuron',ylabel='Median network runtime (s)');ax.set_title('c   Scaling: wall-clock times',loc='left',fontsize=9);ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=6.5,loc='upper left')
    fig.suptitle('NEST network timing: Li–Rinzel and two-threshold PWA',fontsize=11)
    if args.local:
        first=network[0]
        footer=(f"{first['duration_s']:g} s biological time  •  dt = {first['dt_ms']:g} ms  •  "
                f"{first['threads']} threads  •  SIC enabled  •  medians of {len(first['lr_runs_s'])} same-seed runs")
    else:
        footer='60 s biological time  •  dt = 0.1 ms  •  16 threads  •  SIC enabled  •  medians of 3 same-seed runs'
    fig.text(.5,.11,footer,ha='center',fontsize=8)
    for ext in ('pdf','svg','png'):fig.savefig(OUT/f'figure7_nest_speed.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    print('Saved',OUT/'figure7_nest_speed.{pdf,svg,png}')

if __name__=='__main__':main()
