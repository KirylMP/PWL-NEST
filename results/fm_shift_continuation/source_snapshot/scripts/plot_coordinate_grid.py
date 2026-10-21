"""Standalone publication exports; blank cells never masquerade as low error."""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'.cache/matplotlib'))
os.environ.setdefault('XDG_CACHE_HOME',str(Path(__file__).resolve().parents[1]/'.cache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(out):
    config=json.loads((out/'config.json').read_text());cs,hs=config['c_kinds'],config['h_kinds']
    for case in config['cases']:
        fig,axes=plt.subplots(1,3,figsize=(max(12,len(hs)*2.3),max(4,len(cs)*.48)),constrained_layout=True)
        for ax,key,title in zip(axes,['test_C_RMSE','test_phase_aligned_C_RMSE','verified_train_loss'],
                ['Held-out causal C RMSE (uM)','Held-out aligned C RMSE (uM)','Verified training loss']):
            matrix=np.full((len(cs),len(hs)),np.nan);records={}
            for i,c in enumerate(cs):
                for j,h in enumerate(hs):
                    path=out/'best'/case/f'{c}__{h}.json'
                    if path.exists():
                        r=json.loads(path.read_text());records[i,j]=r
                        if r.get(key) is not None:matrix[i,j]=r[key]
            cmap=plt.get_cmap('viridis_r').copy();cmap.set_bad('#eeeeee')
            handle=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',cmap=cmap)
            for i in range(len(cs)):
                for j in range(len(hs)):
                    r=records.get((i,j));v=matrix[i,j]
                    text='—' if r is None else 'FAIL' if not np.isfinite(v) else f'{v:.3g}'
                    if r is not None and (not r.get('test_all_physical',False) or r.get('test_peak_count_matches')!=4):text+='!'
                    ax.text(j,i,text,ha='center',va='center',fontsize=8,color='black',bbox=dict(facecolor='white',alpha=.65,edgecolor='none',pad=.5))
            ax.set_xticks(range(len(hs)),hs,rotation=35,ha='right');ax.set_yticks(range(len(cs)),cs)
            ax.set(xlabel='h coordinate',ylabel='C coordinate',title=title);fig.colorbar(handle,ax=ax,shrink=.7)
        fig.suptitle(f'{case}: train-selected candidates; ! = physical or peak-count failure; — = no verified model')
        for suffix in ['png','pdf']:fig.savefig(out/f'{case}_coordinate_grid.{suffix}',dpi=180)
        plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');args=p.parse_args();plot(Path(args.output).resolve())
