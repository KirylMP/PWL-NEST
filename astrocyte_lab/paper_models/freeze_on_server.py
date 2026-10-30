#!/usr/bin/env python3
"""Create a new immutable-by-convention six-model snapshot from tracked inputs."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
DEFAULT='astrocyte_lab/runs/fm2_peak_20261029T175253Z_4BaiP8/comparison/rank_01'
ROOT=HERE.parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--fm-mixed',type=Path,default=ROOT/DEFAULT);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 sources={f'{mode}_{coord}':HERE/f'{mode}_{coord}' for mode in ('fm','afm','am') for coord in ('physical','mixed') if (mode,coord)!=('fm','mixed')}
 sources['fm_mixed']=a.fm_mixed.resolve();expected={'fm':.05,'afm':.075,'am':.1}
 for name,folder in sources.items():
  for file in ('parameters.json','validation.json','comparison.png'):
   if not (folder/file).is_file():raise FileNotFoundError(folder/file)
  data=json.loads((folder/'parameters.json').read_text());mode,coord=name.split('_')
  if abs(data['K']-expected[mode])>1e-9:raise ValueError(f'Wrong K in {folder}')
  if coord=='mixed' and data.get('coordinate_family')!='mixed':raise ValueError(f'Expected mixed model: {folder}')
  if coord=='physical' and data.get('observable_calcium')!='physical C':raise ValueError(f'Expected direct physical C: {folder}')
  if coord=='mixed' and data.get('status')!='training_cycles_found':raise ValueError(f'Mixed candidate is not training_cycles_found: {folder}')
 out=a.out.resolve()
 if out.exists():raise FileExistsError(f'Choose a new destination; existing snapshot is not overwritten: {out}')
 out.mkdir(parents=True)
 commit=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
 manifest={'description':'Six frozen two-threshold models for paper; hashes identify exact files, no modification protection.', 'code_commit':commit, 'models':{}}
 for name,folder in sorted(sources.items()):
  dest=out/name;dest.mkdir();hashes={}
  for file in ('parameters.json','validation.json','comparison.png'):
   src=folder/file;shutil.copy2(src,dest/file);hashes[file]=hashlib.sha256((dest/file).read_bytes()).hexdigest()
  manifest['models'][name]={'source':str(folder),'K_ER':expected[name.split('_')[0]],'sha256':hashes}
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print('FROZEN',out)
if __name__=='__main__':main()
