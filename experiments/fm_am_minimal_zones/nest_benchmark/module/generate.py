#!/usr/bin/env python3
"""Generate two NEST 3.10 astrocyte models from selected fitted JSON files."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from latent_flow import Candidate,ROOT


def literal(v):
    v=float(v)
    if not np.isfinite(v):raise ValueError('Non-finite coefficient')
    return format(v,'.17g')


def build(case,path):
    model=Candidate(path);orig=json.loads(Path(path).read_text())
    expected={'am':2,'fm':3}[case]
    if len(model.thresholds)!=expected:raise ValueError(f'{case} requires {expected} physical thresholds')
    class_name=f'astrocyte_{case}_pwa';out=HERE/'generated';out.mkdir(exist_ok=True)
    raw=Path(path).read_bytes();sha=hashlib.sha256(raw).hexdigest()
    transform=model.transform or {};vals=[transform.get(k,default) for k,default in [('p',0),('e',0),('n',1),('k',0),('C_ref',.5)]]
    # Current candidate family uses the fixed normalized C_ref=.5 map.
    if abs(float(vals[4])-.5)>1e-14:raise ValueError('Unexpected C_ref')
    txt='#pragma once\n#include "bridge.h"\n'
    txt+='static const double pwa_thresholds[] = {'+','.join(map(literal,model.z_thresholds))+'};\n'
    txt+='static const double pwa_coeff[] = {'+','.join(map(literal,model.coefficients.ravel()))+'};\n'
    txt+='static const PwaConfig pwa_config = {'+str(len(model.thresholds)+1)+',pwa_thresholds,pwa_coeff,'+','.join(map(literal,vals))+','+('false' if model.transform else 'true')+'};\n'
    (out/f'{case}_constants.h').write_text(txt)
    for ext in ('h','cpp'):
        s=(HERE/'upstream'/f'astrocyte_lr_1994.{ext}').read_text()
        s=s.replace('astrocyte_lr_1994',class_name).replace('ASTROCYTE_LR_1994',class_name.upper())
        s='/* Modified 2026-10-27 from NEST 3.10 astrocyte_lr_1994 for fitted PWA propagation; see module/COPYING. */\n'+s
        if ext=='h':
            for header in ('gsl_errno.h','gsl_matrix.h','gsl_odeiv.h','gsl_sf_exp.h'):
                s=s.replace(f'#include <gsl/{header}>\n','')
            s=s.replace(f'extern "C" int {class_name}_dynamics( double, const double*, double*, void* );\n','')
            s=s.replace(f'  friend int {class_name}_dynamics( double, const double*, double*, void* );\n','')
            start=s.index('    /** GSL ODE stuff */')
            end=s.index('    double step_;',start)
            s=s[:start]+s[end:]
            s=s.replace('    double IntegrationStep_;  //!< current integration time step, updated by GSL\n','')
            needle='  ptmp.set( d, this );        // throws if BadProperty'
            replacement=needle+'''
  // Fitted field coefficients are embedded at build time. LR biophysical
  // parameters cannot retune it via set_status; reject misleading changes.
  if ( ptmp.Ca_tot_ != P_.Ca_tot_ || ptmp.Kd_IP3_1_ != P_.Kd_IP3_1_ ||
       ptmp.Kd_IP3_2_ != P_.Kd_IP3_2_ || ptmp.Kd_act_ != P_.Kd_act_ ||
       ptmp.Kd_inh_ != P_.Kd_inh_ || ptmp.Km_SERCA_ != P_.Km_SERCA_ ||
       ptmp.k_IP3R_ != P_.k_IP3R_ || ptmp.rate_IP3R_ != P_.rate_IP3R_ ||
       ptmp.rate_L_ != P_.rate_L_ || ptmp.rate_SERCA_ != P_.rate_SERCA_ ||
       ptmp.ratio_ER_cyt_ != P_.ratio_ER_cyt_ )
    throw BadProperty( "PWA biophysical coefficients are compiled from the selected fit; rebuild to change them" );'''
            if needle not in s:raise ValueError('Upstream set_status structure changed')
            s=s.replace(needle,replacement)
        if ext=='cpp':
            start=s.index('extern "C" int\n'+class_name+'_dynamics')
            end=s.index('\n}\n}\n\n/*',start)
            s=s[:start]+s[end+3:]  # preserve the namespace closing brace
            s=s.replace('  , s_( nullptr )\n  , c_( nullptr )\n  , e_( nullptr )\n','')
            start=s.index(f'nest::{class_name}::~{class_name}()')
            end=s.index('\n/* ----------------------------------------------------------------',start)
            s=s[:start]+f'nest::{class_name}::~{class_name}() = default;\n'+s[end:]
            start=s.index('  B_.IntegrationStep_ = B_.step_;')
            end=s.index('  B_.J_noise_ = 0.0;',start)
            s=s[:start]+s[end:]
            if case=='fm':
                s=s.replace('Km_SERCA_( 0.1 )', 'Km_SERCA_( 0.05 )')
            target=f'#include "{class_name}.h"'
            s=s.replace(target,target+f'\n#include "bridge.h"\n#include "{case}_constants.h"')
            begin=s.index('    double t = 0.0;',s.index(f'nest::{class_name}::update('))
            end=s.index('    // this is to add the incoming spikes to IP3',begin)
            replacement='''    // Fixed fitted PWA flow uses seconds; NEST resolution and tau are ms.
    // Physical CurrentEvent forcing is not part of this experimental model.
    if ( B_.J_noise_ != 0.0 )
      throw KernelException( "PWA astrocyte does not support CurrentEvent input" );
    const int pwa_status = pwa_advance( pwa_config,
      S_.y_[ State_::Ca_astro ], S_.y_[ State_::h_IP3R ], S_.y_[ State_::IP3 ],
      P_.IP3_0_, P_.tau_IP3_ / 1000.0, B_.step_ / 1000.0 );
    if ( pwa_status != 0 )
      throw KernelException( std::string( "PWA astrocyte integration failed, status=" ) + std::to_string( pwa_status ) );

'''
            s=s[:begin]+replacement+s[end:]
        (out/f'{class_name}.{ext}').write_text(s)
    return dict(path=str(Path(path).resolve()),sha256=sha,physical_thresholds_C=model.thresholds.tolist(),coefficients_from='alpha,beta,a,b,transform',case=case)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--am',type=Path,default=ROOT/'experiments/fm_am_minimal_zones/am_two_thresholds/fits/mixed_power_fixed/candidate.json')
    p.add_argument('--fm',type=Path,default=ROOT/'experiments/fm_am_minimal_zones/fm_four_zone_fit/candidate.json')
    args=p.parse_args()
    report={k:build(k,v) for k,v in [('am',args.am),('fm',args.fm)]}
    shutil.copyfile(ROOT/'native/pwa_event.cpp',HERE/'pwa_event.cpp')
    (HERE/'generated'/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
