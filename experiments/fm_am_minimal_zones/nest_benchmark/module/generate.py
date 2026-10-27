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
            s=s.replace('  Parameters_ P_;\n  State_ S_;\n  Buffers_ B_;',
                '  Parameters_ P_;\n  State_ S_;\n  Buffers_ B_;\n  double latent_z_ = 0.;\n  mutable double calcium_hint_ = .073;\n  bool latent_ready_ = false;\n  double calcium_physical_() const;\n  void reset_latent_from_calcium_();')
            s=s.replace('    return S_.y_[ elem ];','    return elem == State_::Ca_astro ? calcium_physical_() : S_.y_[ elem ];')
            s=s.replace('  S_.get( d );','  State_ physical = S_;\n  physical.y_[ State_::Ca_astro ] = calcium_physical_();\n  physical.get( d );')
            s=s.replace('  State_ stmp = S_;', '  State_ stmp = S_;\n  stmp.y_[ State_::Ca_astro ] = calcium_physical_();')
            s=s.replace('  P_ = ptmp;\n  S_ = stmp;',
                '  if ( !std::isfinite(stmp.y_[ State_::Ca_astro ]) || stmp.y_[ State_::Ca_astro ] <= 1e-14 || stmp.y_[ State_::Ca_astro ] >= 2.0/1.185 )\n    throw BadProperty( "PWA calcium must lie in the physical interval" );\n  P_ = ptmp;\n  S_ = stmp;\n  reset_latent_from_calcium_();')
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
            s=s.replace('  , B_( n.B_, *this )\n', '  , B_( n.B_, *this )\n  , latent_z_( n.latent_z_ )\n  , calcium_hint_( n.calcium_hint_ )\n  , latent_ready_( n.latent_ready_ )\n')
            s=s.replace('  B_.logger_.init();\n}', '  B_.logger_.init();\n  if ( !latent_ready_ ) reset_latent_from_calcium_();\n  if ( pwa_prepare( pwa_config, P_.IP3_0_, P_.tau_IP3_ / 1000.0, B_.step_ / 1000.0, P_.SIC_th_ ) != 0 )\n    throw BadProperty( "Invalid PWA solver configuration" );\n}')
            marker=f'inline void\nnest::{class_name}::update('
            helper=f'''double
nest::{class_name}::calcium_physical_() const
{{
  if ( !latent_ready_ ) return S_.y_[ State_::Ca_astro ];
  if ( pwa_to_c( pwa_config, latent_z_, calcium_hint_ ) != 0 )
    throw KernelException( "PWA calcium inversion failed" );
  return calcium_hint_;
}}

void
nest::{class_name}::reset_latent_from_calcium_()
{{
  double z;
  if ( pwa_to_z( pwa_config, S_.y_[ State_::Ca_astro ], z ) != 0 )
    throw BadProperty( "PWA calcium must lie in the physical interval" );
  latent_z_ = z;
  calcium_hint_ = S_.y_[ State_::Ca_astro ];
  latent_ready_ = true;
}}

'''
            if marker not in s:raise ValueError('Upstream update marker changed')
            s=s.replace(marker,helper+marker)
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
    const int pwa_status = pwa_advance_z( pwa_config,
      latent_z_, S_.y_[ State_::h_IP3R ], S_.y_[ State_::IP3 ],
      P_.IP3_0_, P_.tau_IP3_ / 1000.0, B_.step_ / 1000.0 );
    if ( pwa_status != 0 )
      throw KernelException( std::string( "PWA astrocyte integration failed, status=" ) + std::to_string( pwa_status ) );

'''
            s=s[:begin]+replacement+s[end:]
            begin=s.index('    const double calc_thr =',s.index(f'nest::{class_name}::update('))
            end=s.index('    B_.sic_values[ lag ] = sic_value;',begin)+len('    B_.sic_values[ lag ] = sic_value;')
            s=s[:begin]+'''    double sic_value;
    if ( pwa_sic( pwa_config, latent_z_, P_.SIC_th_, P_.SIC_scale_, calcium_hint_, sic_value ) != 0 )
      throw KernelException( "PWA SIC evaluation failed" );
    B_.sic_values[ lag ] = sic_value;'''+s[end:]
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
