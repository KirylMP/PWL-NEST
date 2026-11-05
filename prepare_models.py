"""Derive spike-emitting, no-SIC models from NEST 3.10 and the project's PWA."""
from pathlib import Path

root=Path(__file__).resolve().parent/'native'
h=(root/'astrocyte_lr_1994.h').read_text().replace('astrocyte_lr_1994','astrocyte_lr_ca_event').replace('ASTROCYTE_LR_1994_H','ASTROCYTE_LR_CA_EVENT_H')
h=h.replace('#include <gsl/gsl_errno.h>', '#include <cmath>\n#include <gsl/gsl_errno.h>')
c=(root/'astrocyte_lr_1994.cpp').read_text().replace('astrocyte_lr_1994','astrocyte_lr_ca_event')
needle='  void get_status( Dictionary& ) const override;'
assert needle in h
h=h.replace(needle,'''  size_t send_test_event( Node& target, size_t receptor_type, synindex, bool ) override
  {
    SpikeEvent e; e.set_sender( *this ); return target.handles_test_event( e, receptor_type );
  }

'''+needle,1)
needle='private:\n  void init_buffers_() override;'
assert needle in h
h=h.replace(needle,'''private:
  double ca_event_threshold_ = 0.2;
  long ca_event_count_ = 0;
  void init_buffers_() override;''',1)
needle='  P_.get( d );\n  S_.get( d );'
assert needle in h
h=h.replace(needle,needle+'\n  d["ca_event_threshold"] = ca_event_threshold_;\n  d["ca_event_count"] = ca_event_count_;',1)
needle='  Parameters_ ptmp = P_;'
assert needle in h
h=h.replace(needle,'''  double threshold = ca_event_threshold_;
  d.update_value( "ca_event_threshold", threshold );
  if ( not std::isfinite( threshold ) or threshold < 0.0 ) throw BadProperty( "Invalid calcium event threshold" );
  Parameters_ ptmp = P_;''',1)
needle='  P_ = ptmp;\n  S_ = stmp;'
assert needle in h
h=h.replace(needle,needle+'\n  ca_event_threshold_ = threshold;',1)
needle='  , B_( n.B_, *this )'
assert needle in c
c=c.replace(needle,needle+'\n  , ca_event_threshold_( n.ca_event_threshold_ )',1)
needle='nest::astrocyte_lr_ca_event::init_buffers_()\n{'
assert needle in c
c=c.replace(needle,needle+'\n  ca_event_count_ = 0;',1)
needle='nest::astrocyte_lr_ca_event::update( Time const& origin, const long from, const long to )\n{\n  for ( long lag = from; lag < to; ++lag )\n  {'
assert needle in c
c=c.replace(needle,needle+'\n    const double previous_calcium = S_.y_[ State_::Ca_astro ];',1)
start=c.index('    // SIC generation according to Nadkarni & Jung, 2003')
end=c.index('    // log state data',start)
c=c[:start]+'''    if ( previous_calcium < ca_event_threshold_ && S_.y_[ State_::Ca_astro ] >= ca_event_threshold_ )
    {
      SpikeEvent spike;
      kernel().event_delivery_manager.send( *this, spike, lag );
      ++ca_event_count_;
    }

'''+c[end:]
start=c.index('  // send SIC event',start);end=c.index('\n}',start)
c=c[:start]+c[end:]
(root/'astrocyte_lr_ca_event.h').write_text(h);(root/'astrocyte_lr_ca_event.cpp').write_text(c)

p=(root/'nest_pwa_module.cpp').read_text()
p=p.replace('class AstrocytePWA;', 'void register_astrocyte_lr_ca_event(const std::string& name);\nclass AstrocytePWA;')
needle='  void sends_secondary_event(SICEvent&) override {}'
assert needle in p
p=p.replace(needle,'''  size_t send_test_event(Node& target,size_t receptor_type,synindex,bool) override {
    SpikeEvent e;e.set_sender(*this);return target.handles_test_event(e,receptor_type);
  }''',1)
p=p.replace('sic_scale_(n.sic_scale_),sic_th_(n.sic_th_),logger_(*this) {}', 'sic_scale_(n.sic_scale_),sic_th_(n.sic_th_),ca_event_threshold_(n.ca_event_threshold_),logger_(*this) {}')
p=p.replace('    d["input_spikes"]=input_spikes_;', '    d["ca_event_threshold"]=ca_event_threshold_;d["ca_event_count"]=ca_event_count_;\n    d["input_spikes"]=input_spikes_;')
p=p.replace('long ck=c_kind_,hk=h_kind_;', 'long ck=c_kind_,hk=h_kind_;\n    double event_threshold=ca_event_threshold_;\n    d.update_value("ca_event_threshold",event_threshold);\n    if(!std::isfinite(event_threshold)||event_threshold<0) throw BadProperty("Invalid calcium event threshold");')
p=p.replace('    allow_extrapolation_=extrapolate;', '    allow_extrapolation_=extrapolate;ca_event_threshold_=event_threshold;')
p=p.replace('  long input_spikes_=0;', '  double ca_event_threshold_=0.2;\n  long ca_event_count_=0;\n  long input_spikes_=0;')
p=p.replace('    spikes_.clear();logger_.reset();', '    spikes_.clear();logger_.reset();ca_event_count_=0;')
p=p.replace('      solver_.step(y_);', '      const double previous_calcium=calcium();\n      solver_.step(y_);')
p=p.replace('      sic_[lag]=sic_value();', '''      if(previous_calcium<ca_event_threshold_ && calcium()>=ca_event_threshold_) {
        SpikeEvent spike;kernel().event_delivery_manager.send(*this,spike,lag);++ca_event_count_;
      }''')
p=p.replace('    SICEvent e;e.set_coeffarray(sic_);kernel().event_delivery_manager.send_secondary(*this,e);', '')
p=p.replace('register_node_model<AstrocytePWA>("astrocyte_lr_pwa");', 'register_node_model<AstrocytePWA>("astrocyte_pwa_ca_event");register_astrocyte_lr_ca_event("astrocyte_lr_ca_event");')
(root/'nest_ca_event_module.cpp').write_text(p)
print('Generated event models from NEST 3.10 and local PWA sources')
