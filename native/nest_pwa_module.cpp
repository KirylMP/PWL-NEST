// NEST 3.10 extension. State and coefficients use seconds; NEST uses ms.
// Spike delivery and SIC follow astrocyte_lr_1994 (NEST, GPL-2.0-or-later).
#include "nest_extension_interface.h"
#include "node.h"
#include "recordables_map.h"
#include "ring_buffer.h"
#include "universal_data_logger_impl.h"
#include "pwa_step.h"

namespace nest {
class AstrocytePWA;
template<> void RecordablesMap<AstrocytePWA>::create();
class AstrocytePWA : public Node {
public:
  using Node::handle;
  using Node::handles_test_event;
  using Node::sends_secondary_event;
  AstrocytePWA(): logger_(*this) {recordables_.create();}
  AstrocytePWA(const AstrocytePWA& n): Node(n),solver_(n.solver_),y_(n.y_),
    c_ref_(n.c_ref_),c_scale_(n.c_scale_),h_ref_(n.h_ref_),
    c_kind_(n.c_kind_),h_kind_(n.h_kind_),delta_(n.delta_),
    sic_scale_(n.sic_scale_),sic_th_(n.sic_th_),logger_(*this) {}

  void sends_secondary_event(SICEvent&) override {}
  size_t handles_test_event(SpikeEvent&,size_t r) override {check(r);return 0;}
  size_t handles_test_event(DataLoggingRequest& e,size_t r) override {
    check(r);return logger_.connect_logging_device(e,recordables_);
  }
  void handle(SpikeEvent& e) override {
    if (e.get_weight()<0) throw BadProperty("PWA astrocyte requires nonnegative input weights");
    spikes_.add_value(e.get_rel_delivery_steps(kernel().simulation_manager.get_slice_origin()),
      e.get_weight()*e.get_multiplicity());
  }
  void handle(DataLoggingRequest& e) override {logger_.handle(e);}
  void get_status(Dictionary& d) const override {
    d["thresholds"]=solver_.thresholds;d["coefficients"]=solver_.coefficients;
    d["input_thresholds"]=solver_.input_thresholds;
    d["allow_ip3_extrapolation"]=allow_extrapolation_;
    d["IP3_0"]=solver_.baseline;d["tau_IP3"]=solver_.tau*1000.;
    d["delta_IP3"]=delta_;d["SIC_scale"]=sic_scale_;d["SIC_th"]=sic_th_;
    d["c_reference"]=c_ref_;d["c_scale"]=c_scale_;d["h_reference"]=h_ref_;
    d["c_coordinate"]=c_kind_;d["h_coordinate"]=h_kind_;
    d["u"]=y_[0];d["v"]=y_[1];d["IP3"]=y_[2];
    d["Ca_astro"]=calcium();d["h_IP3R"]=gate();
    d["affine_flows"]=solver_.flows;d["region_crossings"]=solver_.crossings;
    d["input_crossings"]=solver_.input_crossings;
    d["input_spikes"]=input_spikes_;d[names::recordables]=recordables_.get_list();
  }
  void set_status(const Dictionary& d) override {
    auto p=solver_; auto y=y_;
    double cr=c_ref_,cs=c_scale_,hr=h_ref_,delta=delta_,ss=sic_scale_,st=sic_th_,tau=p.tau*1000.;
    long ck=c_kind_,hk=h_kind_;
    bool extrapolate=allow_extrapolation_;
    d.update_value("allow_ip3_extrapolation",extrapolate);
    d.update_value("thresholds",p.thresholds);d.update_value("coefficients",p.coefficients);
    d.update_value("input_thresholds",p.input_thresholds);
    d.update_value("IP3_0",p.baseline);d.update_value("tau_IP3",tau);
    d.update_value("delta_IP3",delta);d.update_value("SIC_scale",ss);d.update_value("SIC_th",st);
    d.update_value("c_reference",cr);d.update_value("c_scale",cs);d.update_value("h_reference",hr);
    d.update_value("c_coordinate",ck);d.update_value("h_coordinate",hk);
    d.update_value("u",y[0]);d.update_value("v",y[1]);d.update_value("IP3",y[2]);
    if (p.coefficients.size()!=8*(p.thresholds.size()+1)*(p.input_thresholds.size()+1) || !(tau>0) || !(cr>0&&cs>0&&hr>0&&hr<1)
      || ck<0 || ck>1 || hk<0 || hk>2 || p.baseline<0 || (!extrapolate&&p.baseline>1.3) || delta<0
      || y[2]<0 || (!extrapolate&&y[2]>1.3) || !std::isfinite(tau+cr+cs+hr+delta+ss+st+p.baseline))
      throw BadProperty("Invalid PWA coefficients, coordinates, IP3 or tau");
    for(size_t j=0;j<p.thresholds.size();++j)
      if (!std::isfinite(p.thresholds[j]) || (j && p.thresholds[j]<=p.thresholds[j-1]))
        throw BadProperty("Thresholds must be finite and strictly increasing");
    for(double a:p.coefficients) if(!std::isfinite(a)) throw BadProperty("Nonfinite coefficients");
    for(size_t j=0;j<p.input_thresholds.size();++j)
      if(!std::isfinite(p.input_thresholds[j]) || p.input_thresholds[j]<=0
        || (j&&p.input_thresholds[j]<=p.input_thresholds[j-1])) throw BadProperty("Invalid IP3 thresholds");
    for(double a:y) if(!std::isfinite(a)) throw BadProperty("Nonfinite state");
    p.tau=tau/1000.;
    solver_=p;y_=y;c_ref_=cr;c_scale_=cs;h_ref_=hr;c_kind_=ck;h_kind_=hk;
    delta_=delta;sic_scale_=ss;sic_th_=st;
    allow_extrapolation_=extrapolate;
  }
private:
  friend class RecordablesMap<AstrocytePWA>;
  friend class UniversalDataLogger<AstrocytePWA>;
  pwla::Stepper solver_{{},{0,0,0,0,0,0,0,0}};
  V4 y_{0,0,.45,1};
  double c_ref_=.05,c_scale_=1.,h_ref_=.8;
  long c_kind_=0,h_kind_=0; // C: identity/sqrt_tails; h: identity/cube/log
  double delta_=.0002,sic_scale_=1.,sic_th_=.19669;
  bool allow_extrapolation_=false;
  long input_spikes_=0;
  RingBuffer spikes_;
  UniversalDataLogger<AstrocytePWA> logger_;
  std::vector<double> sic_;
  static RecordablesMap<AstrocytePWA> recordables_;
  void check(size_t r) const {if(r) throw UnknownReceptorType(r,get_name());}
  double calcium() const {
    if (c_kind_==0) return c_ref_+c_scale_*y_[0];
    const double cmax=2./1.185,lo=.005,hi=1.3;
    auto forward=[&](double c){return 2*c_ref_/c_scale_*(std::sqrt(c/c_ref_)-1);};
    auto deriv=[&](double c){return std::sqrt(c_ref_/c)/c_scale_;};
    if(y_[0]<forward(lo)) return lo*std::exp((y_[0]-forward(lo))/(lo*deriv(lo)));
    if(y_[0]>forward(hi)) return cmax-(cmax-hi)*std::exp(-(y_[0]-forward(hi))/((cmax-hi)*deriv(hi)));
    const double b=1+y_[0]*c_scale_/(2*c_ref_);
    return c_ref_*b*b;
  }
  double gate() const {
    if(h_kind_==0) return h_ref_+h_ref_*(1-h_ref_)*y_[1];
    if(h_kind_==1) return std::cbrt(std::pow(h_ref_,3)*(1+3*(1-h_ref_)*y_[1]));
    return h_ref_*std::exp((1-h_ref_)*y_[1]);
  }
  double ip3() const {return y_[2];}
  double sic_value() const {
    const double a=(calcium()-sic_th_)*1000.;
    return a>1?std::log(a)*sic_scale_:0.;
  }
  void init_buffers_() override {
    spikes_.clear();logger_.reset();
    sic_.assign(kernel().connection_manager.get_min_delay(),0.);
    solver_.flows=solver_.crossings=solver_.input_crossings=input_spikes_=0;
  }
  void pre_run_hook() override {
    logger_.init();solver_.dt=Time::get_resolution().get_ms()/1000.;solver_.prepare();
  }
  void update(Time const& origin,long from,long to) override {
    for(long lag=from;lag<to;++lag) {
      solver_.step(y_);
      double input=spikes_.get_value(lag);
      y_[2]+=delta_*input;
      if(input!=0) ++input_spikes_; // occupied delivery bins, not multiplicity
      if(!std::isfinite(y_[2]) || y_[2]<0 || (!allow_extrapolation_&&y_[2]>1.3))
        throw BadProperty("Input drove IP3 outside allowed range (enable allow_ip3_extrapolation for IP3 > 1.3)");
      sic_[lag]=sic_value();
      logger_.record_data(origin.get_steps()+lag);
    }
    SICEvent e;e.set_coeffarray(sic_);kernel().event_delivery_manager.send_secondary(*this,e);
  }
};
RecordablesMap<AstrocytePWA> AstrocytePWA::recordables_;
template<> void RecordablesMap<AstrocytePWA>::create() {
  insert_("Ca_astro",&AstrocytePWA::calcium);insert_("h_IP3R",&AstrocytePWA::gate);
  insert_("IP3",&AstrocytePWA::ip3);insert_("SIC",&AstrocytePWA::sic_value);
}
class PWAModule: public NESTExtensionInterface {
public: void initialize() override {register_node_model<AstrocytePWA>("astrocyte_lr_pwa");}
};
}
extern "C" { nest::PWAModule module; }
