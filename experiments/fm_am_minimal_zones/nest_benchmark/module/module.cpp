#include "nest_extension_interface.h"
#include <string>
namespace nest {
void register_astrocyte_am_pwa(const std::string&);
void register_astrocyte_fm_pwa(const std::string&);
}
class PwaModule : public nest::NESTExtensionInterface {
public:
 void initialize() override {
  nest::register_astrocyte_am_pwa("astrocyte_am_pwa");
  nest::register_astrocyte_fm_pwa("astrocyte_fm_pwa");
 }
};
PwaModule pwa_astrocyte_module_LTX_module;
