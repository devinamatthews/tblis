#include "config.hpp"
#include "util/cpuid.hpp"

namespace tblis
{

int excavator_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_AMD ||
        !check_features(features, FEATURE_AVX|
                                  FEATURE_FMA3|
                                  FEATURE_AVX2)) return -1;

    if (family != 0x15 || model < 0x60 || model > 0x7F)
        return -1;

    return 4;
}

TBLIS_CONFIG_INSTANTIATE(excavator);

}
