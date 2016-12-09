#include "config.hpp"
#include "util/cpuid.hpp"

namespace tblis
{

int bulldozer_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_AMD ||
        !check_features(features, FEATURE_AVX|FEATURE_FMA4)) return -1;

    if (family != 0x15 || (model != 0x00 && model != 0x01))
        return -1;

    return 1;
}

TBLIS_CONFIG_INSTANTIATE(bulldozer);

}
