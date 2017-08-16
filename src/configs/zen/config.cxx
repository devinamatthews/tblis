#include "config.hpp"

#include "util/cpuid.hpp"

namespace tblis
{

int zen_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_AMD ||
        !check_features(features, FEATURE_AVX|
                                  FEATURE_FMA3|
                                  FEATURE_AVX2)) return -1;

    if (family != 0x17) return -1;

    return 1;
}

TBLIS_CONFIG_INSTANTIATE(zen);

}
