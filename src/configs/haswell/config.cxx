#include "util/cpuid.hpp"
#include "config.hpp"

namespace tblis
{

int haswell_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_INTEL ||
        !check_features(features, FEATURE_AVX|
                                  FEATURE_AVX2|
                                  FEATURE_FMA3)) return -1;

    return 3;
}

TBLIS_CONFIG_INSTANTIATE(haswell_d12x4);
TBLIS_CONFIG_INSTANTIATE(haswell_d4x12);
TBLIS_CONFIG_INSTANTIATE(haswell_d8x6);
TBLIS_CONFIG_INSTANTIATE(haswell_d6x8);

}
