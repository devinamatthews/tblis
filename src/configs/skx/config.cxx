#include "../skx/config.hpp"

#include "util/cpuid.hpp"

namespace tblis
{

int skx_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_INTEL ||
        !check_features(features, FEATURE_AVX512F|
                                  FEATURE_AVX512DQ|
                                  FEATURE_AVX512BW|
                                  FEATURE_AVX512VL)) return -1;

    return 3;
}

TBLIS_CONFIG_INSTANTIATE(skx_knl);

}
