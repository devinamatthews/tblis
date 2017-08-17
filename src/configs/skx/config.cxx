#include "config.hpp"

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

TBLIS_CONFIG_INSTANTIATE(skx_32x6_l1);
TBLIS_CONFIG_INSTANTIATE(skx_32x6_l2);
TBLIS_CONFIG_INSTANTIATE(skx_24x8_l1);
TBLIS_CONFIG_INSTANTIATE(skx_24x8_l2);
TBLIS_CONFIG_INSTANTIATE(skx_16x12_l1);
TBLIS_CONFIG_INSTANTIATE(skx_16x12_l2);
TBLIS_CONFIG_INSTANTIATE(skx_12x16_l1);
TBLIS_CONFIG_INSTANTIATE(skx_12x16_l2);
TBLIS_CONFIG_INSTANTIATE(skx_8x24_l1);
TBLIS_CONFIG_INSTANTIATE(skx_8x24_l2);
TBLIS_CONFIG_INSTANTIATE(skx_6x32_l1);
TBLIS_CONFIG_INSTANTIATE(skx_6x32_l2);
TBLIS_CONFIG_INSTANTIATE(skx_knl);

}
