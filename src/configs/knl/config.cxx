#include "util/cpuid.hpp"
#include "config.hpp"

namespace tblis
{

TBLIS_CONFIG_CHECK(knl_d30x8_knc, check)
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_INTEL ||
        !check_features(features, FEATURE_AVX|
                                  FEATURE_AVX2|
                                  FEATURE_FMA3|
                                  FEATURE_AVX512F|
                                  FEATURE_AVX512PF)) return -1;

    return 4;
}

TBLIS_CONFIG_CHECK(knl_d30x8, check)
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_INTEL ||
        !check_features(features, FEATURE_AVX|
                                  FEATURE_AVX2|
                                  FEATURE_FMA3|
                                  FEATURE_AVX512F|
                                  FEATURE_AVX512PF)) return -1;

    return 4;
}

TBLIS_CONFIG_CHECK(knl_d24x8, check)
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_INTEL ||
        !check_features(features, FEATURE_AVX|
                                  FEATURE_AVX2|
                                  FEATURE_FMA3|
                                  FEATURE_AVX512F|
                                  FEATURE_AVX512PF)) return -1;

    return 4;
}

TBLIS_CONFIG_INSTANTIATE(knl_d30x8_knc);
TBLIS_CONFIG_INSTANTIATE(knl_d30x8);
TBLIS_CONFIG_INSTANTIATE(knl_d24x8);

}
