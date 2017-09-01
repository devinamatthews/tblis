#include "config.hpp"

#include "util/cpuid.hpp"

extern int vpu_count();

namespace tblis
{

int skx1_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_INTEL ||
        !check_features(features, FEATURE_AVX512F|
                                  FEATURE_AVX512DQ|
                                  FEATURE_AVX512BW|
                                  FEATURE_AVX512VL)) return -1;

    if (vpu_count() != 1) return -1;

    return 3;
}

TBLIS_CONFIG_INSTANTIATE(skx1);

}
