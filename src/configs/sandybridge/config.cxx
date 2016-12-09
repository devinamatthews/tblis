#include "util/cpuid.hpp"
#include "config.hpp"

namespace tblis
{

int sandybridge_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_INTEL ||
        !check_features(features, FEATURE_AVX)) return -1;

    return 2;
}

TBLIS_CONFIG_INSTANTIATE(sandybridge);

}
