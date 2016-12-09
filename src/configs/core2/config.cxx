#include "config.hpp"
#include "util/cpuid.hpp"

namespace tblis
{

int core2_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_INTEL ||
        !check_features(features, FEATURE_SSE3|FEATURE_SSSE3)) return -1;

    return 1;
}

TBLIS_CONFIG_INSTANTIATE(core2);

}
