#include "config.hpp"
#include "util/cpuid.hpp"

namespace tblis
{

int core2_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_INTEL)
    {
        if (get_verbose() >= 1) printf("tblis: core2: Wrong vendor.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_SSE3))
    {
        if (get_verbose() >= 1) printf("tblis: core2: Doesn't support SSE3.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_SSSE3))
    {
        if (get_verbose() >= 1) printf("tblis: core2: Doesn't support SSSE3.\n");
        return -1;
    }

    return 1;
}

}
