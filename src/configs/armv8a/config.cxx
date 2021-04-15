#include "config.hpp"

#include "util/cpuid.hpp"

namespace tblis
{

int armv8a_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_ARM)
    {
        if (get_verbose() >= 1) printf("tblis: armv8a: Wrong vendor.\n");
        return -1;
    }

    // All Cortex-A supports NEON.
    // if (!check_features(features, FEATURE_NEON))
    // {
    //     if (get_verbose() >= 1) printf("tblis: armv8a: Doesn't support NEON.\n");
    //     return -1;
    // }

    return 1;
}

}
