#include "config.hpp"

#include "util/cpuid.hpp"

namespace tblis
{

int armv8a_check()
{
    int model, part, features;
    int vendor = get_cpu_type(model, part, features);

    if (model != MODEL_ARMV8)
    {
        if (get_verbose() >= 1) printf("tblis: armv8a: Not Arm64.\n");
        return -1;
    }

    // All Cortex-A supports NEON.
    // if (!check_features(features, FEATURE_NEON))
    // {
    //     if (get_verbose() >= 1) printf("tblis: armv8a: Doesn't support NEON.\n");
    //     return -1;
    // }

    return 2;
}

}
