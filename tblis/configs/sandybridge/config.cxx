#include <tblis/internal/cpuid.hpp>
#include "config.hpp"

namespace tblis
{

int sandybridge_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_INTEL)
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: sandybridge: Wrong vendor.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX))
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: sandybridge: Doesn't support AVX.\n");
        return -1;
    }

    return 2;
}

}
