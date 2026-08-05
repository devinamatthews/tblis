#include "config.hpp"
#include "util/cpuid.hpp"

namespace tblis
{

int sandybridge_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_INTEL)
    {
        if (get_verbose() >= 1)
            printf("tblis: sandybridge: Wrong vendor.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX))
    {
        if (get_verbose() >= 1)
            printf("tblis: sandybridge: Doesn't support AVX.\n");
        return -1;
    }

    return 2;
}

} // namespace tblis
