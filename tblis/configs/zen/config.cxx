#include "config.hpp"

#include <tblis/internal/cpuid.hpp>

namespace tblis
{

int zen_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_AMD)
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: zen: Wrong vendor.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX))
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: zen: Doesn't support AVX.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_FMA3))
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: zen: Doesn't support FMA3.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX2))
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: zen: Doesn't support AVX2.\n");
        return -1;
    }

    if (family != 0x17)
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: zen: Wrong family (%xh).\n", family);
        return -1;
    }

    return 1;
}

}
