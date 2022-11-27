#include "config.hpp"
#include <tblis/base/env.h>
#include <tblis/internal/cpuid.hpp>

namespace tblis
{

int excavator_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_AMD)
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: excavator: Wrong vendor.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX))
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: excavator: Doesn't support AVX.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_FMA3))
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: excavator: Doesn't support FMA3.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX2))
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: excavator: Doesn't support AVX2.\n");
        return -1;
    }

    if (family != 0x15)
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: excavator: Wrong family (%xh).\n", family);
        return -1;
    }

    if (model < 0x60 || model > 0x7F)
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: excavator: Wrong model (%xh).\n", model);
        return -1;
    }

    return 4;
}

}
