#include "config.hpp"
#include <tblis/base/env.h>
#include <tblis/internal/cpuid.hpp>

namespace tblis
{

int bulldozer_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_AMD)
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: bulldozer: Wrong vendor.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX))
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: bulldozer: Doesn't support AVX.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_FMA4))
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: bulldozer: Doesn't support FMA4.\n");
        return -1;
    }

    if (family != 0x15)
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: bulldozer: Wrong family (%xh).\n", family);
        return -1;
    }

    if (model != 0x00 && model != 0x01)
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: bulldozer: Wrong model (%xh).\n", model);
        return -1;
    }

    return 1;
}

}
