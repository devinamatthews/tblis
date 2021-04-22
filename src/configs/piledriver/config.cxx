#include "config.hpp"
#include "util/cpuid.hpp"

namespace tblis
{

int piledriver_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_AMD)
    {
        if (get_verbose() >= 1) printf("tblis: piledriver: Wrong vendor.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX))
    {
        if (get_verbose() >= 1) printf("tblis: piledriver: Doesn't support AVX.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_FMA3))
    {
        if (get_verbose() >= 1) printf("tblis: piledriver: Doesn't support FMA3.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_FMA4))
    {
        if (get_verbose() >= 1) printf("tblis: piledriver: Doesn't support FMA4.\n");
        return -1;
    }

    if (family != 0x15)
    {
        if (get_verbose() >= 1) printf("tblis: piledriver: Wrong family (%xh).\n", family);
        return -1;
    }

    if ((model != 0x02 && model < 0x10) || (model > 0x1F && model < 0x30) || model > 0x3F)
    {
        if (get_verbose() >= 1) printf("tblis: piledriver: Wrong model (%xh).\n", model);
        return -1;
    }

    return 2;
}

}
