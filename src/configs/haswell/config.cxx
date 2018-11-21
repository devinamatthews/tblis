#include "util/cpuid.hpp"
#include "config.hpp"

namespace tblis
{

int haswell_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_INTEL)
    {   
        if (get_verbose() >= 1) printf("tblis: haswell: Wrong vendor.\n");
        return -1;
    }

    if (!check_features(features, FEATURE_AVX))
    {   
        if (get_verbose() >= 1) printf("tblis: haswell: Doesn't support AVX.\n");
        return -1;
    }

    //if (!check_features(features, FEATURE_FMA3))
    //{
    //    if (get_verbose() >= 1) printf("tblis: haswell: Doesn't support FMA3.\n");
    //    return -1;
    //}

    if (!check_features(features, FEATURE_AVX2))
    {
        if (get_verbose() >= 1) printf("tblis: haswell: Doesn't support AVX2.\n");
        return -1;
    }
    
    return 3;
}

//TBLIS_CONFIG_INSTANTIATE(haswell_d12x4);
//TBLIS_CONFIG_INSTANTIATE(haswell_d4x12);
//TBLIS_CONFIG_INSTANTIATE(haswell_d8x6);
TBLIS_CONFIG_INSTANTIATE(haswell_d6x8);

}
