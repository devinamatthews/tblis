#include "config.hpp"

#include "util/cpuid.hpp"
#include "bli_vl.h"

namespace tblis
{

int armv8a_sve512_check()
{
    int model, part, features;
    int vendor = get_cpu_type(model, part, features);

    if (model != MODEL_ARMV8)
    {
        if (get_verbose() >= 1) printf("tblis: armv8a_sve512: Not Arm64.\n");
        return -1;
    }

    // if (!check_features(features, FEATURE_SVE))
    // {
    //     if (get_verbose() >= 1) printf("tblis: armv8a_sve512: Doesn't support SVE.\n");
    //     return -1;
    // }
    
    if (bli_vl()*8 != 512)
    {
        if (get_verbose() >= 1) printf("tblis: armv8a_sve512: Vector length not 512bit.\n");
        return -1;
    }

    return 4;
}

}
