#include "config.hpp"

#include "util/cpuid.hpp"
#include "bli_vl.h"

// Spetialized handling for linux on SVE detection.
#ifdef __linux__
#include <asm/hwcap.h>
#include <sys/auxv.h>
#ifndef HWCAP_SVE
#define HWCAP_SVE (1 << 22)
#endif
#endif

namespace tblis
{

int a64fx_check()
{
    int model, part, features;
    int vendor = get_cpu_type(model, part, features);
    int has_sve = check_features(features, FEATURE_SVE)
#ifdef __linux__
        || (getauxval( AT_HWCAP ) & HWCAP_SVE)
#endif
        ;

    if (model != MODEL_ARMV8)
    {
        if (get_verbose() >= 1) printf("tblis: a64fx: Not Arm64.\n");
        return -1;
    }

    // Have to return early in order not to execute unsupported instructions in bli_vl().
    // if (!has_sve)
    // {
    //     if (get_verbose() >= 1) printf("tblis: a64fx: Doesn't support SVE.\n");
    //     return -1;
    // }
    
    if (bli_vl()*8 != 512)
    {
        if (get_verbose() >= 1) printf("tblis: a64fx: Vector length not 512bit.\n");
        return -1;
    }

    return 5;
}

}
