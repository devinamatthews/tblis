#include "config.hpp"
#include <tblis/base/env.h>
#include <tblis/internal/cpuid.hpp>

namespace tblis
{

int firestorm_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_ARM)
    {
        if (tblis_get_verbosity() >= 1) printf("tblis: firestorm: Wrong vendor.\n");
        return -1;
    }

    return 1;
}

}
