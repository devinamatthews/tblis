#include "util/cpuid.hpp"
#include "config.hpp"

namespace tblis
{

int firestorm_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_ARM)
    {
        if (get_verbose() >= 1) printf("tblis: firestorm: Wrong vendor.\n");
        return -1;
    }

    return 1;
}

}
