#include "config.hpp"

#include "util/cpuid.hpp"

namespace tblis
{

int armv7a_check()
{
    int model, part, features;
    int vendor = get_cpu_type(model, part, features);

    // Arm64 has full Arm32 compatibility.
    // if (model != MODEL_ARMV7)
    // {
    //     if (get_verbose() >= 1) printf("tblis: armv7a: Not Arm32.\n");
    //     return -1;
    // }

    return 1;
}

}
