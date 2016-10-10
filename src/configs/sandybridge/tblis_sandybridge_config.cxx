#include "tblis_sandybridge_config.hpp"

#include "util/tblis_cpuid.hpp"

namespace tblis
{

    static int check()
    {
        int family, model, features;
        int vendor = get_cpu_type(family, model, features);

        if (vendor != VENDOR_INTEL || !(features&FEATURE_AVX))
            return -1;

        return 2;
    }

    TBLIS_CONFIG_CHECK(sandybridge_config, check);

    #define TBLIS_CONFIG_NAME sandybridge_config
    #include "configs/tblis_instantiate_default_kernels.hpp"

}
