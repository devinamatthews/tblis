#include "config.hpp"
#include "util/cpuid.hpp"

namespace tblis
{

int piledriver_check()
{
    int family, model, features;
    int vendor = get_cpu_type(family, model, features);

    if (vendor != VENDOR_AMD ||
        !check_features(features, FEATURE_AVX|
                                  FEATURE_FMA3|
                                  FEATURE_FMA4)) return -1;

    /*
     * This checks for both Piledriver (model=0x02,0x10-0x1F) and
     * Streamroller (model=0x30-0x3F).
     */
    if (family != 0x15 || (model != 0x02 && model < 0x10) ||
        (model > 0x1F && model < 0x30) || model > 0x3F)
        return -1;

    return 2;
}

TBLIS_CONFIG_INSTANTIATE(piledriver);

}
