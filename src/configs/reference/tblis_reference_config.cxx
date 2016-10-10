#include "tblis_reference_config.hpp"

namespace tblis
{

static int check()
{
    return 0;
}

TBLIS_CONFIG_CHECK(reference_config, check);

#define TBLIS_CONFIG_NAME reference_config
#include "configs/tblis_instantiate_default_kernels.hpp"

}
