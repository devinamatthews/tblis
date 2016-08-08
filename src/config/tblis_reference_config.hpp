#ifndef _TBLIS_REFERENCE_CONFIG_HPP_
#define _TBLIS_REFERENCE_CONFIG_HPP_

#include "tblis_configs.hpp"

namespace tblis
{

TBLIS_CONFIG(reference_config);

TBLIS_CONFIG_MC(reference_config,    float, 512);
TBLIS_CONFIG_MC(reference_config,   double, 256);
TBLIS_CONFIG_MC(reference_config, scomplex, 256);
TBLIS_CONFIG_MC(reference_config, dcomplex, 128);

TBLIS_CONFIG_NC(reference_config,    float, 4096);
TBLIS_CONFIG_NC(reference_config,   double, 4096);
TBLIS_CONFIG_NC(reference_config, scomplex, 4096);
TBLIS_CONFIG_NC(reference_config, dcomplex, 4096);

TBLIS_CONFIG_KC(reference_config,    float, 256);
TBLIS_CONFIG_KC(reference_config,   double, 256);
TBLIS_CONFIG_KC(reference_config, scomplex, 256);
TBLIS_CONFIG_KC(reference_config, dcomplex, 256);

TBLIS_CONFIG_MR(reference_config,    float, 8);
TBLIS_CONFIG_MR(reference_config,   double, 4);
TBLIS_CONFIG_MR(reference_config, scomplex, 4);
TBLIS_CONFIG_MR(reference_config, dcomplex, 2);

TBLIS_CONFIG_NR(reference_config,    float, 4);
TBLIS_CONFIG_NR(reference_config,   double, 4);
TBLIS_CONFIG_NR(reference_config, scomplex, 2);
TBLIS_CONFIG_NR(reference_config, dcomplex, 2);

TBLIS_CONFIG_KR(reference_config,    float, 1);
TBLIS_CONFIG_KR(reference_config,   double, 1);
TBLIS_CONFIG_KR(reference_config, scomplex, 1);
TBLIS_CONFIG_KR(reference_config, dcomplex, 1);

}

#endif
