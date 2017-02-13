#ifndef _TBLIS_CONFIGS_REFERENCE_CONFIG_HPP_
#define _TBLIS_CONFIGS_REFERENCE_CONFIG_HPP_

#include "configs/config_builder.hpp"

namespace tblis
{

inline int reference_check() { return 0; }

TBLIS_BEGIN_CONFIG(reference)

TBLIS_CONFIG_CHECK(reference_check)

TBLIS_END_CONFIG

}

#endif
