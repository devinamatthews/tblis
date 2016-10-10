
#define INSTANTIATE_FOR_TYPE(type) \
INSTANTIATE_FOR_CONFIG(type, reference_config)
#include "tblis_instantiate_for_types.hpp"

#undef INSTANTIATE_FOR_CONFIG
