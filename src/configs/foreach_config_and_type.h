
#define FOREACH_TYPE(type) \
FOREACH_CONFIG_AND_TYPE(type, dunnington_config)
#include "foreach_type.h"
#define FOREACH_TYPE(type) \
FOREACH_CONFIG_AND_TYPE(type, sandybridge_config)
#include "foreach_type.h"
#define FOREACH_TYPE(type) \
FOREACH_CONFIG_AND_TYPE(type, haswell_config)
#include "foreach_type.h"
#define FOREACH_TYPE(type) \
FOREACH_CONFIG_AND_TYPE(type, knl_config)
#include "foreach_type.h"
#define FOREACH_TYPE(type) \
FOREACH_CONFIG_AND_TYPE(type, bulldozer_config)
#include "foreach_type.h"
#define FOREACH_TYPE(type) \
FOREACH_CONFIG_AND_TYPE(type, piledriver_config)
#include "foreach_type.h"
#define FOREACH_TYPE(type) \
FOREACH_CONFIG_AND_TYPE(type, carrizo_config)
#include "foreach_type.h"
#define FOREACH_TYPE(type) \
FOREACH_CONFIG_AND_TYPE(type, reference_config)
#include "foreach_type.h"

#undef FOREACH_CONFIG_AND_TYPE
