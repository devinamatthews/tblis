#ifndef TBLIS_BASE_ENV_H
#define TBLIS_BASE_ENV_H

#include <tblis/base/macros.h>

TBLIS_EXPORT int tblis_get_verbosity();

TBLIS_EXPORT void tblis_set_verbosity(int level);

TBLIS_EXPORT int tblis_get_num_threads();

TBLIS_EXPORT void tblis_set_num_threads(int num_threads);

#endif //TBLIS_BASE_ENV_H
