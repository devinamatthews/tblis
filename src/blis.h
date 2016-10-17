#ifndef _TBLIS_BLIS_H_
#define _TBLIS_BLIS_H_

#include "util/basic_types.h"

using namespace tblis;

typedef len_type dim_t;
typedef stride_type inc_t;
#define restrict TBLIS_RESTRICT

struct auxinfo_t {};
struct cntx_t {};

#define bli_auxinfo_next_a(x) a
#define bli_auxinfo_next_b(x) b

#endif
