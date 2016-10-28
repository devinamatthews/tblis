#ifndef _TBLIS_BLIS_H_
#define _TBLIS_BLIS_H_

#include "tblis_config.h"
#include "util/basic_types.h"

#ifdef __cplusplus
using namespace tblis;
#endif

typedef len_type dim_t;
typedef stride_type inc_t;

typedef struct {} auxinfo_t;
typedef struct {} cntx_t;

#define bli_auxinfo_next_a(x) a
#define bli_auxinfo_next_b(x) b

typedef enum
{
    BLIS_NO_CONJUGATE      = 0x0,
    BLIS_CONJUGATE         = (1<<4)
} conj_t;

#endif
