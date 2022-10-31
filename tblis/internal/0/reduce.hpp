#ifndef _TBLIS_INTERNAL_0_REDUCE_HPP_
#define _TBLIS_INTERNAL_0_REDUCE_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

void reduce(type_t type, reduce_t op,
            char* A, len_type  idx_A,
            char* B, len_type& idx_B);

}
}

#endif
