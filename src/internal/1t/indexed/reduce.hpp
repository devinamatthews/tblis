#ifndef _TBLIS_INTERNAL_1T_INDEXED_REDUCE_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_REDUCE_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

void reduce(type_t type, const communicator& comm, const config& cfg, reduce_t op,
            const indexed_varray_view<char>& A, const dim_vector&,
            char* result, len_type& idx);

}
}

#endif
