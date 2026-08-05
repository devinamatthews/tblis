#ifndef _TBLIS_INTERNAL_1T_INDEXED_SET_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_SET_HPP_

#include "configs/configs.hpp"
#include "util/basic_types.h"
#include "util/thread.h"

namespace tblis
{
namespace internal
{

void set(type_t type,
         const communicator& comm,
         const config& cfg,
         const scalar& alpha,
         const indexed_marray_view<char>& A,
         const dim_vector&);

}
} // namespace tblis

#endif
