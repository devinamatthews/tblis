#ifndef _TBLIS_INTERNAL_1T_DPD_SET_HPP_
#define _TBLIS_INTERNAL_1T_DPD_SET_HPP_

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
         const dpd_marray_view<char>& A,
         const dim_vector& idx_A);

}
} // namespace tblis

#endif
