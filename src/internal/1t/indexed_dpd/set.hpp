#ifndef _TBLIS_INTERNAL_1T_INDEXED_DPD_SET_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_DPD_SET_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

void set(type_t type, const communicator& comm, const config& cfg,
         const scalar& alpha, const indexed_dpd_varray_view<char>& A, const dim_vector& idx_A);

}
}

#endif
