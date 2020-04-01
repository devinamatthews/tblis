#ifndef _TBLIS_INTERNAL_1T_INDEXED_DPD_SHIFT_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_DPD_SHIFT_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

void shift(type_t type, const communicator& comm, const config& cfg,
           const scalar& alpha, const scalar& beta, bool conj_A,
           const indexed_dpd_varray_view<char>& A, const dim_vector& idx_A_A);

}
}

#endif
