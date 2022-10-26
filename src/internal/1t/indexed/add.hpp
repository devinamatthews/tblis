#ifndef _TBLIS_INTERNAL_1T_INDEXED_ADD_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_ADD_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

void add(type_t type, const communicator& comm, const config& cfg,
         const scalar& alpha, bool conj_A, const indexed_marray_view<char>& A,
         const dim_vector& idx_A,
         const dim_vector& idx_A_AB,
         const scalar&  beta, bool conj_B, const indexed_marray_view<char>& B,
         const dim_vector& idx_B,
         const dim_vector& idx_B_AB);

}
}

#endif
