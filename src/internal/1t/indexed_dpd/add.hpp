#ifndef _TBLIS_INTERNAL_1T_INDEXED_DPD_ADD_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_DPD_ADD_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void add(const communicator& comm, const config& cfg,
         T alpha, bool conj_A, const indexed_dpd_varray_view<const T>& A,
         const dim_vector& idx_A,
         const dim_vector& idx_A_AB,
         T  beta, bool conj_B, const indexed_dpd_varray_view<      T>& B,
         const dim_vector& idx_B,
         const dim_vector& idx_B_AB);

}
}

#endif
