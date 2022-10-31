#ifndef _TBLIS_INTERNAL_3T_DPD_MULT_HPP_
#define _TBLIS_INTERNAL_3T_DPD_MULT_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

enum dpd_impl_t {BLIS, BLOCKED, FULL};
extern dpd_impl_t dpd_impl;

void mult(type_t type, const communicator& comm, const config& cfg,
          const scalar& alpha,
          bool conj_A, const dpd_marray_view<char>& A,
          const dim_vector& idx_A_AB,
          const dim_vector& idx_A_AC,
          const dim_vector& idx_A_ABC,
          bool conj_B, const dpd_marray_view<char>& B,
          const dim_vector& idx_B_AB,
          const dim_vector& idx_B_BC,
          const dim_vector& idx_B_ABC,
          const scalar&  beta,
          bool conj_C, const dpd_marray_view<char>& C,
          const dim_vector& idx_C_AC,
          const dim_vector& idx_C_BC,
          const dim_vector& idx_C_ABC);

}
}

#endif