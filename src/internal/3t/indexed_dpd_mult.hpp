#ifndef _TBLIS_INTERNAL_3T_INDEXED_DPD_MULT_HPP_
#define _TBLIS_INTERNAL_3T_INDEXED_DPD_MULT_HPP_

#include "mult.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void indexed_dpd_mult(const communicator& comm, const config& cfg,
                      T alpha, const indexed_dpd_varray_view<const T>& A,
                      const dim_vector& idx_A_only,
                      const dim_vector& idx_A_AB,
                      const dim_vector& idx_A_AC,
                      const dim_vector& idx_A_ABC,
                               const indexed_dpd_varray_view<const T>& B,
                      const dim_vector& idx_B_only,
                      const dim_vector& idx_B_AB,
                      const dim_vector& idx_B_BC,
                      const dim_vector& idx_B_ABC,
                      T  beta, const indexed_dpd_varray_view<      T>& C,
                      const dim_vector& idx_C_only,
                      const dim_vector& idx_C_AC,
                      const dim_vector& idx_C_BC,
                      const dim_vector& idx_C_ABC);

}
}

#endif
