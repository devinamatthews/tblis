#ifndef _TBLIS_INTERNAL_3T_MULT_HPP_
#define _TBLIS_INTERNAL_3T_MULT_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

enum impl_t {BLIS_BASED, BLAS_BASED, REFERENCE};
extern impl_t impl;

template <typename T>
void mult(const communicator& comm, const config& cfg,
          const len_vector& len_AB,
          const len_vector& len_AC,
          const len_vector& len_BC,
          const len_vector& len_ABC,
          T alpha, bool conj_A, const T* A,
          const stride_vector& stride_A_AB,
          const stride_vector& stride_A_AC,
          const stride_vector& stride_A_ABC,
                   bool conj_B, const T* B,
          const stride_vector& stride_B_AB,
          const stride_vector& stride_B_BC,
          const stride_vector& stride_B_ABC,
          T  beta, bool conj_C,       T* C,
          const stride_vector& stride_C_AC,
          const stride_vector& stride_C_BC,
          const stride_vector& stride_C_ABC);

}
}

#endif
