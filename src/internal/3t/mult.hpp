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
          const std::vector<len_type>& len_A,
          const std::vector<len_type>& len_B,
          const std::vector<len_type>& len_C,
          const std::vector<len_type>& len_AB,
          const std::vector<len_type>& len_AC,
          const std::vector<len_type>& len_BC,
          const std::vector<len_type>& len_ABC,
          T alpha, bool conj_A, const T* A,
          const std::vector<stride_type>& stride_A_A,
          const std::vector<stride_type>& stride_A_AB,
          const std::vector<stride_type>& stride_A_AC,
          const std::vector<stride_type>& stride_A_ABC,
                   bool conj_B, const T* B,
          const std::vector<stride_type>& stride_B_B,
          const std::vector<stride_type>& stride_B_AB,
          const std::vector<stride_type>& stride_B_BC,
          const std::vector<stride_type>& stride_B_ABC,
          T  beta, bool conj_C,       T* C,
          const std::vector<stride_type>& stride_C_C,
          const std::vector<stride_type>& stride_C_AC,
          const std::vector<stride_type>& stride_C_BC,
          const std::vector<stride_type>& stride_C_ABC);

}
}

#endif
