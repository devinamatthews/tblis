#ifndef _TBLIS_INTERNAL_3T_CP_REFORM_HPP_
#define _TBLIS_INTERNAL_3T_CP_REFORM_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void cp_reform(const communicator& comm, const config& cfg,
               const len_vector& len_m, len_type len_r,
               T alpha, const ptr_vector<const T>& U,
               const stride_vector& stride_U_m,
               const stride_vector& stride_U_r,
               T  beta, T* A,
               const stride_vector& stride_A_m);

}
}

#endif
