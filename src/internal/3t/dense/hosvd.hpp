#ifndef _TBLIS_INTERNAL_3T_HOSVD_HPP_
#define _TBLIS_INTERNAL_3T_HOSVD_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void hosvd(len_vector& len_A, T* A, const stride_vector& stride_A,
           const ptr_vector<T>& U, const stride_vector& ld_U,
           double tol, bool iterate);

}
}

#endif
