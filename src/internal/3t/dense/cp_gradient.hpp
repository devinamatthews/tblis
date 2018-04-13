#ifndef _TBLIS_INTERNAL_3T_CP_GRADIENT_HPP_
#define _TBLIS_INTERNAL_3T_CP_GRADIENT_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

enum cp_impl_t {PHAN, DIRECT, NAIVE};
extern cp_impl_t cp_impl;

template <typename T>
void cp_gradient(const communicator& comm, const config& cfg,
                 const len_vector& len_m, len_type len_n, len_type len_r,
                 const T* A,
                 const stride_vector& stride_A_m, stride_type stride_A_n,
                 const ptr_vector<const T>& U,
                 const stride_vector& stride_U_m,
                 const stride_vector& stride_U_r,
                 T* G, stride_type stride_G_n, stride_type stride_G_r);

}
}

#endif
