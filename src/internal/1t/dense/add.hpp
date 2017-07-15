#ifndef _TBLIS_INTERNAL_1T_ADD_HPP_
#define _TBLIS_INTERNAL_1T_ADD_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void add(const communicator& comm, const config& cfg,
         const len_vector& len_A,
         const len_vector& len_B,
         const len_vector& len_AB,
         T alpha, bool conj_A, const T* A,
         const stride_vector& stride_A,
         const stride_vector& stride_A_AB,
         T  beta, bool conj_B,       T* B,
         const stride_vector& stride_B,
         const stride_vector& stride_B_AB);

}
}

#endif
