#ifndef _TBLIS_INTERNAL_1T_SHIFT_HPP_
#define _TBLIS_INTERNAL_1T_SHIFT_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void shift(const communicator& comm, const config& cfg,
           const len_vector& len_A,
           T alpha, T beta, bool conj_A, T* A, const stride_vector& stride_A);

}
}

#endif
