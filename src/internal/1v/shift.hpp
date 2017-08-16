#ifndef _TBLIS_INTERNAL_1V_SHIFT_HPP_
#define _TBLIS_INTERNAL_1V_SHIFT_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void shift(const communicator& comm, const config& cfg, len_type n,
           T alpha, T beta, bool conj_A, T* A, stride_type inc_A);

}
}

#endif
