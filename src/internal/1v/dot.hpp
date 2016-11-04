#ifndef _TBLIS_INTERNAL_1V_DOT_HPP_
#define _TBLIS_INTERNAL_1V_DOT_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dot(const communicator& comm, const config& cfg, len_type n,
         bool conj_A, const T* A, stride_type inc_A,
         bool conj_B, const T* B, stride_type inc_B, T& result);

}
}

#endif
