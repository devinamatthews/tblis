#ifndef _TBLIS_INTERNAL_1M_DOT_HPP_
#define _TBLIS_INTERNAL_1M_DOT_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dot(const communicator& comm, const config& cfg, len_type m, len_type n,
         bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
         bool conj_B, const T* B, stride_type rs_B, stride_type cs_B, T& result);

}
}

#endif
