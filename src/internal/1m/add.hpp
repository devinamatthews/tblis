#ifndef _TBLIS_INTERNAL_1M_ADD_HPP_
#define _TBLIS_INTERNAL_1M_ADD_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void add(const communicator& comm, const config& cfg, len_type m, len_type n,
         T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
         T  beta, bool conj_B,       T* B, stride_type rs_B, stride_type cs_B);

}
}

#endif
