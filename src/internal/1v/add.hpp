#ifndef _TBLIS_INTERNAL_1V_ADD_HPP_
#define _TBLIS_INTERNAL_1V_ADD_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void add(const communicator& comm, const config& cfg, len_type n,
         T alpha, bool conj_A, const T* A, stride_type inc_A,
         T  beta, bool conj_B,       T* B, stride_type inc_B);

}
}

#endif
