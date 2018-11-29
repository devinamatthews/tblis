#ifndef _TBLIS_INTERNAL_1V_MULT_HPP_
#define _TBLIS_INTERNAL_1V_MULT_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void mult(const communicator& comm, const config& cfg, len_type n,
          T alpha, bool conj_A, const T* A, stride_type inc_A,
                   bool conj_B, const T* B, stride_type inc_B,
          T  beta, bool conj_C,       T* C, stride_type inc_C);

}
}

#endif
