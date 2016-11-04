#ifndef _TBLIS_INTERNAL_1V_SET_HPP_
#define _TBLIS_INTERNAL_1V_SET_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void set(const communicator& comm, const config& cfg, len_type n,
         T alpha, T* A, stride_type inc_A);

}
}

#endif
