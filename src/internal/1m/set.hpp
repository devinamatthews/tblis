#ifndef _TBLIS_INTERNAL_1M_SET_HPP_
#define _TBLIS_INTERNAL_1M_SET_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void set(const communicator& comm, const config& cfg, len_type m, len_type n,
         T alpha, T* A, stride_type rs_A, stride_type cs_A);

}
}

#endif
