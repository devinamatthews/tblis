#ifndef _TBLIS_INTERNAL_1M_SCALE_HPP_
#define _TBLIS_INTERNAL_1M_SCALE_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void scale(const communicator& comm, const config& cfg, len_type m, len_type n,
           T alpha, bool conj_A, T* A, stride_type rs_A, stride_type cs_A);

}
}

#endif
