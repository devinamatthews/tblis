#ifndef _TBLIS_INTERNAL_1M_REDUCE_HPP_
#define _TBLIS_INTERNAL_1M_REDUCE_HPP_

#include "util/basic_types.h"
#include "util/thread.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void reduce(const communicator& comm, const config& cfg, reduce_t op,
            len_type m, len_type n, const T* A, stride_type rs_A, stride_type cs_A,
            T& result, len_type& idx);


}
}

#endif
