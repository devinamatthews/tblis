#ifndef _TBLIS_INTERNAL_1T_REDUCE_HPP_
#define _TBLIS_INTERNAL_1T_REDUCE_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void reduce(const communicator& comm, const config& cfg, reduce_t op,
            const std::vector<len_type>& len_A,
            const T* A, const std::vector<stride_type>& stride_A,
            T& result, len_type& idx);

}
}

#endif
