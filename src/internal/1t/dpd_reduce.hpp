#ifndef _TBLIS_INTERNAL_1T_DPD_REDUCE_HPP_
#define _TBLIS_INTERNAL_1T_DPD_REDUCE_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dpd_reduce(const communicator& comm, const config& cfg, reduce_t op,
                const dpd_varray_view<const T>& A, const std::vector<unsigned>&,
                T& result, len_type& idx);

}
}

#endif
