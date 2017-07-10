#ifndef _TBLIS_INTERNAL_1T_DPD_SET_HPP_
#define _TBLIS_INTERNAL_1T_DPD_SET_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dpd_set(const communicator& comm, const config& cfg,
             T alpha, const dpd_varray_view<T>& A, const std::vector<unsigned>&);

}
}

#endif
