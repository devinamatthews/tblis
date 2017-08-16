#ifndef _TBLIS_INTERNAL_1T_INDEXED_DPD_SCALE_HPP_
#define _TBLIS_INTERNAL_1T_INDEXED_DPD_SCALE_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void scale(const communicator& comm, const config& cfg,
           T alpha, bool conj_A, const indexed_dpd_varray_view<T>& A,
           const dim_vector&);

}
}

#endif
