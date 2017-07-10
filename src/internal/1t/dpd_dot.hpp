#ifndef _TBLIS_INTERNAL_1T_DPD_DOT_HPP_
#define _TBLIS_INTERNAL_1T_DPD_DOT_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dpd_dot(const communicator& comm, const config& cfg,
             bool conj_A, const dpd_varray_view<const T>& A,
             const std::vector<unsigned>& idx_A_A,
             const std::vector<unsigned>& idx_A_AB,
             bool conj_B, const dpd_varray_view<const T>& B,
             const std::vector<unsigned>& idx_B_B,
             const std::vector<unsigned>& idx_B_AB,
             T& result);

}
}

#endif
