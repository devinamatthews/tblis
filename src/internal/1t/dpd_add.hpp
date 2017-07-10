#ifndef _TBLIS_INTERNAL_1T_DPD_ADD_HPP_
#define _TBLIS_INTERNAL_1T_DPD_ADD_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dpd_add(const communicator& comm, const config& cfg,
             T alpha, bool conj_A, const dpd_varray_view<const T>& A,
             const std::vector<unsigned>& idx_A,
             const std::vector<unsigned>& idx_A_AB,
             T  beta, bool conj_B, const dpd_varray_view<      T>& B,
             const std::vector<unsigned>& idx_B,
             const std::vector<unsigned>& idx_B_AB);

}
}

#endif
