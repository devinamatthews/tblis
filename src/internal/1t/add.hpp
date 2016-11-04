#ifndef _TBLIS_INTERNAL_1T_ADD_HPP_
#define _TBLIS_INTERNAL_1T_ADD_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void add(const communicator& comm, const config& cfg,
         const std::vector<len_type>& len_A,
         const std::vector<len_type>& len_B,
         const std::vector<len_type>& len_AB,
         T alpha, bool conj_A, const T* A,
         const std::vector<stride_type>& stride_A,
         const std::vector<stride_type>& stride_A_AB,
         T  beta, bool conj_B,       T* B,
         const std::vector<stride_type>& stride_B,
         const std::vector<stride_type>& stride_B_AB);

}
}

#endif
