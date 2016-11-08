#ifndef _TBLIS_INTERNAL_1T_DOT_HPP_
#define _TBLIS_INTERNAL_1T_DOT_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dot(const communicator& comm, const config& cfg,
         const std::vector<len_type>& len_A,
         const std::vector<len_type>& len_B,
         const std::vector<len_type>& len_AB,
         bool conj_A, const T* A, const std::vector<stride_type>& stride_A,
                                  const std::vector<stride_type>& stride_A_AB,
         bool conj_B, const T* B, const std::vector<stride_type>& stride_B,
                                  const std::vector<stride_type>& stride_B_AB,
         T& result);

}
}

#endif
