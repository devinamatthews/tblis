#ifndef _TBLIS_INTERNAL_1T_SCALE_HPP_
#define _TBLIS_INTERNAL_1T_SCALE_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void scale(const communicator& comm, const config& cfg,
           const std::vector<len_type>& len_A,
           T alpha, bool conj_A, T* A, const std::vector<stride_type>& stride_A);

}
}

#endif
