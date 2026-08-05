#ifndef _TBLIS_INTERNAL_1T_SET_HPP_
#define _TBLIS_INTERNAL_1T_SET_HPP_

#include "configs/configs.hpp"
#include "util/basic_types.h"
#include "util/thread.h"

namespace tblis
{
namespace internal
{

void set(type_t type,
         const communicator& comm,
         const config& cfg,
         const len_vector& len_A,
         const scalar& alpha,
         char* A,
         const stride_vector& stride_A);

}
} // namespace tblis

#endif
