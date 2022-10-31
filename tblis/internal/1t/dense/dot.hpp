#ifndef _TBLIS_INTERNAL_1T_DOT_HPP_
#define _TBLIS_INTERNAL_1T_DOT_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

void dot(type_t type, const communicator& comm, const config& cfg,
         const len_vector& len_AB,
         bool conj_A, char* A, const stride_vector& stride_A_AB,
         bool conj_B, char* B, const stride_vector& stride_B_AB,
         char* result);

}
}

#endif
