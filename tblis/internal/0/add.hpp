#ifndef _TBLIS_INTERNAL_0_ADD_HPP_
#define _TBLIS_INTERNAL_0_ADD_HPP_

#include "util/thread.h"
#include "util/basic_types.h"
#include "configs/configs.hpp"

namespace tblis
{
namespace internal
{

void add(type_t type, const scalar& alpha, bool conj_A, char* A,
                      const scalar&  beta, bool conj_B, char* B);

}
}

#endif
