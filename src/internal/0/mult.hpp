#ifndef _TBLIS_INTERNAL_0_MULT_HPP_
#define _TBLIS_INTERNAL_0_MULT_HPP_

#include "configs/configs.hpp"
#include "util/basic_types.h"
#include "util/thread.h"

namespace tblis
{
namespace internal
{

void mult(type_t type,
          const scalar& alpha,
          bool conj_A,
          char* A,
          bool conj_B,
          char* B,
          const scalar& beta,
          bool conj_C,
          char* C);

}
} // namespace tblis

#endif
