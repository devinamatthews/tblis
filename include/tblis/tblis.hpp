#ifndef _TENSOR_TBLIS_HPP_
#define _TENSOR_TBLIS_HPP_

#include "blis++/matrix.hpp"
#include "blis++/memory.hpp"
#include "blis++/partition.hpp"
#include "blis++/vector.hpp"
#include "blis++/scalar.hpp"

namespace tblis
{

using namespace blis;

namespace matrix_constants
{
    enum {MAT_A, MAT_B, MAT_C};
    enum {DIM_M, DIM_N, DIM_K};
}

}

#endif
