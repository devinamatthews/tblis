#ifndef _TBLIS_ZEROV_HPP_
#define _TBLIS_ZEROV_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T> void tblis_zerov(dim_t n, T* A, inc_t inc_A);

template <typename T> void tblis_zerov(Matrix<T>& A);

}
}

#endif
