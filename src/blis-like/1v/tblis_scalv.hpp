#ifndef _TBLIS_SCALV_HPP_
#define _TBLIS_SCALV_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T> void tblis_scalv(dim_t n, T alpha, T* A, inc_t inc_A);

template <typename T> void tblis_scalv(T alpha, Matrix<T>& A);

}
}

#endif
