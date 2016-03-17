#ifndef _TBLIS_SUMV_HPP_
#define _TBLIS_SUMV_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T> void tblis_sumv(dim_t n, const T* A, inc_t inc_A, T& sum);

template <typename T>    T tblis_sumv(dim_t n, const T* A, inc_t inc_A);

template <typename T> void tblis_sumv(const Matrix<T>& A, T& sum);

template <typename T>    T tblis_sumv(const Matrix<T>& A);

}
}

#endif
