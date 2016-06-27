#ifndef _TBLIS_AXPYV_HPP_
#define _TBLIS_AXPYV_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T>
void tblis_axpyv(bool conj_A, dim_t n,
                 T alpha, const T* A, inc_t inc_A,
                                T* B, inc_t inc_B);

template <typename T>
void tblis_axpyv(T alpha, const Matrix<T>& A, Matrix<T>& B);

}
}

#endif
