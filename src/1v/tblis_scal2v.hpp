#ifndef _TBLIS_SCAL2V_HPP_
#define _TBLIS_SCAL2V_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T>
void tblis_scal2v(bool conj_A, dim_t n,
                  T alpha, const T* A, inc_t inc_A,
                                 T* B, inc_t inc_B);

template <typename T>
void tblis_scal2v(T alpha, const Matrix<T>& A, Matrix<T>& B);

}
}

#endif
