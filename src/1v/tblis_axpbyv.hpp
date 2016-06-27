#ifndef _TBLIS_AXPBYV_HPP_
#define _TBLIS_AXPBYV_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T>
void tblis_axpbyv(bool conj_A, dim_t n,
                  T alpha, const T* A, dim_t inc_A,
                  T  beta,       T* B, dim_t inc_B);

template <typename T>
void tblis_axpbyv(T alpha, const Matrix<T>& A, T beta, Matrix<T>& B);

}
}

#endif
