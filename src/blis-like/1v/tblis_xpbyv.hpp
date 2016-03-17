#ifndef _TBLIS_XPBYV_HPP_
#define _TBLIS_XPBYV_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T>
void tblis_xpbyv(bool conj_A, dim_t n,
                         const T* A, inc_t inc_A,
                 T beta,       T* B, inc_t inc_B);

template <typename T>
void tblis_xpbyv(const Matrix<T>& A, T beta, Matrix<T>& B);

}
}

#endif
