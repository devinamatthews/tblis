#ifndef _TBLIS_DOTV_HPP_
#define _TBLIS_DOTV_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T>
void tblis_dotv(bool conj_A, bool conj_B, dim_t n,
                const T* A, inc_t inc_A,
                const T* B, inc_t inc_B, T& sum);

template <typename T>
T tblis_dotv(bool conj_A, bool conj_B, dim_t n,
             const T* A, inc_t inc_A,
             const T* B, inc_t inc_B);

template <typename T>
void tblis_dotv(const Matrix<T>& A, const Matrix<T>& B, T& sum);

template <typename T>
T tblis_dotv(const Matrix<T>& A, const Matrix<T>& B);

}
}

#endif
