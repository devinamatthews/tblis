#ifndef _TBLIS_ADDV_HPP_
#define _TBLIS_ADDV_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T>
void tblis_addv(bool conj_A, dim_t n,
                const T* A, inc_t inc_A,
                      T* B, inc_t inc_B);

template <typename T>
void tblis_addv(const Matrix<T>& A, Matrix<T>& B);

}
}

#endif
