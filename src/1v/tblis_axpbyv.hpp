#ifndef _TBLIS_AXPBYV_HPP_
#define _TBLIS_AXPBYV_HPP_

#include "tblis.hpp"

namespace tblis
{

template <typename T>
void tblis_axpbyv(bool conj_A, idx_type n,
                  T alpha, const T* A, idx_type inc_A,
                  T  beta,       T* B, idx_type inc_B);

template <typename T>
void tblis_axpbyv(T alpha, const_row_view<T> A, T beta, row_view<T> B);

}

#endif
