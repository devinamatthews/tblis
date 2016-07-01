#ifndef _TBLIS_ADDV_HPP_
#define _TBLIS_ADDV_HPP_

#include "tblis.hpp"

namespace tblis
{

template <typename T>
void tblis_addv(bool conj_A, idx_type n,
                const T* A, stride_type inc_A,
                      T* B, stride_type inc_B);

template <typename T>
void tblis_addv(const_row_view<T> A, row_view<T> B);

}

#endif
