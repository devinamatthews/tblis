#ifndef _TBLIS_SETV_HPP_
#define _TBLIS_SETV_HPP_

#include "tblis.hpp"

namespace tblis
{

template <typename T> void tblis_setv(idx_type n, T alpha, T* A, stride_type inc_A);

template <typename T> void tblis_setv(T alpha, row_view<T> A);

}

#endif
