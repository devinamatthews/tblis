#ifndef _TBLIS_NORMFV_HPP_
#define _TBLIS_NORMFV_HPP_

#include "tblis.hpp"

namespace tblis
{

template <typename T> void tblis_normfv_ref(thread_communicator& comm, idx_type n, const T* A, stride_type inc_A, T& norm);

template <typename T> void tblis_normfv(idx_type n, const T* A, stride_type inc_A, T& norm);

template <typename T>    T tblis_normfv(idx_type n, const T* A, stride_type inc_A);

template <typename T> void tblis_normfv(const_row_view<T> A, T& norm);

template <typename T>    T tblis_normfv(const_row_view<T> A);

}

#endif
