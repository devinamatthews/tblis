#ifndef _TBLIS_ASUMV_HPP_
#define _TBLIS_ASUMV_HPP_

#include "tblis_marray.hpp"
#include "tblis_thread.hpp"

namespace tblis
{

template <typename T> void tblis_asumv_ref(thread_communicator& comm, idx_type n, const T* A, stride_type inc_A, T& sum);

template <typename T> void tblis_asumv(idx_type n, const T* A, stride_type inc_A, T& sum);

template <typename T>    T tblis_asumv(idx_type n, const T* A, stride_type inc_A);

template <typename T> void tblis_asumv(const_row_view<T> A, T& sum);

template <typename T>    T tblis_asumv(const_row_view<T> A);

}

#endif
