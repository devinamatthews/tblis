#ifndef _TBLIS_ASUMV_HPP_
#define _TBLIS_ASUMV_HPP_

#include "../../external/tci/src/tblis_thread.hpp"
#include "tblis_marray.hpp"

namespace tblis
{

template <typename T> void tblis_asumv_ref(thread_communicator& comm, len_type n, const T* A, stride_type inc_A, T& sum);

template <typename T> void tblis_asumv(len_type n, const T* A, stride_type inc_A, T& sum);

template <typename T>    T tblis_asumv(len_type n, const T* A, stride_type inc_A);

template <typename T> void tblis_asumv(const_row_view<T> A, T& sum);

template <typename T>    T tblis_asumv(const_row_view<T> A);

}

#endif
