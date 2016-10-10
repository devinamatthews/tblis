#ifndef _TBLIS_SUMV_HPP_
#define _TBLIS_SUMV_HPP_

#include "../../external/tci/src/tblis_thread.hpp"
#include "tblis_marray.hpp"

namespace tblis
{

template <typename T> void tblis_sumv_ref(thread_communicator& comm, len_type n, const T* A, stride_type inc_A, T& sum);

template <typename T> void tblis_sumv(len_type n, const T* A, stride_type inc_A, T& sum);

template <typename T>    T tblis_sumv(len_type n, const T* A, stride_type inc_A);

template <typename T> void tblis_sumv(const_row_view<T> A, T& sum);

template <typename T>    T tblis_sumv(const_row_view<T> A);

}

#endif
