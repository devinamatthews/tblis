#ifndef _TBLIS_NORMFV_HPP_
#define _TBLIS_NORMFV_HPP_

#include "../../external/tci/src/tblis_thread.hpp"
#include "tblis_marray.hpp"

namespace tblis
{

template <typename T> void tblis_normfv_ref(thread_communicator& comm, len_type n, const T* A, stride_type inc_A, T& norm);

template <typename T> void tblis_normfv(len_type n, const T* A, stride_type inc_A, T& norm);

template <typename T>    T tblis_normfv(len_type n, const T* A, stride_type inc_A);

template <typename T> void tblis_normfv(const_row_view<T> A, T& norm);

template <typename T>    T tblis_normfv(const_row_view<T> A);

}

#endif
