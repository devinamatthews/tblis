#ifndef _TBLIS_SETV_HPP_
#define _TBLIS_SETV_HPP_

#include "../../external/tci/src/tblis_thread.hpp"
#include "tblis_marray.hpp"

namespace tblis
{

template <typename T> void tblis_setv_ref(thread_communicator& comm, len_type n, T alpha, T* A, stride_type inc_A);

template <typename T> void tblis_setv(len_type n, T alpha, T* A, stride_type inc_A);

template <typename T> void tblis_setv(T alpha, row_view<T> A);

}

#endif
