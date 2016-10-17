#ifndef _TBLIS_SCALV_HPP_
#define _TBLIS_SCALV_HPP_

#include "../../external/tci/src/tblis_thread.hpp"
#include "../util/marray.hpp"

namespace tblis
{

template <typename T> void tblis_scalv_ref(thread_communicator& comm, len_type n, T alpha, T* A, stride_type inc_A);

template <typename T> void tblis_scalv(len_type n, T alpha, T* A, stride_type inc_A);

template <typename T> void tblis_scalv(T alpha, row_view<T> A);

}

#endif
