#ifndef _TBLIS_COPYV_HPP_
#define _TBLIS_COPYV_HPP_

#include "../../external/tci/src/tblis_thread.hpp"
#include "tblis_marray.hpp"

namespace tblis
{

template <typename T>
void tblis_copyv_ref(thread_communicator& comm,
                     bool conj_A, len_type n,
                     const T* A, stride_type inc_A,
                           T* B, stride_type inc_B);

template <typename T>
void tblis_copyv(bool conj_A, len_type n,
                 const T* A, stride_type inc_A,
                       T* B, stride_type inc_B);

template <typename T>
void tblis_copyv(const_row_view<T> A, row_view<T> B);

}

#endif
