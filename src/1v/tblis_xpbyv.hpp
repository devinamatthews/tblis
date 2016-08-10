#ifndef _TBLIS_XPBYV_HPP_
#define _TBLIS_XPBYV_HPP_

#include "tblis_marray.hpp"
#include "tblis_thread.hpp"

namespace tblis
{

template <typename T>
void tblis_xpbyv_ref(thread_communicator& comm,
                     bool conj_A, idx_type n,
                             const T* A, stride_type inc_A,
                     T beta,       T* B, stride_type inc_B);

template <typename T>
void tblis_xpbyv(bool conj_A, idx_type n,
                         const T* A, stride_type inc_A,
                 T beta,       T* B, stride_type inc_B);

template <typename T>
void tblis_xpbyv(const_row_view<T> A, T beta, row_view<T> B);

}

#endif
