#ifndef _TBLIS_XPBYV_HPP_
#define _TBLIS_XPBYV_HPP_

#include "../../external/tci/src/tblis_thread.hpp"
#include "tblis_marray.hpp"

namespace tblis
{

template <typename T>
void tblis_xpbyv_ref(thread_communicator& comm,
                     bool conj_A, len_type n,
                             const T* A, stride_type inc_A,
                     T beta,       T* B, stride_type inc_B);

template <typename T>
void tblis_xpbyv(bool conj_A, len_type n,
                         const T* A, stride_type inc_A,
                 T beta,       T* B, stride_type inc_B);

template <typename T>
void tblis_xpbyv(const_row_view<T> A, T beta, row_view<T> B);

}

#endif
