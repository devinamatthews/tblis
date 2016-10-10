#ifndef _TBLIS_AXPBYV_HPP_
#define _TBLIS_AXPBYV_HPP_

#include "../../external/tci/src/tblis_thread.hpp"
#include "tblis_marray.hpp"

namespace tblis
{

template <typename T>
void tblis_axpbyv_ref(thread_communicator& comm,
                      bool conj_A, len_type n,
                      T alpha, const T* A, stride_type inc_A,
                      T  beta,       T* B, stride_type inc_B);

template <typename T>
void tblis_axpbyv(bool conj_A, len_type n,
                  T alpha, const T* A, stride_type inc_A,
                  T  beta,       T* B, stride_type inc_B);

template <typename T>
void tblis_axpbyv(T alpha, const_row_view<T> A, T beta, row_view<T> B);

}

#endif
