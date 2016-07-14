#ifndef _TBLIS_DOTV_HPP_
#define _TBLIS_DOTV_HPP_

#include "tblis.hpp"

namespace tblis
{

template <typename T>
void tblis_dotv_ref(ThreadCommunicator& comm,
                    bool conj_A, bool conj_B, idx_type n,
                    const T* A, stride_type inc_A,
                    const T* B, stride_type inc_B, T& dot);

template <typename T>
void tblis_dotv(bool conj_A, bool conj_B, idx_type n,
                const T* A, stride_type inc_A,
                const T* B, stride_type inc_B, T& dot);

template <typename T>
T tblis_dotv(bool conj_A, bool conj_B, idx_type n,
             const T* A, stride_type inc_A,
             const T* B, stride_type inc_B);

template <typename T>
void tblis_dotv(const_row_view<T> A, const_row_view<T> B, T& dot);

template <typename T>
T tblis_dotv(const_row_view<T> A, const_row_view<T> B);

}

#endif
