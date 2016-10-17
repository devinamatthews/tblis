#ifndef _TBLIS_1V_AXPBY_H_
#define _TBLIS_1V_AXPBY_H_

#include "util/thread.h"
#include "util/basic_types.h"
#include "util/assert.h"

#ifdef __cplusplus

#include "util/marray.hpp"

namespace tblis
{

extern "C"
{

#endif

void tblis_axpbyv(bool conj_A, len_type n,
                  const void* alpha, type_t type_alpha,
                  const void*     A, type_t type_A, stride_type inc_A,
                  const void*  beta, type_t type_beta,
                        void*     B, type_t type_B, stride_type inc_B);

void tblis_axpbyv_single(bool conj_A, len_type n,
                         const void* alpha, type_t type_alpha,
                         const void*     A, type_t type_A, stride_type inc_A,
                         const void*  beta, type_t type_beta,
                               void*     B, type_t type_B, stride_type inc_B);

void tblis_axpbyv_coll(tci_comm_t* comm,
                       bool conj_A, len_type n,
                       const void* alpha, type_t type_alpha,
                       const void*     A, type_t type_A, stride_type inc_A,
                       const void*  beta, type_t type_beta,
                             void*     B, type_t type_B, stride_type inc_B);

#ifdef __cplusplus

}

template <typename T>
void axpbyv(T alpha, const_row_view<T> A, T beta, row_view<T> B)
{
    type_t type = type_tag<T>::value;
    TBLIS_ASSERT(A.length() == B.length());
    tblis_axpbyv(false, A.length(),
                 &alpha, type, A.data(), type, A.stride(),
                 &beta, type, B.data(), type, B.stride());
}

template <typename T>
void axpbyv(single_t s, T alpha, const_row_view<T> A, T beta, row_view<T> B)
{
    type_t type = type_tag<T>::value;
    TBLIS_ASSERT(A.length() == B.length());
    tblis_axpbyv_single(false, A.length(),
                        &alpha, type, A.data(), type, A.stride(),
                        &beta, type, B.data(), type, B.stride());
}

template <typename T>
void axpbyv(communicator& comm, T alpha, const_row_view<T> A, T beta, row_view<T> B)
{
    type_t type = type_tag<T>::value;
    TBLIS_ASSERT(A.length() == B.length());
    tblis_axpbyv_coll(comm, false, A.length(),
                      &alpha, type, A.data(), type, A.stride(),
                      &beta, type, B.data(), type, B.stride());
}

}

#endif

#endif
