#ifndef _TBLIS_1M_REDUCE_H_
#define _TBLIS_1M_REDUCE_H_

#include "util/basic_types.h"
#include "util/thread.h"

#ifdef __cplusplus

#include "util/marray.hpp"

namespace tblis
{

extern "C"
{

#endif

void tblis_matrix_reduce(len_type m, len_type n,
                         const void* A, type_t type_A,
                         stride_type rs_A, stride_type cs_A,
                         void* norm, type_t type_norm);

void tblis_matrix_reduce_single(len_type m, len_type n,
                                const void* A, type_t type_A,
                                stride_type rs_A, stride_type cs_A,
                                void* norm, type_t type_norm);

void tblis_matrix_reduce_coll(tci_comm_t* comm,
                              len_type m, len_type n,
                              const void* A, type_t type_A,
                              stride_type rs_A, stride_type cs_A,
                              void* norm, type_t type_norm);

#ifdef __cplusplus

}

template <typename T, typename U>
void reduce(const_matrix_view<T> A, U& norm)
{
    tblis_matrix_reduce(A.length(0), A.length(1),
                        A.data(), type_tag<T>::value,
                        A.stride(0), A.stride(1),
                        &norm, type_tag<U>::value);
}

template <typename T, typename U>
void reduce(single_t s, const_matrix_view<T> A, U& norm)
{
    tblis_matrix_reduce_single(A.length(0), A.length(1),
                               A.data(), type_tag<T>::value,
                               A.stride(0), A.stride(1),
                               &norm, type_tag<U>::value);
}

template <typename T, typename U>
void reduce(communicator& comm, const_matrix_view<T> A, U& norm)
{
    tblis_matrix_reduce_coll(comm, A.length(0), A.length(1),
                             A.data(), type_tag<T>::value,
                             A.stride(0), A.stride(1),
                             &norm, type_tag<U>::value);
}

template <typename T, typename U=T>
U reduce(const_matrix_view<T> A)
{
    U norm;
    reduce(A, norm);
    return norm;
}

template <typename T, typename U=T>
U reduce(single_t s, const_matrix_view<T> A)
{
    U norm;
    reduce(s, A, norm);
    return norm;
}

template <typename T, typename U=T>
U reduce(communicator& comm, const_matrix_view<T> A)
{
    U norm;
    reduce(comm, A, norm);
    return norm;
}

}

#endif

#endif
