#ifndef _TBLIS_MATRIX_REDUCE_HPP_
#define _TBLIS_MATRIX_REDUCE_HPP_

#include "../../external/tci/src/tblis_thread.hpp"
#include "tblis_marray.hpp"

namespace tblis
{

extern "C"
{

void tblis_matrix_reduce(len_type m, len_type n,
                         const void* A, type_t type_A,
                         stride_type rs_A, stride_type cs_A,
                         void* norm, type_t type_norm);

void tblis_matrix_reduce_single(len_type m, len_type n,
                                const void* A, type_t type_A,
                                stride_type rs_A, stride_type cs_A,
                                void* norm, type_t type_norm);

/*
void tblis_matrix_reduce_coll(len_type m, len_type n,
                              const void* A, type_t type_A,
                              stride_type rs_A, stride_type cs_A,
                              void* norm, type_t type_norm);
*/

}

template <typename T, typename U>
void matrix_reduce(const_matrix_view<T> A, U& norm)
{
    tblis_matrix_reduce(A.length(0), A.length(1),
                        A.data(), type_tag<T>::value,
                        A.stride(0), A.stride(1),
                        &norm, type_tag<U>::value);
}

template <typename T, typename U>
void matrix_reduce(single_t s, const_matrix_view<T> A, U& norm)
{
    tblis_matrix_reduce_single(A.length(0), A.length(1),
                               A.data(), type_tag<T>::value,
                               A.stride(0), A.stride(1),
                               &norm, type_tag<U>::value);
}

template <typename T, typename U=T>
U matrix_reduce(const_matrix_view<T> A)
{
    U norm;
    matrix_reduce(A, norm);
    return norm;
}

template <typename T, typename U=T>
U matrix_reduce(single_t s, const_matrix_view<T> A)
{
    U norm;
    matrix_reduce(s, A, norm);
    return norm;
}

}

#endif
