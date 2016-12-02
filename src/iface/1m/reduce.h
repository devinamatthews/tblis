#ifndef _TBLIS_IFACE_1M_REDUCE_H_
#define _TBLIS_IFACE_1M_REDUCE_H_

#include "../../util/basic_types.h"
#include "../../util/thread.h"

#ifdef __cplusplus

#include <utility>

namespace tblis
{

extern "C"
{

#endif

void tblis_matrix_reduce(const tblis_comm* comm, const tblis_config* cfg,
                         reduce_t op, const tblis_matrix* A,
                         tblis_scalar* result, len_type* idx);

#ifdef __cplusplus

}

template <typename T>
void reduce(reduce_t op, const_matrix_view<T> A, T& result, len_type& idx)
{
    tblis_matrix A_s(A);
    tblis_scalar result_s(result);
    tblis_matrix_reduce(nullptr, nullptr, op, &A_s, &result_s, &idx);
    result = result_s.get<T>();
}

template <typename T>
void reduce(single_t, reduce_t op, const_matrix_view<T> A,
            T& result, len_type& idx)
{
    tblis_matrix A_s(A);
    tblis_scalar result_s(result);
    tblis_matrix_reduce(tblis_single, nullptr, op, &A_s, &result_s, &idx);
    result = result_s.get<T>();
}

template <typename T>
void reduce(const communicator& comm, reduce_t op, const_matrix_view<T> A,
            T& result, len_type& idx)
{
    tblis_matrix A_s(A);
    tblis_scalar result_s(result);
    tblis_matrix_reduce(comm, nullptr, op, &A_s, &result_s, &idx);
    result = result_s.get<T>();
}

template <typename T>
std::pair<T,len_type> reduce(reduce_t op, const_matrix_view<T> A)
{
    std::pair<T,len_type> result;
    reduce(op, A, result.first, result.second);
    return result;
}

template <typename T>
std::pair<T,len_type> reduce(single_t, reduce_t op, const_matrix_view<T> A)
{
    std::pair<T,len_type> result;
    reduce(single, op, A, result.first, result.second);
    return result;
}

template <typename T>
std::pair<T,len_type> reduce(const communicator& comm, reduce_t op,
                             const_matrix_view<T> A)
{
    std::pair<T,len_type> result;
    reduce(comm, op, A, result.first, result.second);
    return result;
}

}

#endif

#endif
