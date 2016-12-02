#ifndef _TBLIS_IFACE_1M_DOT_H_
#define _TBLIS_IFACE_1M_DOT_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_matrix_dot(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_matrix* A, const tblis_matrix* B,
                      tblis_scalar* result);

#ifdef __cplusplus

}

template <typename T>
void dot(const_matrix_view<T> A, const_matrix_view<T> B, T& result)
{
    tblis_matrix A_s(A);
    tblis_matrix B_s(B);
    tblis_scalar result_s(result);
    tblis_matrix_dot(nullptr, nullptr, &A_s, &B_s, &result_s);
    result = result_s.get<T>();
}

template <typename T>
void dot(single_t, const_matrix_view<T> A, const_matrix_view<T> B, T& result)
{
    tblis_matrix A_s(A);
    tblis_matrix B_s(B);
    tblis_scalar result_s(result);
    tblis_matrix_dot(tblis_single, nullptr, &A_s, &B_s, &result_s);
    result = result_s.get<T>();
}

template <typename T>
void dot(const communicator& comm, const_matrix_view<T> A, const_matrix_view<T> B, T& result)
{
    tblis_matrix A_s(A);
    tblis_matrix B_s(B);
    tblis_scalar result_s(result);
    tblis_matrix_dot(comm, nullptr, &A_s, &B_s, &result_s);
    result = result_s.get<T>();
}

template <typename T>
T dot(const_matrix_view<T> A, const_matrix_view<T> B)
{
    T result;
    dot(A, B, result);
    return result;
}

template <typename T>
T dot(single_t, const_matrix_view<T> A, const_matrix_view<T> B)
{
    T result;
    dot(single, A, B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm, const_matrix_view<T> A, const_matrix_view<T> B)
{
    T result;
    dot(comm, A, B, result);
    return result;
}

}

#endif

#endif
