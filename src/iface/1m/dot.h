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
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void dot(matrix_view<const T> A, matrix_view<const T> B, T& result)
{
    tblis_matrix A_s(A);
    tblis_matrix B_s(B);
    tblis_scalar result_s(result);
    tblis_matrix_dot(nullptr, nullptr, &A_s, &B_s, &result_s);
    result = result_s.get<T>();
}

template <typename T>
void dot(const communicator& comm, matrix_view<const T> A, matrix_view<const T> B, T& result)
{
    tblis_matrix A_s(A);
    tblis_matrix B_s(B);
    tblis_scalar result_s(result);
    tblis_matrix_dot(comm, nullptr, &A_s, &B_s, &result_s);
    result = result_s.get<T>();
}

template <typename T>
T dot(matrix_view<const T> A, matrix_view<const T> B)
{
    T result;
    dot(A, B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm, matrix_view<const T> A, matrix_view<const T> B)
{
    T result;
    dot(comm, A, B, result);
    return result;
}

#endif

#ifdef __cplusplus
}
#endif

#endif
