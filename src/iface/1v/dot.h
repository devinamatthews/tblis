#ifndef _TBLIS_IFACE_1V_DOT_H_
#define _TBLIS_IFACE_1V_DOT_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_vector_dot(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_vector* A, const tblis_vector* B,
                      tblis_scalar* result);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void dot(row_view<const T> A, row_view<const T> B, T& result)
{
    tblis_vector A_s(A);
    tblis_vector B_s(B);
    tblis_scalar result_s(result);
    tblis_vector_dot(nullptr, nullptr, &A_s, &B_s, &result_s);
    result = result_s.get<T>();
}

template <typename T>
void dot(const communicator& comm, row_view<const T> A, row_view<const T> B, T& result)
{
    tblis_vector A_s(A);
    tblis_vector B_s(B);
    tblis_scalar result_s(result);
    tblis_vector_dot(comm, nullptr, &A_s, &B_s, &result_s);
    result = result_s.get<T>();
}

template <typename T>
T dot(row_view<const T> A, row_view<const T> B)
{
    T result;
    dot(A, B, result);
    return result;
}

template <typename T>
T dot(const communicator& comm, row_view<const T> A, row_view<const T> B)
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
