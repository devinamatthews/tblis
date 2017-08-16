#ifndef _TBLIS_IFACE_1V_ADD_H_
#define _TBLIS_IFACE_1V_ADD_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_vector_add(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_vector* A, tblis_vector* B);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void add(T alpha, row_view<const T> A, T beta, row_view<T> B)
{
    tblis_vector A_s(alpha, A);
    tblis_vector B_s(beta, B);

    tblis_vector_add(nullptr, nullptr, &A_s, &B_s);
}

template <typename T>
void add(const communicator& comm, T alpha, row_view<const T> A,
                                   T  beta,       row_view<T> B)
{
    tblis_vector A_s(alpha, A);
    tblis_vector B_s(beta, B);

    tblis_vector_add(comm, nullptr, &A_s, &B_s);
}

#endif

#ifdef __cplusplus
}
#endif

#endif
