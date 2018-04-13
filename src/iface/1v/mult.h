#ifndef _TBLIS_IFACE_1V_MULT_H_
#define _TBLIS_IFACE_1V_MULT_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_vector_mult(const tblis_comm* comm, const tblis_config* cfg,
                       const tblis_vector* A, const tblis_vector* B, tblis_vector* C);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void mult(T alpha, row_view<const T> A, row_view<const T> B,
          T  beta, row_view<      T> C)
{
    tblis_vector A_s(alpha, A);
    tblis_vector B_s(B);
    tblis_vector C_s(beta, C);

    tblis_vector_mult(nullptr, nullptr, &A_s, &B_s, &C_s);
}

template <typename T>
void mult(const communicator& comm,
          T alpha, row_view<const T> A, row_view<const T> B,
          T  beta, row_view<      T> C)
{
    tblis_vector A_s(alpha, A);
    tblis_vector B_s(B);
    tblis_vector C_s(beta, C);

    tblis_vector_mult(comm, nullptr, &A_s, &B_s, &C_s);
}

#endif

#ifdef __cplusplus
}
#endif

#endif
