#ifndef _TBLIS_IFACE_1V_SHIFT_H_
#define _TBLIS_IFACE_1V_SHIFT_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_vector_shift(const tblis_comm* comm, const tblis_config* cfg,
                        const tblis_scalar* alpha, tblis_vector* A);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void shift(T alpha, T beta, row_view<T> A)
{
    tblis_scalar alpha_s(alpha);
    tblis_vector A_s(A, beta);

    tblis_vector_shift(nullptr, nullptr, &alpha_s, &A_s);
}

template <typename T>
void shift(const communicator& comm, T alpha, T beta, row_view<T> A)
{
    tblis_scalar alpha_s(alpha);
    tblis_vector A_s(A, beta);

    tblis_vector_shift(comm, nullptr, &alpha_s, &A_s);
}

#endif

#ifdef __cplusplus
}
#endif

#endif
