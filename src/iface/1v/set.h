#ifndef _TBLIS_IFACE_1V_SET_H_
#define _TBLIS_IFACE_1V_SET_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_vector_set(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_scalar* alpha, tblis_vector* A);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void set(T alpha, row_view<T> A)
{
    tblis_scalar alpha_s(alpha);
    tblis_vector A_s(A);

    tblis_vector_set(nullptr, nullptr, &alpha_s, &A_s);
}

template <typename T>
void set(const communicator& comm, T alpha, row_view<T> A)
{
    tblis_scalar alpha_s(alpha);
    tblis_vector A_s(A);

    tblis_vector_set(comm, nullptr, &alpha_s, &A_s);
}

#endif

#ifdef __cplusplus
}
#endif

#endif
