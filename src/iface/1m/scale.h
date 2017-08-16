#ifndef _TBLIS_IFACE_1M_SCALE_H_
#define _TBLIS_IFACE_1M_SCALE_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_matrix_scale(const tblis_comm* comm, const tblis_config* cfg,
                        tblis_matrix* A);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <typename T>
void scale(T alpha, matrix_view<T> A)
{
    tblis_matrix A_s(alpha, A);

    tblis_matrix_scale(nullptr, nullptr, &A_s);
}

template <typename T>
void scale(const communicator& comm, T alpha, matrix_view<T> A)
{
    tblis_matrix A_s(alpha, A);

    tblis_matrix_scale(comm, nullptr, &A_s);
}

#endif

#ifdef __cplusplus
}
#endif

#endif
