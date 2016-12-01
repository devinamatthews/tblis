#ifndef _TBLIS_IFACE_1V_SCALE_H_
#define _TBLIS_IFACE_1V_SCALE_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_vector_scale(const tblis_comm* comm, const tblis_config* cfg,
                        tblis_vector* A);

#ifdef __cplusplus

}

template <typename T>
void scale(T alpha, row_view<T> A)
{
    tblis_vector A_s(alpha, A);

    tblis_vector_scale(nullptr, nullptr, &A_s);
}

template <typename T>
void scale(single_t, T alpha, row_view<T> A)
{
    tblis_vector A_s(alpha, A);

    tblis_vector_scale(tblis_single, nullptr, &A_s);
}

template <typename T>
void scale(const communicator& comm, T alpha, row_view<T> A)
{
    tblis_vector A_s(alpha, A);

    tblis_vector_scale(comm, nullptr, &A_s);
}

}

#endif

#endif
