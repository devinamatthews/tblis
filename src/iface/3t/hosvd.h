#ifndef _TBLIS_IFACE_3T_HOSVD_H_
#define _TBLIS_IFACE_3T_HOSVD_H_

#include "khatri_rao.h"

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_tensor_hosvd(tblis_tensor* A, tblis_matrix* const * U,
                        double tol, bool iterate);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <typename T, typename Container>
void hosvd(varray_view<T> A, const Container& U,
           double tol, bool iterate=false)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(T(1), U);
    tblis_tensor A_s(A);
    std::vector<tblis_matrix*> U_s(U.size());

    for (unsigned i = 0;i < U.size();i++) U_s[i] = &U_s_[i];

    tblis_tensor_hosvd(&A_s, U_s.data(), tol, iterate);
}

template <typename T, typename Container>
std::vector<matrix<T>> hosvd(varray_view<T> A,
                             double tol, bool iterate=false)
{
    std::vector<matrix<T>> U(A.dimension());
    for (unsigned i = 0;i < A.dimension();i++)
        U[i].reset({A.length(i), A.length(i)}, uninitialized, COLUMN_MAJOR);

    hosvd(A, U, tol, iterate);

    return U;
}

#endif

#ifdef __cplusplus
}
#endif

#endif
