#include "scale.h"

#include "util/macros.h"
#include "util/tensor.hpp"
#include "internal/1t/dense/scale.hpp"
#include "internal/1t/dense/set.hpp"
#include "internal/1t/dpd/scale.hpp"
#include "internal/1t/dpd/set.hpp"
#include "internal/1t/indexed/scale.hpp"
#include "internal/1t/indexed/set.hpp"
#include "internal/1t/indexed_dpd/scale.hpp"
#include "internal/1t/indexed_dpd/set.hpp"

namespace tblis
{

extern "C"
{

void tblis_tensor_scale(const tblis_comm* comm, const tblis_config* cfg,
                        tblis_tensor* A, const label_type* idx_A_)
{
    unsigned ndim_A = A->ndim;
    len_vector len_A;
    stride_vector stride_A;
    label_vector idx_A;
    diagonal(ndim_A, A->len, A->stride, idx_A_, len_A, stride_A, idx_A);

    if (idx_A.empty())
    {
        len_A.push_back(1);
        stride_A.push_back(0);
        idx_A.push_back(0);
    }

    fold(len_A, idx_A, stride_A);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        parallelize_if(
        [&](const communicator& comm)
        {
            if (A->alpha<T>() == T(0))
            {
                internal::set<T>(comm, get_config(cfg), len_A,
                                 T(0), static_cast<T*>(A->data), stride_A);
            }
            else if (A->alpha<T>() != T(1) || (is_complex<T>::value && A->conj))
            {
                internal::scale<T>(comm, get_config(cfg), len_A,
                                   A->alpha<T>(), A->conj,
                                   static_cast<T*>(A->data), stride_A);
            }
        }, comm);

        A->alpha<T>() = T(1);
        A->conj = false;
    })
}

}

template <typename T>
void scale(const communicator& comm,
           T alpha, dpd_varray_view<T> A, const label_type* idx_A)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();

    for (unsigned i = 1;i < ndim_A;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    dim_vector idx_A_A = range(ndim_A);

    if (alpha == T(0))
    {
        internal::set<T>(comm, get_default_config(), alpha, A, idx_A_A);
    }
    else
    {
        internal::scale<T>(comm, get_default_config(), alpha, false, A, idx_A_A);
    }
}

#define FOREACH_TYPE(T) \
template void scale(const communicator& comm, \
                    T alpha, dpd_varray_view<T> A, const label_type* idx_A);
#include "configs/foreach_type.h"

template <typename T>
void scale(const communicator& comm,
           T alpha, indexed_varray_view<T> A, const label_type* idx_A)
{
    unsigned ndim_A = A.dimension();

    for (unsigned i = 1;i < ndim_A;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    dim_vector idx_A_A = range(ndim_A);

    if (alpha == T(0))
    {
        internal::set<T>(comm, get_default_config(), alpha, A, idx_A_A);
    }
    else
    {
        internal::scale<T>(comm, get_default_config(), alpha, false, A, idx_A_A);
    }
}

#define FOREACH_TYPE(T) \
template void scale(const communicator& comm, \
                    T alpha, indexed_varray_view<T> A, const label_type* idx_A);
#include "configs/foreach_type.h"

template <typename T>
void scale(const communicator& comm,
           T alpha, indexed_dpd_varray_view<T> A, const label_type* idx_A)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();

    for (unsigned i = 1;i < ndim_A;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    dim_vector idx_A_A = range(ndim_A);

    if (alpha == T(0))
    {
        internal::set<T>(comm, get_default_config(), alpha, A, idx_A_A);
    }
    else
    {
        internal::scale<T>(comm, get_default_config(), alpha, false, A, idx_A_A);
    }
}

#define FOREACH_TYPE(T) \
template void scale(const communicator& comm, \
                    T alpha, indexed_dpd_varray_view<T> A, const label_type* idx_A);
#include "configs/foreach_type.h"

}
