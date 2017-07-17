#include "add.h"

#include "util/macros.h"
#include "util/tensor.hpp"
#include "internal/1t/dense/add.hpp"
#include "internal/1t/dense/scale.hpp"
#include "internal/1t/dense/set.hpp"
#include "internal/1t/dpd/add.hpp"
#include "internal/1t/dpd/scale.hpp"
#include "internal/1t/dpd/set.hpp"
#include "internal/1t/indexed/add.hpp"
#include "internal/1t/indexed/scale.hpp"
#include "internal/1t/indexed/set.hpp"
#include "internal/1t/indexed_dpd/add.hpp"
#include "internal/1t/indexed_dpd/scale.hpp"
#include "internal/1t/indexed_dpd/set.hpp"

namespace tblis
{

extern "C"
{

void tblis_tensor_add(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_tensor* A, const label_type* idx_A_,
                            tblis_tensor* B, const label_type* idx_B_)
{
    TBLIS_ASSERT(A->type == B->type);

    unsigned ndim_A = A->ndim;
    len_vector len_A;
    stride_vector stride_A;
    label_vector idx_A;
    diagonal(ndim_A, A->len, A->stride, idx_A_, len_A, stride_A, idx_A);

    unsigned ndim_B = B->ndim;
    len_vector len_B;
    stride_vector stride_B;
    label_vector idx_B;
    diagonal(ndim_B, B->len, B->stride, idx_B_, len_B, stride_B, idx_B);

    if (idx_A.empty() || idx_B.empty())
    {
        len_A.push_back(1);
        len_B.push_back(1);
        stride_A.push_back(0);
        stride_B.push_back(0);
        label_type idx = detail::free_idx(idx_A, idx_B);
        idx_A.push_back(idx);
        idx_B.push_back(idx);
    }

    auto idx_AB = stl_ext::intersection(idx_A, idx_B);
    auto len_AB = stl_ext::select_from(len_A, idx_A, idx_AB);
    TBLIS_ASSERT(len_AB == stl_ext::select_from(len_B, idx_B, idx_AB));
    auto stride_A_AB = stl_ext::select_from(stride_A, idx_A, idx_AB);
    auto stride_B_AB = stl_ext::select_from(stride_B, idx_B, idx_AB);

    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB);
    auto len_A_only = stl_ext::select_from(len_A, idx_A, idx_A_only);
    auto stride_A_only = stl_ext::select_from(stride_A, idx_A, idx_A_only);

    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB);
    auto len_B_only = stl_ext::select_from(len_B, idx_B, idx_B_only);
    auto stride_B_only = stl_ext::select_from(stride_B, idx_B, idx_B_only);

    TBLIS_ASSERT(idx_A_only.empty() || idx_B_only.empty());

    fold(len_AB, idx_AB, stride_A_AB, stride_B_AB);
    fold(len_A_only, idx_A_only, stride_A_only);
    fold(len_B_only, idx_B_only, stride_B_only);

    TBLIS_WITH_TYPE_AS(A->type, T,
    {
        T* data_A = static_cast<T*>(A->data);
        T* data_B = static_cast<T*>(B->data);

        parallelize_if(
        [&](const communicator& comm)
        {
            if (A->alpha<T>() == T(0))
            {
                if (B->alpha<T>() == T(0))
                {
                    internal::set<T>(comm, get_config(cfg), len_B_only+len_AB,
                                     T(0), data_B, stride_B_only+stride_B_AB);
                }
                else if (B->alpha<T>() != T(1) || (is_complex<T>::value && B->conj))
                {
                    internal::scale<T>(comm, get_config(cfg),
                                       len_B_only+len_AB,
                                       B->alpha<T>(), B->conj, data_B,
                                       stride_B_only+stride_B_AB);
                }
            }
            else
            {
                internal::add<T>(comm, get_config(cfg),
                                 len_A_only, len_B_only, len_AB,
                                 A->alpha<T>(), A->conj, data_A,
                                 stride_A_only, stride_A_AB,
                                 B->alpha<T>(), B->conj, data_B,
                                 stride_B_only, stride_B_AB);
            }
        }, comm);

        B->alpha<T>() = T(1);
        B->conj = false;
    })
}

}

template <typename T>
void add(const communicator& comm,
         T alpha, dpd_varray_view<const T> A, const label_type* idx_A_,
         T  beta, dpd_varray_view<      T> B, const label_type* idx_B_)
{
    unsigned nirrep = A.num_irreps();
    TBLIS_ASSERT(B.num_irreps() == nirrep);

    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();

    std::string idx_A(idx_A_, idx_A_+ndim_A);
    std::string idx_B(idx_B_, idx_B_+ndim_B);

    for (unsigned i = 1;i < ndim_A;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (unsigned i = 1;i < ndim_B;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_B[i] != idx_B[j]);

    auto idx_AB = stl_ext::intersection(idx_A, idx_B);
    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB);

    TBLIS_ASSERT(idx_A_only.empty() || idx_B_only.empty());
    TBLIS_ASSERT(!idx_AB.empty());

    TBLIS_ASSERT(stl_ext::intersection(idx_A_only, idx_B_only).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_A_only, idx_AB).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_B_only, idx_AB).empty());

    dim_vector range_A = range(ndim_A);
    dim_vector range_B = range(ndim_B);

    auto idx_A_AB = stl_ext::select_from(range_A, idx_A, idx_AB);
    auto idx_B_AB = stl_ext::select_from(range_B, idx_B, idx_AB);
    auto idx_A_A = stl_ext::select_from(range_A, idx_A, idx_A_only);
    auto idx_B_B = stl_ext::select_from(range_B, idx_B, idx_B_only);

    for (unsigned i = 0;i < idx_AB.size();i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            TBLIS_ASSERT(A.length(idx_A_AB[i], irrep) ==
                         B.length(idx_B_AB[i], irrep));
        }
    }

    if (alpha == T(0) || (idx_A_only.empty() && idx_B_only.empty() && A.irrep() != B.irrep()))
    {
        if (beta == T(0))
        {
            internal::set<T>(comm, get_default_config(),
                             beta, B, idx_B_B+idx_B_AB);
        }
        else
        {
            internal::scale<T>(comm, get_default_config(),
                               beta, false, B, idx_B_B+idx_B_AB);
        }
    }
    else
    {
        internal::add<T>(comm, get_default_config(),
                         alpha, false, A, idx_A_A, idx_A_AB,
                          beta, false, B, idx_B_B, idx_B_AB);
    }
}

#define FOREACH_TYPE(T) \
template void add(const communicator& comm, \
                   T alpha, dpd_varray_view<const T> A, const label_type* idx_A, \
                   T  beta, dpd_varray_view<      T> B, const label_type* idx_B);
#include "configs/foreach_type.h"

template <typename T>
void add(const communicator& comm,
         T alpha, indexed_varray_view<const T> A, const label_type* idx_A_,
         T  beta, indexed_varray_view<      T> B, const label_type* idx_B_)
{
    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();

    std::string idx_A(idx_A_, idx_A_+ndim_A);
    std::string idx_B(idx_B_, idx_B_+ndim_B);

    for (unsigned i = 1;i < ndim_A;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (unsigned i = 1;i < ndim_B;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_B[i] != idx_B[j]);

    auto idx_AB = stl_ext::intersection(idx_A, idx_B);
    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB);

    TBLIS_ASSERT(idx_A_only.empty() || idx_B_only.empty());
    TBLIS_ASSERT(!idx_AB.empty());

    TBLIS_ASSERT(stl_ext::intersection(idx_A_only, idx_B_only).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_A_only, idx_AB).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_B_only, idx_AB).empty());

    dim_vector range_A = range(ndim_A);
    dim_vector range_B = range(ndim_B);

    auto idx_A_AB = stl_ext::select_from(range_A, idx_A, idx_AB);
    auto idx_B_AB = stl_ext::select_from(range_B, idx_B, idx_AB);
    auto idx_A_A = stl_ext::select_from(range_A, idx_A, idx_A_only);
    auto idx_B_B = stl_ext::select_from(range_B, idx_B, idx_B_only);

    for (unsigned i = 0;i < idx_AB.size();i++)
    {
        TBLIS_ASSERT(A.length(idx_A_AB[i]) ==
                     B.length(idx_B_AB[i]));
    }

    if (alpha == T(0))
    {
        if (beta == T(0))
        {
            internal::set<T>(comm, get_default_config(),
                             beta, B, idx_B_B+idx_B_AB);
        }
        else
        {
            internal::scale<T>(comm, get_default_config(),
                               beta, false, B, idx_B_B+idx_B_AB);
        }
    }
    else
    {
        internal::add<T>(comm, get_default_config(),
                         alpha, false, A, idx_A_A, idx_A_AB,
                          beta, false, B, idx_B_B, idx_B_AB);
    }
}

#define FOREACH_TYPE(T) \
template void add(const communicator& comm, \
                   T alpha, indexed_varray_view<const T> A, const label_type* idx_A, \
                   T  beta, indexed_varray_view<      T> B, const label_type* idx_B);
#include "configs/foreach_type.h"

template <typename T>
void add(const communicator& comm,
         T alpha, indexed_dpd_varray_view<const T> A, const label_type* idx_A_,
         T  beta, indexed_dpd_varray_view<      T> B, const label_type* idx_B_)
{
    unsigned nirrep = A.num_irreps();
    TBLIS_ASSERT(B.num_irreps() == nirrep);

    unsigned ndim_A = A.dimension();
    unsigned ndim_B = B.dimension();

    std::string idx_A(idx_A_, idx_A_+ndim_A);
    std::string idx_B(idx_B_, idx_B_+ndim_B);

    for (unsigned i = 1;i < ndim_A;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (unsigned i = 1;i < ndim_B;i++)
        for (unsigned j = 0;j < i;j++)
            TBLIS_ASSERT(idx_B[i] != idx_B[j]);

    auto idx_AB = stl_ext::intersection(idx_A, idx_B);
    auto idx_A_only = stl_ext::exclusion(idx_A, idx_AB);
    auto idx_B_only = stl_ext::exclusion(idx_B, idx_AB);

    TBLIS_ASSERT(idx_A_only.empty() || idx_B_only.empty());
    TBLIS_ASSERT(!idx_AB.empty());

    TBLIS_ASSERT(stl_ext::intersection(idx_A_only, idx_B_only).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_A_only, idx_AB).empty());
    TBLIS_ASSERT(stl_ext::intersection(idx_B_only, idx_AB).empty());

    dim_vector range_A = range(ndim_A);
    dim_vector range_B = range(ndim_B);

    auto idx_A_AB = stl_ext::select_from(range_A, idx_A, idx_AB);
    auto idx_B_AB = stl_ext::select_from(range_B, idx_B, idx_AB);
    auto idx_A_A = stl_ext::select_from(range_A, idx_A, idx_A_only);
    auto idx_B_B = stl_ext::select_from(range_B, idx_B, idx_B_only);

    for (unsigned i = 0;i < idx_AB.size();i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            TBLIS_ASSERT(A.length(idx_A_AB[i], irrep) ==
                         B.length(idx_B_AB[i], irrep));
        }
    }

    if (alpha == T(0) || (idx_A_only.empty() && idx_B_only.empty() && A.irrep() != B.irrep()))
    {
        if (beta == T(0))
        {
            internal::set<T>(comm, get_default_config(),
                             beta, B, idx_B_B+idx_B_AB);
        }
        else
        {
            internal::scale<T>(comm, get_default_config(),
                               beta, false, B, idx_B_B+idx_B_AB);
        }
    }
    else
    {
        internal::add<T>(comm, get_default_config(),
                         alpha, false, A, idx_A_A, idx_A_AB,
                          beta, false, B, idx_B_B, idx_B_AB);
    }
}

#define FOREACH_TYPE(T) \
template void add(const communicator& comm, \
                   T alpha, indexed_dpd_varray_view<const T> A, const label_type* idx_A, \
                   T  beta, indexed_dpd_varray_view<      T> B, const label_type* idx_B);
#include "configs/foreach_type.h"

}
