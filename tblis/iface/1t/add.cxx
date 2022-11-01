#include <tblis/internal/indexed_dpd.hpp>

namespace tblis
{

TBLIS_EXPORT
void tblis_tensor_add(const tblis_comm* comm,
                      const tblis_config* cfg,
                      const tblis_tensor* A,
                      const label_type* idx_A_,
                            tblis_tensor* B,
                      const label_type* idx_B_)
{
    TBLIS_ASSERT(A->type == B->type);

    len_vector len_A(A->len, A->len+A->ndim);
    stride_vector stride_A(A->stride, A->stride+A->ndim);
    label_vector idx_A(idx_A_, idx_A_+A->ndim);
    internal::canonicalize(len_A, stride_A, idx_A);

    len_vector len_B(B->len, B->len+B->ndim);
    stride_vector stride_B(B->stride, B->stride+B->ndim);
    label_vector idx_B(idx_B_, idx_B_+B->ndim);
    internal::canonicalize(len_B, stride_B, idx_B);

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

    internal::fold(len_AB, stride_B_AB, stride_A_AB, idx_AB);
    internal::fold(len_A_only, stride_A_only, idx_A_only);
    internal::fold(len_B_only, stride_B_only, idx_B_only);

    parallelize_if(
    [&](const communicator& comm)
    {
        if (A->scalar.is_zero())
        {
            len_B_only.insert(len_B_only.end(), len_AB.begin(), len_AB.end());
            stride_B_only.insert(stride_B_only.end(), stride_B_AB.begin(), stride_B_AB.end());

            if (B->scalar.is_zero())
            {
                internal::set(A->type, comm, get_config(cfg),
                              len_B_only, B->scalar,
                              reinterpret_cast<char*>(B->data),
                              stride_B_only);
            }
            else if (!B->scalar.is_one() || (B->scalar.is_complex() && B->conj))
            {
                internal::scale(A->type, comm, get_config(cfg),
                                len_B_only, B->scalar, B->conj,
                                reinterpret_cast<char*>(B->data),
                                stride_B_only);
            }
        }
        else
        {
            internal::add(A->type, comm, get_config(cfg),
                          len_A_only, len_B_only, len_AB,
                          A->scalar, A->conj, reinterpret_cast<char*>(A->data),
                          stride_A_only, stride_A_AB,
                          B->scalar, B->conj, reinterpret_cast<char*>(B->data),
                          stride_B_only, stride_B_AB);
        }
    }, comm);

    B->scalar = 1;
    B->conj = false;
}

template <typename T>
void add(const communicator& comm,
         T alpha, dpd_marray_view<const T> A, const label_string& idx_A_,
         T  beta, dpd_marray_view<      T> B, const label_string& idx_B_)
{
    auto nirrep = A.num_irreps();
    TBLIS_ASSERT(B.num_irreps() == nirrep);

    auto ndim_A = A.dimension();
    auto ndim_B = B.dimension();

    label_vector idx_A(idx_A_.idx, idx_A_.idx+ndim_A);
    label_vector idx_B(idx_B_.idx, idx_B_.idx+ndim_B);

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (auto i : range(1,ndim_B))
    for (auto j : range(i))
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

    for (auto i : range(idx_A_AB.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(A.length(idx_A_AB[i], irrep) ==
                     B.length(idx_B_AB[i], irrep));
    }

    if (alpha == T(0) || (idx_A_only.empty() && idx_B_only.empty() && A.irrep() != B.irrep()))
    {
        idx_B_B.insert(idx_B_B.end(), idx_B_AB.begin(), idx_B_AB.end());

        if (beta == T(0))
        {
            internal::set(type_tag<T>::value, comm, get_default_config(),
                          beta, reinterpret_cast<dpd_marray_view<char>&>(B), idx_B_B);
        }
        else
        {
            internal::scale(type_tag<T>::value, comm, get_default_config(),
                            beta, false, reinterpret_cast<dpd_marray_view<char>&>(B), idx_B_B);
        }
    }
    else
    {
        internal::add(type_tag<T>::value, comm, get_default_config(),
                      alpha, false, reinterpret_cast<dpd_marray_view<char>&>(A), idx_A_A, idx_A_AB,
                       beta, false, reinterpret_cast<dpd_marray_view<char>&>(B), idx_B_B, idx_B_AB);
    }
}

#define FOREACH_TYPE(T) \
template void add(const communicator& comm, \
                   T alpha, dpd_marray_view<const T> A, const label_string& idx_A, \
                   T  beta, dpd_marray_view<      T> B, const label_string& idx_B);
#include <tblis/internal/foreach_type.h>

template <typename T>
void add(const communicator& comm,
         T alpha, indexed_marray_view<const T> A, const label_string& idx_A_,
         T  beta, indexed_marray_view<      T> B, const label_string& idx_B_)
{
    auto ndim_A = A.dimension();
    auto ndim_B = B.dimension();

    label_vector idx_A(idx_A_.idx, idx_A_.idx+ndim_A);
    label_vector idx_B(idx_B_.idx, idx_B_.idx+ndim_B);

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (auto i : range(1,ndim_B))
    for (auto j : range(i))
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

    for (auto i : range(idx_A_AB.size()))
    {
        TBLIS_ASSERT(A.length(idx_A_AB[i]) ==
                     B.length(idx_B_AB[i]));
    }

    if (alpha == T(0))
    {
        idx_B_B.insert(idx_B_B.end(), idx_B_AB.begin(), idx_B_AB.end());

        if (beta == T(0))
        {
            internal::set(type_tag<T>::value, comm, get_default_config(),
                          beta, reinterpret_cast<indexed_marray_view<char>&>(B), idx_B_B);
        }
        else
        {
            internal::scale(type_tag<T>::value, comm, get_default_config(),
                            beta, false, reinterpret_cast<indexed_marray_view<char>&>(B), idx_B_B);
        }
    }
    else
    {
        internal::add(type_tag<T>::value, comm, get_default_config(),
                      alpha, false, reinterpret_cast<indexed_marray_view<char>&>(A), idx_A_A, idx_A_AB,
                       beta, false, reinterpret_cast<indexed_marray_view<char>&>(B), idx_B_B, idx_B_AB);
    }
}

#define FOREACH_TYPE(T) \
template void add(const communicator& comm, \
                   T alpha, indexed_marray_view<const T> A, const label_string& idx_A, \
                   T  beta, indexed_marray_view<      T> B, const label_string& idx_B);
#include <tblis/internal/foreach_type.h>

template <typename T>
void add(const communicator& comm,
         T alpha, indexed_dpd_marray_view<const T> A, const label_string& idx_A_,
         T  beta, indexed_dpd_marray_view<      T> B, const label_string& idx_B_)
{
    auto nirrep = A.num_irreps();
    TBLIS_ASSERT(B.num_irreps() == nirrep);

    auto ndim_A = A.dimension();
    auto ndim_B = B.dimension();

    label_vector idx_A(idx_A_.idx, idx_A_.idx+ndim_A);
    label_vector idx_B(idx_B_.idx, idx_B_.idx+ndim_B);

    for (auto i : range(1,ndim_A))
    for (auto j : range(i))
        TBLIS_ASSERT(idx_A[i] != idx_A[j]);

    for (auto i : range(1,ndim_B))
    for (auto j : range(i))
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

    for (auto i : range(idx_A_AB.size()))
    for (auto irrep : range(nirrep))
    {
        TBLIS_ASSERT(A.length(idx_A_AB[i], irrep) ==
                     B.length(idx_B_AB[i], irrep));
    }

    if (alpha == T(0) || (idx_A_only.empty() && idx_B_only.empty() && A.irrep() != B.irrep()))
    {
        idx_B_B.insert(idx_B_B.end(), idx_B_AB.begin(), idx_B_AB.end());

        if (beta == T(0))
        {
            internal::set(type_tag<T>::value, comm, get_default_config(),
                          beta, reinterpret_cast<indexed_dpd_marray_view<char>&>(B), idx_B_B);
        }
        else
        {
            internal::scale(type_tag<T>::value, comm, get_default_config(),
                            beta, false, reinterpret_cast<indexed_dpd_marray_view<char>&>(B), idx_B_B);
        }
    }
    else
    {
        internal::add(type_tag<T>::value, comm, get_default_config(),
                      alpha, false, reinterpret_cast<indexed_dpd_marray_view<char>&>(A), idx_A_A, idx_A_AB,
                       beta, false, reinterpret_cast<indexed_dpd_marray_view<char>&>(B), idx_B_B, idx_B_AB);
    }
}

#define FOREACH_TYPE(T) \
template void add(const communicator& comm, \
                   T alpha, indexed_dpd_marray_view<const T> A, const label_string& idx_A, \
                   T  beta, indexed_dpd_marray_view<      T> B, const label_string& idx_B);
#include <tblis/internal/foreach_type.h>

void add(const communicator& comm,
         const scalar& alpha,
         const const_tensor& A_,
         const label_string& idx_A,
         const scalar& beta,
         const tensor& B_,
         const label_string& idx_B)
{
    tensor A(A_.tensor_);
    tensor B(B_);
    A.scalar *= alpha.convert(A.type);
    B.scalar *= beta.convert(B.type);
    tblis_tensor_add(comm, nullptr, &A, idx_A.idx, &B, idx_B.idx);
}

}
