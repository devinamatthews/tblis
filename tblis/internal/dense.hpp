#ifndef TBLIS_INTERNAL_DENSE_HPP
#define TBLIS_INTERNAL_DENSE_HPP

#include <tblis/base/thread.h>
#include <tblis/base/configs.h>

#include <tblis/internal/scalar.hpp>

#include <marray/varray.hpp>

#include <stl_ext/algorithm.hpp>

namespace tblis
{

using MArray::viterator;
using MArray::varray;
using MArray::varray_view;

namespace internal
{

enum impl_t {BLIS_BASED, BLAS_BASED, REFERENCE};
extern impl_t impl;

void add(type_t type, const communicator& comm, const config& cfg,
         const len_vector& len_A,
         const len_vector& len_B,
         const len_vector& len_AB,
         const scalar& alpha, bool conj_A, char* A,
         const stride_vector& stride_A,
         const stride_vector& stride_A_AB,
         const scalar&  beta, bool conj_B, char* B,
         const stride_vector& stride_B,
         const stride_vector& stride_B_AB);

void dot(type_t type, const communicator& comm, const config& cfg,
         const len_vector& len_AB,
         bool conj_A, char* A, const stride_vector& stride_A_AB,
         bool conj_B, char* B, const stride_vector& stride_B_AB,
         char* result);

void reduce(type_t type, const communicator& comm, const config& cfg, reduce_t op,
            const len_vector& len_A,
            char* A, const stride_vector& stride_A,
            char* result, len_type& idx);

void scale(type_t type, const communicator& comm, const config& cfg,
           const len_vector& len_A, const scalar& alpha,
           bool conj_A, char* A, const stride_vector& stride_A);

void set(type_t type, const communicator& comm, const config& cfg,
         const len_vector& len_A,
         const scalar& alpha, char* A, const stride_vector& stride_A);

void shift(type_t type, const communicator& comm, const config& cfg,
           const len_vector& len_A,
           const scalar& alpha, const scalar& beta,
           bool conj_A, char* A, const stride_vector& stride_A);

void mult(type_t type, const communicator& comm, const config& cfg,
          const len_vector& len_AB,
          const len_vector& len_AC,
          const len_vector& len_BC,
          const len_vector& len_ABC,
          const scalar& alpha, bool conj_A, char* A,
          const stride_vector& stride_A_AB,
          const stride_vector& stride_A_AC,
          const stride_vector& stride_A_ABC,
                               bool conj_B, char* B,
          const stride_vector& stride_B_AB,
          const stride_vector& stride_B_BC,
          const stride_vector& stride_B_ABC,
          const scalar&  beta, bool conj_C, char* C,
          const stride_vector& stride_C_AC,
          const stride_vector& stride_C_BC,
          const stride_vector& stride_C_ABC);

template <typename T>
dim_vector relative_permutation(const T& a, const T& b)
{
    dim_vector perm; perm.reserve(a.size());

    for (auto& e : b)
    {
        for (auto i : range(a.size()))
        {
            if (a[i] == e) perm.push_back(i);
        }
    }

    return perm;
}

template <typename T>
dim_vector inverse_permutation(const T& a)
{
    dim_vector perm; perm.reserve(a.size());

    for (auto i : range(a.size()))
        perm[a[i]] = i;

    return perm;
}

void canonicalize(len_vector& len,
                  stride_vector& stride,
                  label_vector& idx);

void fold(len_vector& len,
          stride_vector& stride,
          label_vector& idx);

void fold(len_vector& len,
          stride_vector& stride1,
          stride_vector& stride2,
          label_vector& idx);

void fold(len_vector& len,
          stride_vector& stride1,
          stride_vector& stride2,
          stride_vector& stride3,
          label_vector& idx);

inline int unit_dim(const stride_vector& stride)
{
    for (auto i : range(stride.size()))
        if (stride[i] == 1)
            return i;

    return -1;
}

inline int unit_dim(const stride_vector& stride, const dim_vector& idx)
{
    for (auto i : range(idx.size()))
        if (stride[idx[i]] == 1)
            return idx[i];

    return -1;
}

#if 0

template <typename T>
matrix_view<T> matricize(const varray_view<T>& A, int split)
{
    auto ndim = A.dimension();
    TBLIS_ASSERT(split <= ndim);
    if (ndim > 0 && A.stride(0) < A.stride(ndim-1))
    {
        for (auto i : range(1,split))
            TBLIS_ASSERT(A.stride(i) == A.stride(i-1)*A.length(i-1));
        for (auto i : range(split+1,ndim))
            TBLIS_ASSERT(A.stride(i) == A.stride(i-1)*A.length(i-1));
    }
    else
    {
        for (auto i : range(1,split))
            TBLIS_ASSERT(A.stride(i-1) == A.stride(i)*A.length(i));
        for (auto i : range(split+1,ndim))
            TBLIS_ASSERT(A.stride(i-1) == A.stride(i)*A.length(i));
    }

    len_type m = 1;
    for (auto i : range(split))
    {
        m *= A.length(i);
    }

    len_type n = 1;
    for (auto i : range(split,ndim))
    {
        n *= A.length(i);
    }

    stride_type rs, cs;

    if (ndim == 0)
    {
        rs = cs = 1;
    }
    else if (m == 1)
    {
        rs = n;
        cs = 1;
    }
    else if (n == 1)
    {
        rs = 1;
        cs = m;
    }
    else if (A.stride(0) < A.stride(ndim-1))
    {
        rs = (split ==    0 ? 1 : A.stride(    0));
        cs = (split == ndim ? m : A.stride(split));
    }
    else
    {
        rs = (split ==    0 ? n : A.stride(split-1));
        cs = (split == ndim ? 1 : A.stride( ndim-1));
    }

    return matrix_view<T>{{m, n}, A.data(), {rs, cs}};
}

template <typename T>
matrix_view<T> matricize(varray<T>& A, int split)
{
    return matricize(A.view(), split);
}

template <typename T>
matrix_view<const T> matricize(const varray<T>& A, int split)
{
    return matricize(A.view(), split);
}

#endif

}
}

#endif //TBLIS_INTERNAL_DENSE_HPP
