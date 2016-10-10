#ifndef _TBLIS_PARTITION_HPP_
#define _TBLIS_PARTITION_HPP_

#include <vector>

#include "tblis_assert.hpp"
#include "tblis_basic_types.hpp"
#include "tblis_marray.hpp"

namespace tblis
{

template <typename T>
void partition(const_tensor_view<T> A,
               const_tensor_view<T>& A0, const_tensor_view<T>& A1,
               unsigned dim, len_type off)
{
    TBLIS_ASSERT(&A0 != &A1);
    TBLIS_ASSERT(dim < A.dimension());
    TBLIS_ASSERT(off >= 0);

    std::vector<len_type> len = A.lengths();
    off = std::min(off, len[dim]);

    len[dim] -= off;
    A1.reset(len, A.data()+off*A.stride(dim), A.strides());

    len[dim] = off;
    A0.reset(len, A.data(), A.strides());
}

template <typename T>
void partition(tensor_view<T> A,
               tensor_view<T>& A0, tensor_view<T>& A1,
               unsigned dim, len_type off)
{
    partition(A,
              reinterpret_cast<const_tensor_view<T>&>(A0),
              reinterpret_cast<const_tensor_view<T>&>(A1),
              dim, off);
}

template <typename T>
void unpartition(const_tensor_view<T> A0, const_tensor_view<T> A1,
                 const_tensor_view<T>& A,
                 unsigned dim)
{
    TBLIS_ASSERT(dim < A0.dimension());
    TBLIS_ASSERT(detail::are_congruent_along(A0, A1, dim));
    TBLIS_ASSERT(A0.data()+A0.length(dim)*A0.stride(dim) == A1.data());

    std::vector<len_type> len = A0.lengths();
    len[dim] += A1.length(dim);
    A.reset(len, A0.data(), A0.strides());
}

template <typename T>
void unpartition(tensor_view<T> A0, tensor_view<T> A1,
                 tensor_view<T>& A,
                 unsigned dim)
{
    unpartition(A0, A1,
                reinterpret_cast<const_tensor_view<T>&>(A),
                dim);
}

template <typename T>
void slice(const_tensor_view<T> A,
           const_tensor_view<T>& A0, const_tensor_view<T>& a1, const_tensor_view<T>& A2,
           unsigned dim, len_type off)
{
    TBLIS_ASSERT(&A0 != &a1);
    TBLIS_ASSERT(&A0 != &A2);
    TBLIS_ASSERT(dim < A.dimension());
    TBLIS_ASSERT(off >= 0 && off < A.length(dim));

    std::vector<len_type> len = A.lengths();
    std::vector<len_type> stride = A.strides();

    len[dim] -= off+1;
    A2.reset(len, A.data()+(off+1)*stride[dim], stride);

    len[dim] = off;
    A0.reset(len, A.data(), stride);

    len.erase(len.begin()+dim);
    stride.erase(stride.begin()+dim);
    a1.reset(len, A.data()+off*stride[dim], stride);
}

template <typename T>
void slice(tensor_view<T> A,
           tensor_view<T>& A0, tensor_view<T>& a1, tensor_view<T>& A2,
           unsigned dim, len_type off)
{
    slice(A,
          reinterpret_cast<const_tensor_view<T>&>(A0),
          reinterpret_cast<const_tensor_view<T>&>(a1),
          reinterpret_cast<const_tensor_view<T>&>(A2),
          dim, off);
}

template <typename T>
void slice_front(const_tensor_view<T> A,
                 const_tensor_view<T>& a0, const_tensor_view<T>& A1,
                 unsigned dim)
{
    TBLIS_ASSERT(&a0 != &A1);
    TBLIS_ASSERT(dim < A.dimension());

    std::vector<len_type> len = A.lengths();
    std::vector<len_type> stride = A.strides();

    len[dim]--;
    A1.reset(len, A.data()+stride[dim], stride);

    len.erase(len.begin()+dim);
    stride.erase(stride.begin()+dim);
    a0.reset(len, A.data(), stride);
}

template <typename T>
void slice_front(tensor_view<T> A,
                 tensor_view<T>& a0, tensor_view<T>& A1,
                 unsigned dim)
{
    slice_front(A,
                reinterpret_cast<const_tensor_view<T>&>(a0),
                reinterpret_cast<const_tensor_view<T>&>(A1),
                dim);
}

template <typename T>
void slice_back(const_tensor_view<T> A,
                const_tensor_view<T>& A0, const_tensor_view<T>& a1,
                unsigned dim)
{
    TBLIS_ASSERT(&A0 != &a1);
    TBLIS_ASSERT(dim < A.dimension());

    std::vector<len_type> len = A.lengths();
    std::vector<len_type> stride = A.strides();

    len[dim]--;
    A0.reset(len, A.data(), stride);

    len.erase(len.begin()+dim);
    stride.erase(stride.begin()+dim);
    a1.reset(len, A.data()+(A.length(dim)-1)*stride[dim], stride);
}

template <typename T>
void slice_back(tensor_view<T> A,
                tensor_view<T>& A0, tensor_view<T>& a1,
                unsigned dim)
{
    slice_back(A,
               reinterpret_cast<const_tensor_view<T>&>(A0),
               reinterpret_cast<const_tensor_view<T>&>(a1),
               dim);
}

template <typename T>
void unslice(const_tensor_view<T> A0, const_tensor_view<T> a1, const_tensor_view<T> A2,
             const_tensor_view<T>& A,
             unsigned dim)
{
    TBLIS_ASSERT(dim < A0.dimension());
    TBLIS_ASSERT(A0.dimension() == a1.dimension()+1);
    TBLIS_ASSERT(A2.dimension() == a1.dimension()+1);
    TBLIS_ASSERT(detail::are_congruent_along(A0, a1, dim));
    TBLIS_ASSERT(detail::are_congruent_along(A0, A2, dim));
    TBLIS_ASSERT(a1.data() == A0.data()+A0.length(dim)*A0.stride(dim));
    TBLIS_ASSERT(A2.data() == A0.data()+(A0.length(dim)+1)*A0.stride(dim));

    std::vector<len_type> len = A0.lengths();
    len[dim] += A2.length(dim)+1;
    A.reset(len, A0.data(), A0.strides());
}

template <typename T>
void unslice(tensor_view<T> A0, tensor_view<T> a1, tensor_view<T> A2,
             tensor_view<T>& A,
             unsigned dim)
{
    unslice(A0, a1, A2,
            reinterpret_cast<const_tensor_view<T>&>(A),
            dim);
}

template <typename T>
void unslice_front(const_tensor_view<T> a0, const_tensor_view<T> A1,
                   const_tensor_view<T>& A,
                   unsigned dim)
{
    TBLIS_ASSERT(dim < A1.dimension());
    TBLIS_ASSERT(A1.dimension() == a0.dimension()+1);
    TBLIS_ASSERT(detail::are_congruent_along(a0, A1, dim));
    TBLIS_ASSERT(A1.data() == a0.data()+A1.stride(dim));

    std::vector<len_type> len = A1.lengths();
    len[dim]++;
    A.reset(len, a0.data(), A1.strides());
}

template <typename T>
void unslice_front(tensor_view<T> a0, tensor_view<T> A1,
                   tensor_view<T>& A,
                   unsigned dim)
{
    unslice_front(a0, A1,
                  reinterpret_cast<const_tensor_view<T>&>(A),
                  dim);
}

template <typename T>
void unslice_back(const_tensor_view<T> A0, const_tensor_view<T> a1,
                  const_tensor_view<T>& A,
                  unsigned dim)
{
    TBLIS_ASSERT(dim < A0.dimension());
    TBLIS_ASSERT(A0.dimension() == a1.dimension()+1);
    TBLIS_ASSERT(detail::are_congruent_along(A0, a1, dim));
    TBLIS_ASSERT(a1.data() == A0.data()+A0.length(dim)*A0.stride(dim));

    std::vector<len_type> len = A0.lengths();
    len[dim]++;
    A.reset(len, A0.data(), A0.strides());
}

template <typename T>
void unslice_back(tensor_view<T> A0, tensor_view<T> a1,
                  tensor_view<T>& A,
                  unsigned dim)
{
    unslice_back(A0, a1,
                 reinterpret_cast<const_tensor_view<T>&>(A),
                 dim);
}

}

#endif
