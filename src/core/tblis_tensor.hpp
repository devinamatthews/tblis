#ifndef _TBLIS_TENSOR_CLASS_HPP_
#define _TBLIS_TENSOR_CLASS_HPP_

#include "tblis.hpp"

#include "util/util.hpp"
#include "util/tensor_check.hpp"

#define MARRAY_DEFAULT_LAYOUT COLUMN_MAJOR
#include "external/marray/include/varray.hpp"
#include "external/stl_ext/include/algorithm.hpp"
#include "external/stl_ext/include/vector.hpp"

namespace tblis
{

template <typename T>
using const_tensor_view = MArray::const_varray_view<T>;

template <typename T>
using tensor_view = MArray::varray_view<T>;

template <typename T, typename Allocator=MArray::aligned_allocator<T,MARRAY_BASE_ALIGNMENT>>
using tensor = MArray::varray<T, Allocator>;

using MArray::const_marray_view;
using MArray::marray_view;
using MArray::marray;

using MArray::const_matrix_view;
using MArray::matrix_view;
using MArray::matrix;

using MArray::const_row_view;
using MArray::row_view;
using MArray::row;

using MArray::Layout;
using MArray::COLUMN_MAJOR;
using MArray::ROW_MAJOR;
using MArray::DEFAULT;

using MArray::make_array;
using MArray::make_vector;

using MArray::range_t;
using MArray::range;

namespace detail
{
    struct sort_by_idx_helper
    {
        const std::string& idx;

        sort_by_idx_helper(const std::string& idx) : idx(idx) {}

        bool operator()(unsigned i, unsigned j)
        {
            return idx[i] < idx[j];
        }
    };

    template <typename T>
    sort_by_idx_helper sort_by_idx(const const_tensor_view<T>& tensor, const std::string& idx)
    {
        return sort_by_idx_helper(idx);
    }

    template <typename T>
    struct sort_by_stride_helper
    {
        const const_tensor_view<T>& tensor;

        sort_by_stride_helper(const const_tensor_view<T>& tensor) : tensor(tensor) {}

        bool operator()(unsigned i, unsigned j)
        {
            return tensor.stride(i) < tensor.stride(j);
        }
    };

    template <typename T>
    sort_by_stride_helper<T> sort_by_stride(const tensor_view<T>& tensor, const std::string& idx)
    {
        return sort_by_stride_helper<T>(tensor);
    }

    template <typename T>
    bool are_congruent_along(const_tensor_view<T> A, const_tensor_view<T> B, unsigned dim)
    {
        if (A.dimension() < B.dimension()) swap(A, B);

        unsigned ndim = A.dimension();
        auto sA = A.strides().begin();
        auto sB = B.strides().begin();
        auto lA = A.lengths().begin();
        auto lB = B.lengths().begin();

        if (B.dimension() == ndim)
        {
            if (!std::equal(sA, sA+ndim, sB)) return false;
            if (!std::equal(lA, lA+dim, lB)) return false;
            if (!std::equal(lA+dim+1, lA+ndim, lB+dim+1)) return false;
        }
        else if (B.dimension() == ndim-1)
        {
            if (!std::equal(sA, sA+dim, sB)) return false;
            if (!std::equal(sA+dim+1, sA+ndim, sB+dim)) return false;
            if (!std::equal(lA, lA+dim, lB)) return false;
            if (!std::equal(lA+dim+1, lA+ndim, lB+dim)) return false;
        }
        else
        {
            return false;
        }

        return true;
    }
}

template <typename T>
void normalize(const_tensor_view<T>& AN, std::string& idx_AN)
{
    assert(AN.dimension() == idx_AN.size());

    unsigned ndim = AN.dimension();
    std::vector<unsigned> inds_A = MArray::range(ndim);
    stl_ext::sort(inds_A, detail::sort_by_idx(AN, idx_AN));

    std::string idx = idx_AN;
    std::vector<dim_t> len(ndim);
    std::vector<inc_t> stride(ndim);

    for (unsigned i = 0;i < ndim;i++)
    {
        idx_AN[i] = idx[inds_A[i]];
        len[i] = AN.length(inds_A[i]);
        stride[i] = AN.stride(inds_A[i]);
    }

    AN.reset(len, AN.data(), stride);
}

template <typename T>
const_tensor_view<T> normalized(const_tensor_view<T> A, std::string& idx_AN)
{
    normalize(A, idx_AN);
    return A;
}

template <typename T>
void normalize(tensor_view<T>& AN, std::string& idx_AN)
{
    normalize(reinterpret_cast<const_tensor_view<T>&>(AN), idx_AN);
}

template <typename T>
tensor_view<T> normalized(tensor_view<T> A, std::string& idx_AN)
{
    normalize(A, idx_AN);
    return A;
}

template <typename T>
void diagonal(const_tensor_view<T>& AD, std::string& idx_AD)
{
    assert(AD.dimension() == idx_AD.size());

    unsigned ndim_A = AD.dimension();
    std::vector<unsigned> inds_A = MArray::range(ndim_A);
    stl_ext::sort(inds_A, detail::sort_by_idx(AD, idx_AD));

    std::string idx = idx_AD;
    std::vector<dim_t> len(ndim_A);
    std::vector<inc_t> stride(ndim_A);

    unsigned ndim_AD = 0;
    for (unsigned i = 0;i < ndim_A;i++)
    {
        if (i == 0 || idx[inds_A[i]] != idx[inds_A[i-1]])
        {
            idx_AD[ndim_AD] = idx[inds_A[i]];
            len[ndim_AD] = AD.length(inds_A[i]);
            stride[ndim_AD] = AD.stride(inds_A[i]);
            ndim_AD++;
        }
        else
        {
            assert(AD._len[ndim_AD] == AD.length(inds_A[i]));
            stride[ndim_AD] += AD.stride(inds_A[i]);
        }
    }

    idx_AD.resize(ndim_AD);
    len.resize(ndim_AD);
    stride.resize(ndim_AD);

    AD.reset(len, AD.data(), stride);
}

template <typename T>
const_tensor_view<T> diagonal_of(const_tensor_view<T> A, std::string& idx_AD)
{
    diagonal(A, idx_AD);
    return A;
}

template <typename T>
void diagonal(tensor_view<T>& AD, std::string& idx_AD)
{
    diagonal(reinterpret_cast<const_tensor_view<T>&>(AD), idx_AD);
}

template <typename T>
tensor_view<T> diagonal_of(tensor_view<T> A, std::string& idx_AD)
{
    diagonal(A, idx_AD);
    return A;
}

template <typename T>
void partition(const_tensor_view<T>  A,
               const_tensor_view<T>& A0, const_tensor_view<T>& A1,
               unsigned dim, dim_t off)
{
    assert(&A0 != &A1);
    assert(dim < A.dimension());
    assert(off >= 0);

    std::vector<dim_t> len = A.lengths();
    off = std::min(off, len[dim]);

    len[dim] -= off;
    A1.reset(len, A.data()+off*A.stride(dim), A.strides());

    len[dim] = off;
    A0.reset(len, A.data(), A.strides());
}

template <typename T>
void partition(tensor_view<T>  A,
               tensor_view<T>& A0, tensor_view<T>& A1,
               unsigned dim, dim_t off)
{
    partition(A,
              reinterpret_cast<const_tensor_view<T>&>(A0),
              reinterpret_cast<const_tensor_view<T>&>(A1),
              dim, off);
}

template <typename T>
void unpartition(const_tensor_view<T>  A0, const_tensor_view<T> A1,
                 const_tensor_view<T>& A,
                 unsigned dim)
{
    assert(dim < A0.dimension());
    assert(detail::are_congruent_along(A0, A1, dim));
    assert(A0.data()+A0.length(dim)*A0.stride(dim) == A1.data());

    std::vector<dim_t> len = A0.lengths();
    len[dim] += A1.length(dim);
    A.reset(len, A0.data(), A0.strides());
}

template <typename T>
void unpartition(tensor_view<T>  A0, tensor_view<T> A1,
                 tensor_view<T>& A,
                 unsigned dim)
{
    unpartition(A0, A1,
                reinterpret_cast<const_tensor_view<T>&>(A),
                dim);
}

template <typename T>
void slice(const_tensor_view<T>  A,
           const_tensor_view<T>& A0, const_tensor_view<T>& a1, const_tensor_view<T>& A2,
           unsigned dim, dim_t off)
{
    assert(&A0 != &a1);
    assert(&A0 != &A2);
    assert(dim < A.dimension());
    assert(off >= 0 && off < A.length(dim));

    std::vector<dim_t> len = A.lengths();
    std::vector<dim_t> stride = A.strides();

    len[dim] -= off+1;
    A2.reset(len, A.data()+(off+1)*stride[dim], stride);

    len[dim] = off;
    A0.reset(len, A.data(), stride);

    len.erase(len.begin()+dim);
    stride.erase(stride.begin()+dim);
    a1.reset(len, A.data()+off*stride[dim], stride);
}

template <typename T>
void slice(tensor_view<T>  A,
           tensor_view<T>& A0, tensor_view<T>& a1, tensor_view<T>& A2,
           unsigned dim, dim_t off)
{
    slice(A,
          reinterpret_cast<const_tensor_view<T>&>(A0),
          reinterpret_cast<const_tensor_view<T>&>(a1),
          reinterpret_cast<const_tensor_view<T>&>(A2),
          dim, off);
}

template <typename T>
void slice_front(const_tensor_view<T>  A,
                 const_tensor_view<T>& a0, const_tensor_view<T>& A1,
                 unsigned dim)
{
    assert(&a0 != &A1);
    assert(dim < A.dimension());

    std::vector<dim_t> len = A.lengths();
    std::vector<dim_t> stride = A.strides();

    len[dim]--;
    A1.reset(len, A.data()+stride[dim], stride);

    len.erase(len.begin()+dim);
    stride.erase(stride.begin()+dim);
    a0.reset(len, A.data(), stride);
}

template <typename T>
void slice_front(tensor_view<T>  A,
                 tensor_view<T>& a0, tensor_view<T>& A1,
                 unsigned dim)
{
    slice_front(A,
                reinterpret_cast<const_tensor_view<T>&>(a0),
                reinterpret_cast<const_tensor_view<T>&>(A1),
                dim);
}

template <typename T>
void slice_back(const_tensor_view<T>  A,
                const_tensor_view<T>& A0, const_tensor_view<T>& a1,
                unsigned dim)
{
    assert(&A0 != &a1);
    assert(dim < A.dimension());

    std::vector<dim_t> len = A.lengths();
    std::vector<dim_t> stride = A.strides();

    len[dim]--;
    A0.reset(len, A.data(), stride);

    len.erase(len.begin()+dim);
    stride.erase(stride.begin()+dim);
    a1.reset(len, A.data()+(A.length(dim)-1)*stride[dim], stride);
}

template <typename T>
void slice_back(tensor_view<T>  A,
                tensor_view<T>& A0, tensor_view<T>& a1,
                unsigned dim)
{
    slice_back(A,
               reinterpret_cast<const_tensor_view<T>&>(A0),
               reinterpret_cast<const_tensor_view<T>&>(a1),
               dim);
}

template <typename T>
void unslice(const_tensor_view<T>  A0, const_tensor_view<T> a1, const_tensor_view<T> A2,
             const_tensor_view<T>& A,
             unsigned dim)
{
    assert(dim < A0.dimension());
    assert(A0.dimension() == a1.dimension()+1);
    assert(A2.dimension() == a1.dimension()+1);
    assert(detail::are_congruent_along(A0, a1, dim));
    assert(detail::are_congruent_along(A0, A2, dim));
    assert(a1.data() == A0.data()+A0.length(dim)*A0.stride(dim));
    assert(A2.data() == A0.data()+(A0.length(dim)+1)*A0.stride(dim));

    std::vector<dim_t> len = A0.lengths();
    len[dim] += A2.length(dim)+1;
    A.reset(len, A0.data(), A0.strides());
}

template <typename T>
void unslice(tensor_view<T>  A0, tensor_view<T> a1, tensor_view<T> A2,
             tensor_view<T>& A,
             unsigned dim)
{
    unslice(A0, a1, A2,
            reinterpret_cast<const_tensor_view<T>&>(A),
            dim);
}

template <typename T>
void unslice_front(const_tensor_view<T>  a0, const_tensor_view<T> A1,
                   const_tensor_view<T>& A,
                   unsigned dim)
{
    assert(dim < A1.dimension());
    assert(A1.dimension() == a0.dimension()+1);
    assert(detail::are_congruent_along(a0, A1, dim));
    assert(A1.data() == a0.data()+A1.stride(dim));

    std::vector<dim_t> len = A1.lengths();
    len[dim]++;
    A.reset(len, a0.data(), A1.strides());
}

template <typename T>
void unslice_front(tensor_view<T>  a0, tensor_view<T> A1,
                   tensor_view<T>& A,
                   unsigned dim)
{
    unslice_front(a0, A1,
                  reinterpret_cast<const_tensor_view<T>&>(A),
                  dim);
}

template <typename T>
void unslice_back(const_tensor_view<T>  A0, const_tensor_view<T> a1,
                  const_tensor_view<T>& A,
                  unsigned dim)
{
    assert(dim < A0.dimension());
    assert(A0.dimension() == a1.dimension()+1);
    assert(detail::are_congruent_along(A0, a1, dim));
    assert(a1.data() == A0.data()+A0.length(dim)*A0.stride(dim));

    std::vector<dim_t> len = A0.lengths();
    len[dim]++;
    A.reset(len, A0.data(), A0.strides());
}

template <typename T>
void unslice_back(tensor_view<T>  A0, tensor_view<T> a1,
                  tensor_view<T>& A,
                  unsigned dim)
{
    unslice_back(A0, a1,
                 reinterpret_cast<const_tensor_view<T>&>(A),
                 dim);
}

template <typename T>
void fold(const_tensor_view<T>& AF, std::string& idx_AF)
{
    assert(AF.dimension() == idx_AF.size());

    unsigned ndim = AF.dimension();
    std::vector<unsigned> inds_A = MArray::range(ndim);
    stl_ext::sort(inds_A, detail::sort_by_stride(AF, idx_AF));

    std::string idx = idx_AF;
    std::vector<dim_t> len(ndim);
    std::vector<inc_t> stride(ndim);

    unsigned newdim = 0;
    for (unsigned i = 0;i < ndim;i++)
    {
        if (i == 0 || AF.stride(inds_A[i]) != AF.stride(inds_A[i-1])*AF.length(inds_A[i-1]))
        {
            idx_AF[newdim] = idx[inds_A[i]];
            len[newdim] = AF.length(inds_A[i]);
            stride[newdim] = AF.stride(inds_A[i]);
            newdim++;
        }
        else
        {
            len[newdim] *= AF.length(inds_A[i]);
        }
    }

    idx_AF.resize(newdim);
    len.resize(newdim);
    stride.resize(newdim);
    AF.reset(len, AF.data(), stride);
}

template <typename T>
void fold(tensor_view<T>& AF, std::string& idx_AF)
{
    fold(reinterpret_cast<const_tensor_view<T>&>(AF), idx_AF);
}

template <typename T>
void fold(const_tensor_view<T>& AF, std::string& idx_AF,
          const_tensor_view<T>& BF, std::string& idx_BF)
{
    //TODO
}

template <typename T>
void fold(tensor_view<T>& AF, std::string& idx_AF,
          tensor_view<T>& BF, std::string& idx_BF)
{
    fold(reinterpret_cast<const_tensor_view<T>&>(AF), idx_AF,
         reinterpret_cast<const_tensor_view<T>&>(BF), idx_BF);
}

template <typename T>
void fold(const_tensor_view<T>& AF, std::string& idx_AF,
          const_tensor_view<T>& BF, std::string& idx_BF,
          const_tensor_view<T>& CF, std::string& idx_CF)
{
    //TODO
}

template <typename T>
void fold(tensor_view<T>& AF, std::string& idx_AF,
          tensor_view<T>& BF, std::string& idx_BF,
          tensor_view<T>& CF, std::string& idx_CF)
{
    fold(reinterpret_cast<const_tensor_view<T>&>(AF), idx_AF,
         reinterpret_cast<const_tensor_view<T>&>(BF), idx_BF,
         reinterpret_cast<const_tensor_view<T>&>(CF), idx_CF);
}

template <typename T>
void matricize(const_tensor_view<T>  A,
               const_matrix_view<T>& AM, unsigned split)
{
    unsigned ndim = A.dimension();
    assert(split <= ndim);
    if (ndim > 0 && A.stride(0) < A.stride(ndim-1))
    {
        for (unsigned i = 1;i < split;i++)
            assert(A.stride(i) == A.stride(i-1)*A.length(i-1));
        for (unsigned i = split+1;i < ndim;i++)
            assert(A.stride(i) == A.stride(i-1)*A.length(i-1));
    }
    else
    {
        for (unsigned i = 0;i < split-1;i++)
            assert(A.stride(i) == A.stride(i+1)*A.length(i+1));
        for (unsigned i = split;i < ndim-1;i++)
            assert(A.stride(i) == A.stride(i+1)*A.length(i+1));
    }

    dim_t m = 1;
    for (unsigned i = 0;i < split;i++)
    {
        m *= A.length(i);
    }

    dim_t n = 1;
    for (unsigned i = split;i < ndim;i++)
    {
        n *= A.length(i);
    }

    inc_t rs, cs;

    if (ndim == 0)
    {
        rs = cs = 1;
    }
    else if (A.stride(0) < A.stride(ndim-1))
    {
        rs = (split ==    0 ? 1 : A.stride(    0));
        cs = (split == ndim ? 1 : A.stride(split));
    }
    else
    {
        rs = (split ==    0 ? 1 : A.stride(split-1));
        cs = (split == ndim ? 1 : A.stride( ndim-1));
    }

    AM.reset(m, n, A.data(), rs, cs);
}

/**
 * Multiply two tensors together and sum onto a third
 *
 * This form generalizes contraction and weighting with the unary operations trace, transpose, and replicate. Note that
 * the binary contraction operation is similar in form to the unary trace operation, while the binary weighting operation is similar in form to the
 * unary diagonal operation. Any combination of these operations may be performed.
 */
template <typename T>
int tensor_mult(T alpha, const_tensor_view<T> A, std::string idx_A,
                         const_tensor_view<T> B, std::string idx_B,
                T  beta,       tensor_view<T> C, std::string idx_C)
{
    check_tensor_indices(A, idx_A, B, idx_B, C, idx_C,
                         true, true, true,
                         true, true, true,
                         true);

    diagonal(A, idx_A);
    diagonal(B, idx_B);
    diagonal(C, idx_C);

    return impl::tensor_mult_impl(alpha, A, idx_A,
                                         B, idx_B,
                                   beta, C, idx_C);
}

/**
 * Contract two tensors into a third
 *
 * The general form for a contraction is ab...ef... * ef...cd... -> ab...cd... where the indices ef... will be summed over.
 * Indices may be transposed in any tensor. Any index group may be empty (in the case that ef... is empty, this reduces to an outer product).
 */
template <typename T>
int tensor_contract(T alpha, const_tensor_view<T> A, std::string idx_A,
                             const_tensor_view<T> B, std::string idx_B,
                    T  beta,       tensor_view<T> C, std::string idx_C)
{
    check_tensor_indices(A, idx_A, B, idx_B, C, idx_C,
                         false, false, false,
                         true, true, true,
                         false);

    diagonal(A, idx_A);
    diagonal(B, idx_B);
    diagonal(C, idx_C);

    return impl::tensor_contract_impl(alpha, A, idx_A,
                                             B, idx_B,
                                       beta, C, idx_C);
}

/**
 * Weight a tensor by a second and sum onto a third
 *
 * The general form for a weighting is ab...ef... * ef...cd... -> ab...cd...ef... with no indices being summed over.
 * Indices may be transposed in any tensor. Any index group may be empty
 * (in the case that ef... is empty, this reduces to an outer product).
 */
template <typename T>
int tensor_weight(T alpha, const_tensor_view<T> A, std::string idx_A,
                           const_tensor_view<T> B, std::string idx_B,
                  T  beta,       tensor_view<T> C, std::string idx_C)
{
    check_tensor_indices(A, idx_A, B, idx_B, C, idx_C,
                         false, false, false,
                         false, true, true,
                         true);

    diagonal(A, idx_A);
    diagonal(B, idx_B);
    diagonal(C, idx_C);

    return impl::tensor_weight_impl(alpha, A, idx_A,
                                           B, idx_B,
                                     beta, C, idx_C);
}

/**
 * Sum the outer product of two tensors onto a third
 *
 * The general form for an outer product is ab... * cd... -> ab...cd... with no indices being summed over.
 * Indices may be transposed in any tensor.
 */
template <typename T>
int tensor_outer_prod(T alpha, const_tensor_view<T> A, std::string idx_A,
                               const_tensor_view<T> B, std::string idx_B,
                      T  beta,       tensor_view<T> C, std::string idx_C)
{
    check_tensor_indices(A, idx_A, B, idx_B, C, idx_C,
                         false, false, false,
                         false, true, true,
                         false);

    diagonal(A, idx_A);
    diagonal(B, idx_B);
    diagonal(C, idx_C);

    return impl::tensor_outer_prod_impl(alpha, A, idx_A,
                                               B, idx_B,
                                         beta, C, idx_C);
}

/**
 * sum a tensor (presumably operated on in one or more ways) onto a second
 *
 * This form generalizes all of the unary operations trace, transpose, and replicate, which may be performed
 * in any combination.
 */
template <typename T>
int tensor_sum(T alpha, const_tensor_view<T> A, std::string idx_A,
               T  beta,       tensor_view<T> B, std::string idx_B)
{
    check_tensor_indices(A, idx_A, B, idx_B,
                         true, true, true);

    diagonal(A, idx_A);
    diagonal(B, idx_B);

    return impl::tensor_sum_impl(alpha, A, idx_A,
                                  beta, B, idx_B);
}

/**
 * Sum over (semi)diagonal elements of a tensor and sum onto a second
 *
 * The general form for a trace operation is ab...k*l*... -> ab... where k* denotes the index k appearing one or more times, etc. and where
 * the indices kl... will be summed (traced) over. Indices may be transposed, and multiple appearances
 * of the traced indices kl... need not appear together. Either set of indices may be empty, with the special case that when no indices
 * are traced over, the result is the same as transpose.
 */
template <typename T>
int tensor_trace(T alpha, const_tensor_view<T> A, std::string idx_A,
                 T  beta,       tensor_view<T> B, std::string idx_B)
{
    check_tensor_indices(A, idx_A, B, idx_B,
                         true, false, true);

    diagonal(A, idx_A);
    diagonal(B, idx_B);

    return impl::tensor_trace_impl(alpha, A, idx_A,
                                    beta, B, idx_B);
}

/**
 * Replicate a tensor and sum onto a second
 *
 * The general form for a replication operation is ab... -> ab...c*d*... where c* denotes the index c appearing one or more times.
 * Any indices may be transposed.
 */
template <typename T>
int tensor_replicate(T alpha, const_tensor_view<T> A, std::string idx_A,
                     T  beta,       tensor_view<T> B, std::string idx_B)
{
    check_tensor_indices(A, idx_A, B, idx_B,
                         false, true, true);

    diagonal(A, idx_A);
    diagonal(B, idx_B);

    return impl::tensor_replicate_impl(alpha, A, idx_A,
                                        beta, B, idx_B);
}

/**
 * Transpose a tensor and sum onto a second
 *
 * The general form for a transposition operation is ab... -> P(ab...) where P is some permutation. Transposition may change
 * the order in which the elements of the tensor are physically stored.
 */
template <typename T>
int tensor_transpose(T alpha, const_tensor_view<T> A, std::string idx_A,
                     T  beta,       tensor_view<T> B, std::string idx_B)
{
    check_tensor_indices(A, idx_A, B, idx_B,
                         false, false, true);

    diagonal(A, idx_A);
    diagonal(B, idx_B);

    return impl::tensor_transpose_impl(alpha, A, idx_A,
                                        beta, B, idx_B);
}

/**
 * Return the dot product of two tensors
 */
template <typename T>
T tensor_dot(const_tensor_view<T> A, std::string idx_A,
             const_tensor_view<T> B, std::string idx_B)
{
    T val;
    int ret = tensor_dot(A, idx_A, B, idx_B, val);
    return val;
}

template <typename T>
int tensor_dot(const_tensor_view<T> A, std::string idx_A,
               const_tensor_view<T> B, std::string idx_B, T& val)
{
    check_tensor_indices(A, idx_A, B, idx_B,
                         false, false, true);

    diagonal(A, idx_A);
    diagonal(B, idx_B);

    return impl::tensor_dot_impl(A, idx_A,
                                 B, idx_B, val);
}

/**
 * Scale a tensor by a scalar
 */
template <typename T>
int tensor_scale(T alpha, tensor_view<T> A, std::string idx_A)
{
    check_tensor_indices(A, idx_A);

    diagonal(A, idx_A);

    return impl::tensor_scale_impl(alpha, A, idx_A);
}

/**
 * Return the reduction of a tensor, along with the corresponding index (as an offset from A) for MAX, MIN, MAX_ABS, and MIN_ABS reductions
 */
template <typename T>
std::pair<T,inc_t> tensor_reduce(reduce_t op, const_tensor_view<T> A, std::string idx_A)
{
    std::pair<T,inc_t> p;
    int ret = tensor_reduce(op, A, idx_A, p.first, p.second);
    return p;
}

template <typename T>
T tensor_reduce(reduce_t op, const_tensor_view<T> A, std::string idx_A, inc_t& idx)
{
    T val;
    int ret = tensor_reduce(op, A, idx_A, val, idx);
    return val;
}

template <typename T>
int tensor_reduce(reduce_t op, const_tensor_view<T> A, std::string idx_A, T& val)
{
   inc_t idx;
   return tensor_reduce(op, A, idx_A, val, idx);
}

template <typename T>
int tensor_reduce(reduce_t op, const_tensor_view<T> A, std::string idx_A, T& val, inc_t& idx)
{
    check_tensor_indices(A, idx_A);

    diagonal(A, idx_A);

    return impl::tensor_reduce_impl(op, A, idx_A, val, idx);
}

template <typename len_type>
stl_ext::enable_if_t<std::is_integral<detail::pointer_type_t<len_type>>::value,siz_t>
tensor_size(gint_t ndim, const len_type& len)
{
    siz_t size = 1;

    for (gint_t i = 0;i < ndim;i++)
    {
        size *= len[i];
    }

    return size;
}

template <typename len_type, typename stride_type>
stl_ext::enable_if_t<std::is_integral<detail::pointer_type_t<len_type>>::value &&
                     std::is_integral<detail::pointer_type_t<stride_type>>::value,siz_t>
tensor_storage_size(gint_t ndim, const len_type& len, const stride_type& stride)
{
    if (!stride)
    {
       return tensor_size(ndim, len, stride);
    }

    siz_t size = 1;

    for (gint_t i = 0;i < ndim;i++)
    {
        size += std::abs(stride[i])*(len[i]-1);
    }

    return size;
}

}

#endif
