#ifndef _TBLIS_UTIL_TENSOR_HPP_
#define _TBLIS_UTIL_TENSOR_HPP_

#include <initializer_list>
#include <string>

#include "util/basic_types.h"

#include "external/stl_ext/include/algorithm.hpp"
#include "external/stl_ext/include/type_traits.hpp"
#include "external/stl_ext/include/vector.hpp"

namespace MArray
{
    template <typename T, size_t N>
    short_vector<T,N> operator+(const short_vector<T,N>& lhs,
                                const short_vector<T,N>& rhs)
    {
        short_vector<T,N> res;
        res.reserve(lhs.size() + rhs.size());
        res.insert(res.end(), lhs.begin(), lhs.end());
        res.insert(res.end(), rhs.begin(), rhs.end());
        return res;
    }
}

namespace tblis
{

namespace detail
{

inline label_type free_idx(label_vector idx)
{
    if (idx.empty()) return 0;

    stl_ext::sort(idx);

    if (idx[0] > 0) return 0;

    for (auto i : range(1,idx.size()))
    {
        if (idx[i] > idx[i-1]+1) return idx[i-1]+1;
    }

    return idx.back()+1;
}

inline label_type free_idx(const label_vector& idx_A,
                           const label_vector& idx_B)
{
    return free_idx(stl_ext::union_of(idx_A, idx_B));
}

inline label_type free_idx(const label_vector& idx_A,
                           const label_vector& idx_B,
                           const label_vector& idx_C)
{
    return free_idx(stl_ext::union_of(idx_A, idx_B, idx_C));
}

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

struct sort_by_idx_helper
{
    const label_type* idx;

    sort_by_idx_helper(const label_type* idx_) : idx(idx_) {}

    bool operator()(int i, int j) const
    {
        return idx[i] < idx[j];
    }
};

inline sort_by_idx_helper sort_by_idx(const label_type* idx)
{
    return sort_by_idx_helper(idx);
}

template <int N>
struct sort_by_stride_helper
{
    std::array<const stride_vector*, N> strides;

    sort_by_stride_helper(std::initializer_list<const stride_vector*> ilist)
    {
        TBLIS_ASSERT(ilist.size() == N);
        std::copy_n(ilist.begin(), N, strides.begin());
    }

    bool operator()(int i, int j) const
    {
        auto min_i = (*strides[0])[i];
        auto min_j = (*strides[0])[j];

        for (auto k : range(1,N))
        {
            min_i = std::min(min_i, (*strides[k])[i]);
            min_j = std::min(min_j, (*strides[k])[j]);
        }

        if (min_i < min_j) return true;
        if (min_i > min_j) return false;

        for (auto k : range(N))
        {
            auto s_i = (*strides[k])[i];
            auto s_j = (*strides[k])[j];
            if (s_i < s_j) return true;
            if (s_i > s_j) return false;
        }

        return false;
    }
};

inline int check_sizes() { return 0; }

template <typename T, typename... Ts>
int check_sizes(const T& arg, const Ts&... args)
{
    int sz = arg.size();
    TBLIS_ASSERT(sizeof...(Ts) == 0 || sz == check_sizes(args...));
    return sz;
}

template <typename... Strides>
dim_vector sort_by_stride(const Strides&... strides)
{
    dim_vector idx = range(check_sizes(strides...));
    std::sort(idx.begin(), idx.end(), sort_by_stride_helper<sizeof...(Strides)>{&strides...});
    return idx;
}

template <typename T>
bool are_congruent_along(const marray_view<const T>& A,
                         const marray_view<const T>& B, int dim)
{
    if (A.dimension() < B.dimension()) swap(A, B);

    auto ndim = A.dimension();
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

inline bool are_compatible(const len_vector& len_A,
                           const stride_vector& stride_A,
                           const len_vector& len_B,
                           const stride_vector& stride_B)
{
    TBLIS_ASSERT(len_A.size() == stride_A.size());
    auto dims_A = detail::sort_by_stride(stride_A);
    auto len_Ar = stl_ext::permuted(len_A, dims_A);
    auto stride_Ar = stl_ext::permuted(stride_A, dims_A);

    TBLIS_ASSERT(len_B.size() == stride_B.size());
    auto dims_B = detail::sort_by_stride(stride_B);
    auto len_Br = stl_ext::permuted(len_B, dims_B);
    auto stride_Br = stl_ext::permuted(stride_B, dims_B);

    if (stl_ext::prod(len_Ar) != stl_ext::prod(len_Br))
        return false;

    viterator<> it_A(len_Ar, stride_Ar);
    viterator<> it_B(len_Br, stride_Br);

    stride_type off_A = 0, off_B = 0;
    while (it_A.next(off_A) + it_B.next(off_B))
        if (off_A != off_B) return false;

    return true;
}

template <typename T>
bool are_compatible(const marray_view<const T>& A,
                    const marray_view<const T>& B)
{
    return A.data() == B.data() &&
        are_compatible(A.lengths(), A.strides(),
                       B.lengths(), B.strides());
}

}

template <typename... Strides>
void fold(len_vector& lengths, label_vector& idx, stride_vector& stride0, Strides&... _strides)
{
    if (lengths.empty()) return;

    constexpr auto N = sizeof...(Strides)+1;
    std::array<stride_vector*,N> strides{&stride0, &_strides...};

    auto ndim = lengths.size();
    auto inds = detail::sort_by_stride(stride0);

    label_vector oldidx;
    len_vector oldlengths;
    std::array<stride_vector,N> oldstrides;

    oldidx.swap(idx);
    oldlengths.swap(lengths);
    for (auto i : range(N))
        oldstrides[i].swap(*strides[i]);

    idx.push_back(oldidx[inds[0]]);
    lengths.push_back(oldlengths[inds[0]]);
    for (auto i : range(N))
        strides[i]->push_back(oldstrides[i][inds[0]]);

    for (auto i : range(1,ndim))
    {
        bool contig = true;
        for (auto j : range(N))
            if (oldstrides[j][inds[i]] != oldstrides[j][inds[i-1]]*oldlengths[inds[i-1]])
                contig = false;

        if (contig)
        {
            lengths.back() *= oldlengths[inds[i]];
        }
        else
        {
            idx.push_back(oldidx[inds[i]]);
            lengths.push_back(oldlengths[inds[i]]);
            for (auto j : range(N))
                strides[j]->push_back(oldstrides[j][inds[i]]);
        }
    }

    for (auto i : range(N))
        TBLIS_ASSERT(detail::are_compatible(oldlengths, oldstrides[i],
                                            lengths, *strides[i]));
}

inline void diagonal(int& ndim,
                     const len_type* len_in,
                     const stride_type* stride_in,
                     const label_type* idx_in,
                     len_vector& len_out,
                     stride_vector& stride_out,
                     label_vector& idx_out)
{
    len_out.reserve(ndim);
    stride_out.reserve(ndim);
    idx_out.reserve(ndim);

    dim_vector inds = range(ndim);
    stl_ext::sort(inds, detail::sort_by_idx(idx_in));

    auto ndim_in = ndim;

    ndim = 0;
    for (auto i : range(ndim_in))
    {
        if (i == 0 || idx_in[inds[i]] != idx_in[inds[i-1]])
        {
            if (len_in[inds[i]] != 1)
            {
                len_out.push_back(len_in[inds[i]]);
                stride_out.push_back(stride_in[inds[i]]);
                idx_out.push_back(idx_in[inds[i]]);
                ndim++;
            }
        }
        else if (len_in[inds[i]] != 1)
        {
            TBLIS_ASSERT(len_out[ndim-1] == len_in[inds[i]]);
            if (len_in[inds[i]] != 1)
                stride_out[ndim-1] += stride_in[inds[i]];
        }
    }
}

template <typename T>
matrix_view<T> matricize(const marray_view<T>& A, int split)
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
matrix_view<T> matricize(marray<T>& A, int split)
{
    return matricize(A.view(), split);
}

template <typename T>
matrix_view<const T> matricize(const marray<T>& A, int split)
{
    return matricize(A.view(), split);
}

inline int unit_dim(const stride_vector& stride, const dim_vector& reorder)
{
    for (auto i : range(reorder.size()))
        if (stride[reorder[i]] == 1)
            return i;

    return reorder.size();
}

}

#endif
