#ifndef _TBLIS_UTIL_TENSOR_HPP_
#define _TBLIS_UTIL_TENSOR_HPP_

#include "util/basic_types.h"
#include "util/assert.h"

#include "external/stl_ext/include/algorithm.hpp"
#include "external/stl_ext/include/type_traits.hpp"
#include "external/stl_ext/include/vector.hpp"

#include <initializer_list>
#include <string>

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

template <typename T>
dim_vector relative_permutation(const T& a, const T& b)
{
    dim_vector perm; perm.reserve(a.size());

    for (auto& e : b)
    {
        for (unsigned i = 0;i < a.size();i++)
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

    bool operator()(unsigned i, unsigned j) const
    {
        return idx[i] < idx[j];
    }
};

inline sort_by_idx_helper sort_by_idx(const label_type* idx)
{
    return sort_by_idx_helper(idx);
}

template <unsigned N>
struct sort_by_stride_helper
{
    std::array<const stride_vector*, N> strides;

    sort_by_stride_helper(std::initializer_list<const stride_vector*> ilist)
    {
        TBLIS_ASSERT(ilist.size() == N);
        std::copy_n(ilist.begin(), N, strides.begin());
    }

    bool operator()(unsigned i, unsigned j) const
    {
        for (size_t k = 0;k < N;k++)
        {
            auto s_i = (*strides[k])[i];
            auto s_j = (*strides[k])[j];
            if (s_i < s_j) return true;
            if (s_i > s_j) return false;
        }

        return false;
    }
};

inline size_t check_sizes() { return 0; }

template <typename T, typename... Ts>
size_t check_sizes(const T& arg, const Ts&... args)
{
    size_t sz = arg.size();
    if (sizeof...(Ts)) TBLIS_ASSERT(sz == check_sizes(args...));
    return sz;
}

template <typename... Strides>
dim_vector sort_by_stride(const Strides&... strides)
{
    dim_vector idx = range(static_cast<unsigned>(check_sizes(strides...)));
    std::sort(idx.begin(), idx.end(), sort_by_stride_helper<sizeof...(Strides)>{&strides...});
    return idx;
}

template <typename T>
bool are_congruent_along(const varray_view<const T>& A,
                         const varray_view<const T>& B, unsigned dim)
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
bool are_compatible(const varray_view<const T>& A,
                    const varray_view<const T>& B)
{
    return A.data() == B.data() &&
        are_compatible(A.lengths(), A.strides(),
                       B.lengths(), B.strides());
}

template <size_t I, size_t N, typename... Strides>
struct swap_strides_helper
{
    swap_strides_helper(std::tuple<Strides&...>& strides,
                        std::tuple<Strides...>& oldstrides)
    {
        std::get<I>(strides).swap(std::get<I>(oldstrides));
        swap_strides_helper<I+1, N, Strides...>(strides, oldstrides);
    }
};

template <size_t N, typename... Strides>
struct swap_strides_helper<N, N, Strides...>
{
    swap_strides_helper(std::tuple<Strides&...>&,
                        std::tuple<Strides...>&) {}
};

template <typename... Strides>
void swap_strides(std::tuple<Strides&...>& strides,
                  std::tuple<Strides...>& oldstrides)
{
    swap_strides_helper<0, sizeof...(Strides), Strides...>(strides, oldstrides);
}

template <size_t I, size_t N, typename... Strides>
struct are_contiguous_helper
{
    bool operator()(std::tuple<Strides...>& strides,
                    const len_vector& lengths,
                    unsigned i, unsigned im1)
    {
        return std::get<I>(strides)[i] == std::get<I>(strides)[im1]*lengths[im1] &&
            are_contiguous_helper<I+1, N, Strides...>()(strides, lengths, i, im1);
    }
};

template <size_t N, typename... Strides>
struct are_contiguous_helper<N, N, Strides...>
{
    bool operator()(std::tuple<Strides...>&,
                    const len_vector&,
                    unsigned, unsigned)
    {
        return true;
    }
};

template <typename... Strides>
bool are_contiguous(std::tuple<Strides...>& strides,
                    const len_vector& lengths,
                    unsigned i, unsigned im1)
{
    return are_contiguous_helper<0, sizeof...(Strides), Strides...>()(strides, lengths, i, im1);
}

template <size_t I, size_t N, typename... Strides>
struct push_back_strides_helper
{
    push_back_strides_helper(std::tuple<Strides&...>& strides,
                             std::tuple<Strides...>& oldstrides, unsigned i)
    {
        std::get<I>(strides).push_back(std::get<I>(oldstrides)[i]);
        push_back_strides_helper<I+1, N, Strides...>(strides, oldstrides, i);
    }
};

template <size_t N, typename... Strides>
struct push_back_strides_helper<N, N, Strides...>
{
    push_back_strides_helper(std::tuple<Strides&...>&,
                             std::tuple<Strides...>&, unsigned) {}
};

template <typename... Strides>
void push_back_strides(std::tuple<Strides&...>& strides,
                       std::tuple<Strides...>& oldstrides, unsigned i)
{
    push_back_strides_helper<0, sizeof...(Strides), Strides...>(strides, oldstrides, i);
}

template <size_t I, size_t N, typename... Strides>
struct are_compatible_helper
{
    bool operator()(const len_vector& len_A,
                    const std::tuple<Strides...>& stride_A,
                    const len_vector& len_B,
                    const std::tuple<Strides&...>& stride_B)
    {
        return are_compatible(len_A, std::get<I>(stride_A),
                              len_B, std::get<I>(stride_B)) &&
            are_compatible_helper<I+1, N, Strides...>()(len_A, stride_A,
                                                        len_B, stride_B);
    }
};

template <size_t N, typename... Strides>
struct are_compatible_helper<N, N, Strides...>
{
    bool operator()(const len_vector&,
                    const std::tuple<Strides...>&,
                    const len_vector&,
                    const std::tuple<Strides&...>&)
    {
        return true;
    }
};

template <typename... Strides>
bool are_compatible(const len_vector& len_A,
                    const std::tuple<Strides...>& stride_A,
                    const len_vector& len_B,
                    const std::tuple<Strides&...>& stride_B)
{
    return are_compatible_helper<0, sizeof...(Strides), Strides...>()(
        len_A, stride_A, len_B, stride_B);
}

}

template <typename... Strides>
void fold(len_vector& lengths, label_vector& idx,
          Strides&... _strides)
{
    std::tuple<Strides&...> strides(_strides...);

    auto ndim = lengths.size();
    auto inds = detail::sort_by_stride(std::get<0>(strides));

    label_vector oldidx;
    len_vector oldlengths;
    std::tuple<Strides...> oldstrides;

    oldidx.swap(idx);
    oldlengths.swap(lengths);
    detail::swap_strides(strides, oldstrides);

    for (unsigned i = 0;i < ndim;i++)
    {
        if (i != 0 && detail::are_contiguous(oldstrides, oldlengths, inds[i], inds[i-1]))
        {
            lengths.back() *= oldlengths[inds[i]];
        }
        else
        {
            idx.push_back(oldidx[inds[i]]);
            lengths.push_back(oldlengths[inds[i]]);
            detail::push_back_strides(strides, oldstrides, inds[i]);
        }
    }

    TBLIS_ASSERT(detail::are_compatible(oldlengths, oldstrides,
                                        lengths, strides));
}

inline void diagonal(unsigned& ndim,
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

    unsigned ndim_in = ndim;

    ndim = 0;
    for (unsigned i = 0;i < ndim_in;i++)
    {
        if (i == 0 || idx_in[inds[i]] != idx_in[inds[i-1]])
        {
            if (len_in[inds[i]] != 1 || (i == ndim_in-1 && ndim == 0))
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
void matricize(varray_view<const T>  A,
               matrix_view<const T>& AM, unsigned split)
{
    unsigned ndim = A.dimension();
    TBLIS_ASSERT(split <= ndim);
    if (ndim > 0 && A.stride(0) < A.stride(ndim-1))
    {
        for (unsigned i = 1;i < split;i++)
            TBLIS_ASSERT(A.stride(i) == A.stride(i-1)*A.length(i-1));
        for (unsigned i = split+1;i < ndim;i++)
            TBLIS_ASSERT(A.stride(i) == A.stride(i-1)*A.length(i-1));
    }
    else
    {
        for (unsigned i = 0;i+1 < split;i++)
            TBLIS_ASSERT(A.stride(i) == A.stride(i+1)*A.length(i+1));
        for (unsigned i = split;i+1 < ndim;i++)
            TBLIS_ASSERT(A.stride(i) == A.stride(i+1)*A.length(i+1));
    }

    len_type m = 1;
    for (unsigned i = 0;i < split;i++)
    {
        m *= A.length(i);
    }

    len_type n = 1;
    for (unsigned i = split;i < ndim;i++)
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

    AM.reset({m, n}, A.data(), {rs, cs});
}

template <typename T>
void matricize(varray_view<T>  A,
               matrix_view<T>& AM, unsigned split)
{
    matricize<T>(A, reinterpret_cast<matrix_view<const T>&>(AM), split);
}

}

#endif
