#ifndef _TBLIS_TENSOR_DETAIL_HPP_
#define _TBLIS_TENSOR_DETAIL_HPP_

#include <algorithm>
#include <initializer_list>
#include <string>
#include <vector>

#include "external/stl_ext/include/type_traits.hpp"

#include "tblis_assert.hpp"
#include "tblis_basic_types.hpp"
#include "tblis_marray.hpp"

namespace tblis
{
namespace detail
{

struct sort_by_idx_helper
{
    const std::string& idx;

    sort_by_idx_helper(const std::string& idx) : idx(idx) {}

    bool operator()(unsigned i, unsigned j) const
    {
        return idx[i] < idx[j];
    }
};

inline sort_by_idx_helper sort_by_idx(const std::string& idx)
{
    return sort_by_idx_helper(idx);
}

template <unsigned N>
struct sort_by_stride_helper
{
    std::array<const std::vector<stride_type>*, N> strides;

    sort_by_stride_helper(std::initializer_list<const std::vector<stride_type>*> ilist)
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
size_t check_sizes(const T&... arg, const Ts&... args)
{
    size_t sz = arg.size();
    if (sizeof...(Ts)) TBLIS_ASSERT(sz == check_sizes(args...));
    return sz;
}

template <typename... Strides>
std::vector<unsigned> sort_by_stride(const Strides&... strides)
{
    std::vector<unsigned> idx = MArray::range<unsigned>(check_sizes(strides...));
    std::sort(idx.begin(), idx.end(), sort_by_stride_helper<sizeof...(Strides)>{{&strides...}});
    return idx;
}

template <typename T>
bool are_congruent_along(const const_tensor_view<T>& A,
                         const const_tensor_view<T>& B, unsigned dim)
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

inline bool are_compatible(const std::vector<len_type>& len_A,
                           const std::vector<stride_type>& stride_A,
                           const std::vector<len_type>& len_B,
                           const std::vector<stride_type>& stride_B)
{
    TBLIS_ASSERT(len_A.size() == stride_A.size());
    std::vector<size_t> dims_A = range(len_A.size());
    stl_ext::sort(dims_A, detail::sort_by_stride(stride_A));
    auto len_Ar = stl_ext::permuted(len_A, dims_A);
    auto stride_Ar = stl_ext::permuted(stride_A, dims_A);

    TBLIS_ASSERT(len_B.size() == stride_B.size());
    std::vector<size_t> dims_B = range(len_B.size());
    stl_ext::sort(dims_B, detail::sort_by_stride(stride_B));
    auto len_Br = stl_ext::permuted(len_B, dims_B);
    auto stride_Br = stl_ext::permuted(stride_B, dims_B);

    if (stl_ext::prod(len_Ar) != stl_ext::prod(len_Br))
        return false;

    MArray::viterator<> it_A(len_Ar, stride_Ar);
    MArray::viterator<> it_B(len_Br, stride_Br);

    stride_type off_A = 0, off_B = 0;
    while (it_A.next(off_A) + it_B.next(off_B))
        if (off_A != off_B) return false;

    return true;
}

template <typename T>
bool are_compatible(const const_tensor_view<T>& A,
                    const const_tensor_view<T>& B)
{
    return A.data() == B.data() &&
        are_compatible(A.lengths(), A.strides(),
                       B.lengths(), B.strides());
}

template <typename T>
int check_tensor_indices(const const_tensor_view<T>& A, const std::string& idx_A)
{
    using stl_ext::sort;

    std::vector<std::pair<char,len_type> > idx_len;
    idx_len.reserve(A.dimension());

    TBLIS_ASSERT(idx_A.size() == A.dimension());

    for (unsigned i = 0;i < A.dimension();i++)
    {
        idx_len.emplace_back(idx_A[i], A.length(i));
    }

    sort(idx_len);

    for (unsigned i = 1;i < idx_len.size();i++)
    {
        if (idx_len[i].first  == idx_len[i-1].first)
            TBLIS_ASSERT(idx_len[i].second == idx_len[i-1].second);
    }

    return 0;
}

template <typename T>
int check_tensor_indices(const tensor_view<T>& A, const std::string& idx_A)
{
    return check_tensor_indices(reinterpret_cast<const const_tensor_view<T>&>(A), idx_A);
}

template <typename T>
int check_tensor_indices(const const_tensor_view<T>& A, std::string idx_A,
                         const const_tensor_view<T>& B, std::string idx_B,
                         bool has_A_only, bool has_B_only, bool has_AB)
{
    using stl_ext::sort;
    using stl_ext::unique;
    using stl_ext::intersection;
    using stl_ext::exclusion;

    std::vector<std::pair<char,len_type>> idx_len;
    idx_len.reserve(A.dimension()+
                    B.dimension());

    TBLIS_ASSERT(idx_A.size() == A.dimension());
    TBLIS_ASSERT(idx_B.size() == B.dimension());

    for (unsigned i = 0;i < A.dimension();i++)
    {
        idx_len.emplace_back(idx_A[i], A.length(i));
    }

    for (unsigned i = 0;i < B.dimension();i++)
    {
        idx_len.emplace_back(idx_B[i], B.length(i));
    }

    sort(idx_len);

    for (unsigned i = 1;i < idx_len.size();i++)
    {
        if (idx_len[i].first  == idx_len[i-1].first)
            TBLIS_ASSERT(idx_len[i].second == idx_len[i-1].second);
    }

    unique(idx_A);
    unique(idx_B);

    auto idx_AB = intersection(idx_A, idx_B);
    auto idx_A_only = exclusion(idx_A, idx_B);
    auto idx_B_only = exclusion(idx_B, idx_A);

    TBLIS_ASSERT(idx_A_only.empty() || has_A_only);
    TBLIS_ASSERT(idx_B_only.empty() || has_B_only);
    TBLIS_ASSERT(idx_AB.empty()     || has_AB);

    return 0;
}

template <typename T>
int check_tensor_indices(const const_tensor_view<T>& A, std::string idx_A,
                         const       tensor_view<T>& B, std::string idx_B,
                         bool has_A_only, bool has_B_only, bool has_AB)
{
    return check_tensor_indices(A, idx_A,
                                reinterpret_cast<const const_tensor_view<T>&>(B), idx_B,
                                has_A_only, has_B_only, has_AB);
}

template <typename T>
int check_tensor_indices(const const_tensor_view<T>& A, std::string idx_A,
                         const const_tensor_view<T>& B, std::string idx_B,
                         const       tensor_view<T>& C, std::string idx_C,
                         bool has_A_only, bool has_B_only, bool has_C_only,
                         bool has_AB, bool has_AC, bool has_BC,
                         bool has_ABC)
{
    using stl_ext::sort;
    using stl_ext::unique;
    using stl_ext::intersection;
    using stl_ext::exclusion;

    std::vector<std::pair<char,len_type>> idx_len;
    idx_len.reserve(A.dimension()+
                    B.dimension()+
                    C.dimension());

    TBLIS_ASSERT(idx_A.size() == A.dimension());
    TBLIS_ASSERT(idx_B.size() == B.dimension());
    TBLIS_ASSERT(idx_C.size() == C.dimension());

    for (unsigned i = 0;i < A.dimension();i++)
    {
        idx_len.emplace_back(idx_A[i], A.length(i));
    }

    for (unsigned i = 0;i < B.dimension();i++)
    {
        idx_len.emplace_back(idx_B[i], B.length(i));
    }

    for (unsigned i = 0;i < C.dimension();i++)
    {
        idx_len.emplace_back(idx_C[i], C.length(i));
    }

    sort(idx_len);

    for (unsigned i = 1;i < idx_len.size();i++)
    {
        if (idx_len[i].first  == idx_len[i-1].first)
            TBLIS_ASSERT(idx_len[i].second == idx_len[i-1].second);
    }

    unique(idx_A);
    unique(idx_B);
    unique(idx_C);

    auto idx_ABC = intersection(idx_A, idx_B, idx_C);
    auto idx_AB = exclusion(intersection(idx_A, idx_B), idx_C);
    auto idx_AC = exclusion(intersection(idx_A, idx_C), idx_B);
    auto idx_BC = exclusion(intersection(idx_B, idx_C), idx_A);
    auto idx_A_only = exclusion(idx_A, idx_B, idx_C);
    auto idx_B_only = exclusion(idx_B, idx_A, idx_C);
    auto idx_C_only = exclusion(idx_C, idx_A, idx_B);

    TBLIS_ASSERT(idx_A_only.empty() || has_A_only);
    TBLIS_ASSERT(idx_B_only.empty() || has_B_only);
    TBLIS_ASSERT(idx_C_only.empty() || has_C_only);
    TBLIS_ASSERT(idx_AB.empty()     || has_AB);
    TBLIS_ASSERT(idx_AC.empty()     || has_AC);
    TBLIS_ASSERT(idx_BC.empty()     || has_BC);
    TBLIS_ASSERT(idx_ABC.empty()    || has_ABC);

    return 0;
}

template <typename T, typename=void>
struct pointer_type;

template <typename T>
struct pointer_type<T, stl_ext::enable_if_t<std::is_pointer<stl_ext::decay_t<T>>::value>>
{
    typedef stl_ext::remove_pointer_t<stl_ext::decay_t<T>> type;
};

template <typename T>
struct pointer_type<T, stl_ext::has_member<decltype(std::declval<T>().data())>>
{
    typedef stl_ext::remove_pointer_t<decltype(std::declval<T>().data())> type;
};

template <typename T>
using pointer_type_t = typename pointer_type<T>::type;

template <typename... Args> struct check_template_types;

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
          typename U>
struct check_template_types<T, A_ptr, A_len, A_stride, A_idx, U>
{
    typedef stl_ext::enable_if<(std::is_same<float,T>::value ||
                                std::is_same<double,T>::value ||
                                std::is_same<scomplex,T>::value ||
                                std::is_same<dcomplex,T>::value) &&
                               std::is_same<pointer_type_t<A_ptr>,T>::value &&
                               std::is_integral<pointer_type_t<A_len>>::value &&
                               std::is_integral<pointer_type_t<A_stride>>::value &&
                               std::is_integral<pointer_type_t<A_idx>>::value,
                               U> type;
};

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx,
          typename U>
struct check_template_types<T, A_ptr, A_len, A_stride, A_idx,
                               B_ptr, B_len, B_stride, B_idx, U>
{
    typedef stl_ext::enable_if<(std::is_same<float,T>::value ||
                                std::is_same<double,T>::value ||
                                std::is_same<scomplex,T>::value ||
                                std::is_same<dcomplex,T>::value) &&
                               std::is_same<pointer_type_t<A_ptr>,T>::value &&
                               std::is_same<pointer_type_t<B_ptr>,T>::value &&
                               std::is_integral<pointer_type_t<A_len>>::value &&
                               std::is_integral<pointer_type_t<B_len>>::value &&
                               std::is_integral<pointer_type_t<A_stride>>::value &&
                               std::is_integral<pointer_type_t<B_stride>>::value &&
                               std::is_integral<pointer_type_t<A_idx>>::value &&
                               std::is_integral<pointer_type_t<B_idx>>::value,
                               U> type;
};

template <typename T, typename A_ptr, typename A_len, typename A_stride, typename A_idx,
                      typename B_ptr, typename B_len, typename B_stride, typename B_idx,
                      typename C_ptr, typename C_len, typename C_stride, typename C_idx,
          typename U>
struct check_template_types<T, A_ptr, A_len, A_stride, A_idx,
                               B_ptr, B_len, B_stride, B_idx,
                               C_ptr, C_len, C_stride, C_idx, U>
{
    typedef stl_ext::enable_if<(std::is_same<float,T>::value ||
                                std::is_same<double,T>::value ||
                                std::is_same<scomplex,T>::value ||
                                std::is_same<dcomplex,T>::value) &&
                               std::is_same<pointer_type_t<A_ptr>,T>::value &&
                               std::is_same<pointer_type_t<B_ptr>,T>::value &&
                               std::is_same<pointer_type_t<C_ptr>,T>::value &&
                               std::is_integral<pointer_type_t<A_len>>::value &&
                               std::is_integral<pointer_type_t<B_len>>::value &&
                               std::is_integral<pointer_type_t<C_len>>::value &&
                               std::is_integral<pointer_type_t<A_stride>>::value &&
                               std::is_integral<pointer_type_t<B_stride>>::value &&
                               std::is_integral<pointer_type_t<C_stride>>::value &&
                               std::is_integral<pointer_type_t<A_idx>>::value &&
                               std::is_integral<pointer_type_t<B_idx>>::value &&
                               std::is_integral<pointer_type_t<C_idx>>::value,
                               U> type;
};

template <typename... Args>
using check_template_types_t = typename check_template_types<Args...>::type;

template <typename Len>
stl_ext::enable_if_t<std::is_pointer<Len>::value,std::vector<len_type>>
make_len(unsigned ndim, const Len& x)
{
    return {x, x+ndim};
}

template <typename Len>
stl_ext::enable_if_t<!std::is_pointer<Len>::value,std::vector<len_type>>
make_len(unsigned ndim, const Len& x)
{
    TBLIS_ASSERT(x.size() == ndim);
    return {x.data(), x.data()+ndim};
}

template <typename Stride>
stl_ext::enable_if_t<std::is_pointer<Stride>::value,std::vector<stride_type>>
make_stride(unsigned ndim, const Stride& x)
{
    return {x, x+ndim};
}

template <typename Stride>
stl_ext::enable_if_t<!std::is_pointer<Stride>::value,std::vector<stride_type>>
make_stride(unsigned ndim, const Stride& x)
{
    TBLIS_ASSERT(x.size() == ndim);
    return {x.data(), x.data()+ndim};
}

template <typename Idx>
stl_ext::enable_if_t<std::is_pointer<Idx>::value,std::string>
make_idx(unsigned ndim, const Idx& x)
{
    return {x, x+ndim};
}

template <typename Idx>
stl_ext::enable_if_t<!std::is_pointer<Idx>::value,std::string>
make_idx(unsigned ndim, const Idx& x)
{
    TBLIS_ASSERT(x.size() == ndim);
    return {x.data(), x.data()+ndim};
}

template <typename Ptr>
stl_ext::enable_if_t<std::is_pointer<Ptr>::value,pointer_type_t<Ptr>*>
make_ptr(Ptr& x)
{
    return x;
}

template <typename Ptr>
stl_ext::enable_if_t<std::is_pointer<Ptr>::value,const pointer_type_t<Ptr>*>
make_ptr(const Ptr& x)
{
    return x;
}

template <typename Ptr>
stl_ext::enable_if_t<!std::is_pointer<Ptr>::value,pointer_type_t<Ptr>*>
make_ptr(Ptr& x)
{
    return x.data();
}

template <typename Ptr>
stl_ext::enable_if_t<!std::is_pointer<Ptr>::value,const pointer_type_t<Ptr>*>
make_ptr(const Ptr& x)
{
    return x.data();
}

}
}

#endif
