#ifndef _TBLIS_UTIL_TENSOR_CHECK_HPP_
#define _TBLIS_UTIL_TENSOR_CHECK_HPP_

#include <algorithm>
#include <utility>
#include <cassert>
#include <iostream>

namespace tblis
{

template <typename T>
int check_tensor_indices(const const_tensor_view<T>& A, const std::string& idx_A)
{
    using stl_ext::sort;

    std::vector<std::pair<char,dim_t> > idx_len;
    idx_len.reserve(A.dimension());

    assert(idx_A.size() == A.dimension());

    for (gint_t i = 0;i < A.dimension();i++)
    {
        idx_len.emplace_back(idx_A[i], A.length(i));
    }

    sort(idx_len);

    for (gint_t i = 1;i < idx_len.size();i++)
    {
        if (idx_len[i].first  == idx_len[i-1].first)
            assert(idx_len[i].second == idx_len[i-1].second);
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

    std::vector<std::pair<char,dim_t>> idx_len;
    idx_len.reserve(A.dimension()+
                    B.dimension());

    assert(idx_A.size() == A.dimension());
    assert(idx_B.size() == B.dimension());

    for (gint_t i = 0;i < A.dimension();i++)
    {
        idx_len.emplace_back(idx_A[i], A.length(i));
    }

    for (gint_t i = 0;i < B.dimension();i++)
    {
        idx_len.emplace_back(idx_B[i], B.length(i));
    }

    sort(idx_len);

    for (gint_t i = 1;i < idx_len.size();i++)
    {
        if (idx_len[i].first  == idx_len[i-1].first)
            assert(idx_len[i].second == idx_len[i-1].second);
    }

    unique(idx_A);
    unique(idx_B);

    auto idx_AB = intersection(idx_A, idx_B);
    auto idx_A_only = exclusion(idx_A, idx_B);
    auto idx_B_only = exclusion(idx_B, idx_A);

    assert(idx_A_only.empty() || has_A_only);
    assert(idx_B_only.empty() || has_B_only);
    assert(idx_AB.empty()     || has_AB);

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
int check_tensor_indices(const tensor_view<T>& A, std::string idx_A,
                         const tensor_view<T>& B, std::string idx_B,
                         bool has_A_only, bool has_B_only, bool has_AB)
{
    return check_tensor_indices(reinterpret_cast<const const_tensor_view<T>&>(A), idx_A,
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

    std::vector<std::pair<char,dim_t>> idx_len;
    idx_len.reserve(A.dimension()+
                    B.dimension()+
                    C.dimension());

    assert(idx_A.size() == A.dimension());
    assert(idx_B.size() == B.dimension());
    assert(idx_C.size() == C.dimension());

    for (gint_t i = 0;i < A.dimension();i++)
    {
        idx_len.emplace_back(idx_A[i], A.length(i));
    }

    for (gint_t i = 0;i < B.dimension();i++)
    {
        idx_len.emplace_back(idx_B[i], B.length(i));
    }

    for (gint_t i = 0;i < C.dimension();i++)
    {
        idx_len.emplace_back(idx_C[i], C.length(i));
    }

    sort(idx_len);

    for (gint_t i = 1;i < idx_len.size();i++)
    {
        if (idx_len[i].first  == idx_len[i-1].first)
            assert(idx_len[i].second == idx_len[i-1].second);
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

    assert(idx_A_only.empty() || has_A_only);
    assert(idx_B_only.empty() || has_B_only);
    assert(idx_C_only.empty() || has_C_only);
    assert(idx_AB.empty()     || has_AB);
    assert(idx_AC.empty()     || has_AC);
    assert(idx_BC.empty()     || has_BC);
    assert(idx_ABC.empty()    || has_ABC);

    return 0;
}

}

#endif
