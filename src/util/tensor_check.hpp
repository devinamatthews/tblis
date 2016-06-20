#ifndef _TBLIS_UTIL_TENSOR_CHECK_HPP_
#define _TBLIS_UTIL_TENSOR_CHECK_HPP_

#include <algorithm>
#include <utility>
#include <cassert>
#include <iostream>

namespace tblis
{

template <typename T>
int check_tensor_indices(const Tensor<T>& A, const std::string& idx_A)
{
    std::vector<std::pair<char,dim_t> > idx_len;
    idx_len.reserve(A.dimension());

    if (idx_A.size() != A.dimension())
    {
        abort();
    }

    for (gint_t i = 0;i < A.dimension();i++)
    {
        idx_len.push_back(std::make_pair(idx_A[i], A.length(i)));
    }

    sort(idx_len);

    for (gint_t i = 1;i < idx_len.size();i++)
    {
        if (idx_len[i].first  == idx_len[i-1].first &&
            idx_len[i].second != idx_len[i-1].second)
        {
            abort();
        }
    }

    return 0;
}

template <typename T>
int check_tensor_indices(const Tensor<T>& A, std::string idx_A,
                         const Tensor<T>& B, std::string idx_B,
                         bool has_A_only, bool has_B_only, bool has_AB)
{
    std::vector<std::pair<char,dim_t> > idx_len;
    idx_len.reserve(A.dimension()+
                    B.dimension());

    if (idx_A.size() != A.dimension() ||
        idx_B.size() != B.dimension())
    {
        abort();
    }

    for (gint_t i = 0;i < A.dimension();i++)
    {
        idx_len.push_back(std::make_pair(idx_A[i], A.length(i)));
    }

    for (gint_t i = 0;i < B.dimension();i++)
    {
        idx_len.push_back(std::make_pair(idx_B[i], B.length(i)));
    }

    sort(idx_len);

    for (gint_t i = 1;i < idx_len.size();i++)
    {
        if (idx_len[i].first  == idx_len[i-1].first &&
            idx_len[i].second != idx_len[i-1].second)
        {
            abort();
        }
    }

    unique(sort(idx_A));
    unique(sort(idx_B));

    std::string idx_A_only, idx_B_only;
    std::string idx_AB;

    set_intersection(idx_A, idx_B, idx_AB);

    set_difference(idx_A, idx_AB, idx_A_only);
    set_difference(idx_B, idx_AB, idx_B_only);

    assert(idx_A_only.empty() || has_A_only);
    assert(idx_B_only.empty() || has_B_only);
    assert(idx_AB.empty()     || has_AB);

    return 0;
}

template <typename T>
int check_tensor_indices(const Tensor<T>& A, std::string idx_A,
                         const Tensor<T>& B, std::string idx_B,
                         const Tensor<T>& C, std::string idx_C,
                         bool has_A_only, bool has_B_only, bool has_C_only,
                         bool has_AB, bool has_AC, bool has_BC,
                         bool has_ABC)
{
    std::vector<std::pair<char,dim_t> > idx_len;
    idx_len.reserve(A.dimension()+
                    B.dimension()+
                    C.dimension());

    if (idx_A.size() != A.dimension() ||
        idx_B.size() != B.dimension() ||
        idx_C.size() != C.dimension())
    {
        abort();
    }

    for (gint_t i = 0;i < A.dimension();i++)
    {
        idx_len.push_back(std::make_pair(idx_A[i], A.length(i)));
    }

    for (gint_t i = 0;i < B.dimension();i++)
    {
        idx_len.push_back(std::make_pair(idx_B[i], B.length(i)));
    }

    for (gint_t i = 0;i < C.dimension();i++)
    {
        idx_len.push_back(std::make_pair(idx_C[i], C.length(i)));
    }

    sort(idx_len);

    for (gint_t i = 1;i < idx_len.size();i++)
    {
        if (idx_len[i].first  == idx_len[i-1].first &&
            idx_len[i].second != idx_len[i-1].second)
        {
            abort();
        }
    }

    unique(sort(idx_A));
    unique(sort(idx_B));
    unique(sort(idx_C));

    std::string idx_A_only, idx_B_only, idx_C_only;
    std::string idx_AB, idx_AC, idx_BC;
    std::string idx_ABC;

    set_intersection(
    set_intersection(idx_A, idx_B, idx_ABC), idx_C);

    set_difference(
    set_difference(idx_A, idx_B, idx_A_only), idx_C);
    set_difference(
    set_difference(idx_B, idx_C, idx_B_only), idx_A);
    set_difference(
    set_difference(idx_C, idx_A, idx_C_only), idx_B);

    set_difference(
    set_intersection(idx_A, idx_B, idx_AB), idx_ABC);
    set_difference(
    set_intersection(idx_A, idx_C, idx_AC), idx_ABC);
    set_difference(
    set_intersection(idx_B, idx_C, idx_BC), idx_ABC);

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
