#ifndef _TENSOR_UTIL_TENSOR_CHECK_HPP_
#define _TENSOR_UTIL_TENSOR_CHECK_HPP_

#include <algorithm>
#include <utility>
#include <cassert>
#include <iostream>

namespace tensor
{

template <typename T> class Tensor;

namespace util
{

template <typename len_type, typename stride_type>
int check_tensor(gint_t ndim, const len_type len, const stride_type stride)
{
    if (ndim < 0)
    {
        abort();
    }

    if (ndim > 0 && !len)
    {
        abort();
    }

    #if !TENSOR_ALLOW_NULL_STRIDE
    if (ndim > 0 && !stride)
    {
        abort();
    }
    #endif

    for (gint_t i = 0;i < ndim;i++)
    {
        if (len[i] <= 0)
        {
            abort();
        }

        #if TENSOR_REQUIRE_POSITIVE_STRIDES
        if (!!stride && stride[i] <= 0)
        {
            abort();
        }
        #endif
    }

    #if TENSOR_REQUIRE_DISJOINT_STRIDES

    if (!!stride)
    {
        std::vector<std::pair<inc_t,gint_t> > stride_idx;
        stride_idx.reserve(ndim);

        for (gint_t i = 0;i < ndim;i++)
        {
            if (len[i] > 1)
            stride_idx.push_back(std::make_pair(std::abs(stride[i]), i));
        }

        sort(stride_idx);

        for (gint_t i = 1;i < stride_idx.size();i++)
        {
            if (stride_idx[i].first < stride_idx[i-1].first*len[stride_idx[i-1].second])
            {
                abort();
            }
        }
    }

    #endif

    return 0;
}

template <typename len_type, typename stride_type>
int check_tensor(gint_t ndim, const len_type len, const void* data, const stride_type stride)
{
    if (!data)
    {
        abort();
    }

    return check_tensor(ndim, len, stride);
}

template <typename T>
int check_tensor_indices(const Tensor<T>& A, const std::string& idx_A)
{
    std::vector<std::pair<char,dim_t> > idx_len;
    idx_len.reserve(A.getDimension());

    if (idx_A.size() != A.getDimension())
    {
        abort();
    }

    for (gint_t i = 0;i < A.getDimension();i++)
    {
        idx_len.push_back(std::make_pair(idx_A[i], A.getLength(i)));
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
    idx_len.reserve(A.getDimension()+
                    B.getDimension());

    if (idx_A.size() != A.getDimension() ||
        idx_B.size() != B.getDimension())
    {
        abort();
    }

    for (gint_t i = 0;i < A.getDimension();i++)
    {
        idx_len.push_back(std::make_pair(idx_A[i], A.getLength(i)));
    }

    for (gint_t i = 0;i < B.getDimension();i++)
    {
        idx_len.push_back(std::make_pair(idx_B[i], B.getLength(i)));
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
    idx_len.reserve(A.getDimension()+
                    B.getDimension()+
                    C.getDimension());

    if (idx_A.size() != A.getDimension() ||
        idx_B.size() != B.getDimension() ||
        idx_C.size() != C.getDimension())
    {
        abort();
    }

    for (gint_t i = 0;i < A.getDimension();i++)
    {
        idx_len.push_back(std::make_pair(idx_A[i], A.getLength(i)));
    }

    for (gint_t i = 0;i < B.getDimension();i++)
    {
        idx_len.push_back(std::make_pair(idx_B[i], B.getLength(i)));
    }

    for (gint_t i = 0;i < C.getDimension();i++)
    {
        idx_len.push_back(std::make_pair(idx_C[i], C.getLength(i)));
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
}

#endif
