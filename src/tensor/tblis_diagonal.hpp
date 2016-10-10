#ifndef _TBLIS_PARTITION_HPP_
#define _TBLIS_PARTITION_HPP_

#include <string>
#include <vector>

#include "tblis_assert.hpp"
#include "tblis_basic_types.hpp"
#include "tblis_marray.hpp"

namespace tblis
{

inline void diagonal(std::vector<len_type>& len, std::string& idx,
                     std::vector<stride_type>& stride)
{
    unsigned ndim = len.size();
    std::vector<unsigned> inds = MArray::range(ndim);
    stl_ext::sort(inds, detail::sort_by_idx(idx));

    unsigned ndim_old = ndim;
    std::string idx_old = idx;
    std::vector<len_type> len_old = len;
    std::vector<stride_type> stride_old = stride;

    ndim = 0;
    for (unsigned i = 0;i < ndim_old;i++)
    {
        if (i == 0 || idx_old[inds[i]] != idx_old[inds[i-1]])
        {
            if (len_old[inds[i]] != 1)
            {
                idx[ndim] = idx_old[inds[i]];
                len[ndim] = len_old[inds[i]];
                stride[ndim] = stride_old[inds[i]];
                ndim++;
            }
        }
        else if (len_old[inds[i]] != 1)
        {
            TBLIS_ASSERT(len[ndim-1] == len_old[inds[i]]);
            if (len_old[inds[i]] != 1)
                stride[ndim-1] += stride_old[inds[i]];
        }
    }

    len.resize(ndim);
    stride.resize(ndim);
    idx.resize(ndim);
}

template <typename T>
void diagonal(const_tensor_view<T>& AD, std::string& idx_AD)
{
    TBLIS_ASSERT(AD.dimension() == idx_AD.size());

    std::vector<len_type> len_AD = AD.lengths();
    std::vector<stride_type> stride_AD = AD.strides();

    diagonal(len_AD, idx_AD, stride_AD);

    AD.reset(len_AD, AD.data(), stride_AD);
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

}

#endif
