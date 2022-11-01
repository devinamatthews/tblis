#include <tblis/internal/dpd.hpp>

#include <algorithm>

namespace tblis
{
namespace internal
{

std::atomic<long> flops;
dpd_impl_t dpd_impl = BLIS;

void canonicalize(dpd_varray_view<char>& A, label_vector& idx)
{
    (void)A;
    (void)idx;

    /*
     * Remove singleton dimensions (but leave one if all are singleton)
     */

    //TODO

    /*
     * Accumulate strides for repeated indices
     */

    //TODO
}

void fold(dpd_varray_view<char>& A, dim_vector& idx)
{
    int ndim = idx.size();

    /*
     * Sort indices by increasing stride
     */

    std::array<len_vector,1> len;
    std::array<stride_vector,1> stride;
    dense_total_lengths_and_strides(len, stride, A, idx);

    auto needs_swap = [&](int i)
    {
        return stride[i-1] > stride[i];
    };

    for (int i = 0;i < ndim;i++)
    for (int j = i;j > 0 && needs_swap(j);j--)
    {
        std::swap(len[j-1], len[j]);
        std::swap(stride[j-1], stride[j]);
        std::swap(idx[j-1], idx[j]);
    }

    /*
     * Find contiguous runs and replace with a single index
     */

    //TODO
}

void fold(dpd_varray_view<char>& A, dim_vector& idx1,
          dpd_varray_view<char>& B, dim_vector& idx2)
{
    int ndim = idx1.size();

    /*
     * Sort indices by increasing stride
     */

    std::array<len_vector,1> len;
    std::array<stride_vector,1> stride1;
    std::array<stride_vector,1> stride2;
    dense_total_lengths_and_strides(len, stride1, A, idx1);
    dense_total_lengths_and_strides(len, stride2, B, idx2);

    auto needs_swap = [&](int i)
    {
        return stride1[i-1] > stride1[i] ||
            (stride1[i-1] == stride1[i] && stride2[i-1] > stride2[i]);
    };

    for (int i = 0;i < ndim;i++)
    for (int j = i;j > 0 && needs_swap(j);j--)
    {
        std::swap(len[j-1], len[j]);
        std::swap(stride1[j-1], stride1[j]);
        std::swap(stride2[j-1], stride2[j]);
        std::swap(idx1[j-1], idx1[j]);
        std::swap(idx2[j-1], idx2[j]);
    }

    /*
     * Find contiguous runs and replace with a single index
     */

    //TODO
}

void fold(dpd_varray_view<char>& A, dim_vector& idx1,
          dpd_varray_view<char>& B, dim_vector& idx2,
          dpd_varray_view<char>& C, dim_vector& idx3)
{
    int ndim = idx1.size();

    /*
     * Sort indices by increasing stride
     */

    std::array<len_vector,1> len;
    std::array<stride_vector,1> stride1;
    std::array<stride_vector,1> stride2;
    std::array<stride_vector,1> stride3;
    dense_total_lengths_and_strides(len, stride1, A, idx1);
    dense_total_lengths_and_strides(len, stride2, B, idx2);
    dense_total_lengths_and_strides(len, stride3, C, idx3);

    auto needs_swap = [&](int i)
    {
        return stride1[i-1] > stride1[i] ||
            (stride1[i-1] == stride1[i] && stride2[i-1] > stride2[i]) ||
            (stride1[i-1] == stride1[i] && stride2[i-1] == stride2[i] && stride3[i-1] > stride3[i]);
    };

    for (int i = 0;i < ndim;i++)
    for (int j = i;j > 0 && needs_swap(j);j--)
    {
        std::swap(len[j-1], len[j]);
        std::swap(stride1[j-1], stride1[j]);
        std::swap(stride2[j-1], stride2[j]);
        std::swap(stride3[j-1], stride3[j]);
        std::swap(idx1[j-1], idx1[j]);
        std::swap(idx2[j-1], idx2[j]);
        std::swap(idx3[j-1], idx3[j]);
    }

    /*
     * Find contiguous runs and replace with a single index
     */

    //TODO
}

}
}
