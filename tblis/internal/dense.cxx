#include <tblis/internal/dense.hpp>

#include <algorithm>

namespace tblis
{
namespace internal
{

impl_t impl = BLIS_BASED;

void canonicalize(len_vector& len,
                  stride_vector& stride,
                  label_vector& idx)
{
    int ndim = len.size();

    /*
     * Remove singleton dimensions (but leave one if all are singleton)
     */

    int k = 0;
    for (int i = 0;i < ndim;i++)
    {
        if (len[i] != 0)
        {
            len[k] = len[i];
            stride[k] = stride[i];
            k++;
        }
    }
    ndim = k+1;

    /*
     * Accumulate strides for repeated indices
     */

    for (int i = 0;i < ndim;i++)
    {
        int k = i+1;
        for (int j = i+1;j < ndim;j++)
        {
            if (idx[i] == idx[j])
            {
                stride[i] += stride[j];
            }
            else
            {
                len[k] = len[j];
                stride[k] = stride[j];
                k++;
            }
        }
        ndim = k+1;
    }

    len.resize(ndim);
    stride.resize(ndim);
    idx.resize(ndim);
}

void fold(len_vector& len,
          stride_vector& stride,
          label_vector& idx)
{
    int ndim = len.size();

    /*
     * Sort indices by increasing stride
     */

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

    for (int i = 0;i < ndim;)
    {
        int j = i+1;
        while (j < ndim && stride[j] == stride[j-1]*len[j-1])
        {
            len[i] *= len[j];
            j++;
        }

        std::copy(   len.data()+j,    len.data()+ndim,    len.data()+i+1);
        std::copy(stride.data()+j, stride.data()+ndim, stride.data()+i+1);
        std::copy(   idx.data()+j,    idx.data()+ndim,    idx.data()+i+1);

        ndim -= j-i-1;
    }

    len.resize(ndim);
    stride.resize(ndim);
    idx.resize(ndim);
}

void fold(len_vector& len,
          stride_vector& stride1,
          stride_vector& stride2,
          label_vector& idx)
{
    int ndim = len.size();

    /*
     * Sort indices by increasing stride
     */

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
        std::swap(idx[j-1], idx[j]);
    }

    /*
     * Find contiguous runs and replace with a single index
     */

    for (int i = 0;i < ndim;)
    {
        int j = i+1;
        while (j < ndim && stride1[j] == stride1[j-1]*len[j-1]
                        && stride2[j] == stride2[j-1]*len[j-1])
        {
            len[i] *= len[j];
            j++;
        }

        std::copy(    len.data()+j,     len.data()+ndim,     len.data()+i+1);
        std::copy(stride1.data()+j, stride1.data()+ndim, stride1.data()+i+1);
        std::copy(stride2.data()+j, stride2.data()+ndim, stride2.data()+i+1);
        std::copy(    idx.data()+j,     idx.data()+ndim,     idx.data()+i+1);

        ndim -= j-i-1;
    }

    len.resize(ndim);
    stride1.resize(ndim);
    stride2.resize(ndim);
    idx.resize(ndim);
}

void fold(len_vector& len,
          stride_vector& stride1,
          stride_vector& stride2,
          stride_vector& stride3,
          label_vector& idx)
{
    int ndim = len.size();

    /*
     * Sort indices by increasing stride
     */

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
        std::swap(idx[j-1], idx[j]);
    }

    /*
     * Find contiguous runs and replace with a single index
     */

    for (int i = 0;i < ndim;)
    {
        int j = i+1;
        while (j < ndim && stride1[j] == stride1[j-1]*len[j-1]
                        && stride2[j] == stride2[j-1]*len[j-1]
                        && stride3[j] == stride3[j-1]*len[j-1])
        {
            len[i] *= len[j];
            j++;
        }

        std::copy(    len.data()+j,     len.data()+ndim,     len.data()+i+1);
        std::copy(stride1.data()+j, stride1.data()+ndim, stride1.data()+i+1);
        std::copy(stride2.data()+j, stride2.data()+ndim, stride2.data()+i+1);
        std::copy(stride3.data()+j, stride3.data()+ndim, stride3.data()+i+1);
        std::copy(    idx.data()+j,     idx.data()+ndim,     idx.data()+i+1);

        ndim -= j-i-1;
    }

    len.resize(ndim);
    stride1.resize(ndim);
    stride2.resize(ndim);
    stride3.resize(ndim);
    idx.resize(ndim);
}

}
}
