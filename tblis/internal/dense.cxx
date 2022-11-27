#include <tblis/internal/dense.hpp>

#include <algorithm>

#include <stl_ext/iostream.hpp>

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
#if 0
    int k = 0;
    for (int i = 0;i < ndim;i++)
    {
        if (len[i] != 1)
        {
            len[k] = len[i];
            stride[k] = stride[i];
            idx[k] = idx[i];
            k++;
        }
    }
    ndim = k;
#endif

    /*
     * Accumulate strides for repeated indices
     */
    for (int i = 0;i < ndim;i++)
    {
        int k = i+1;
        for (int j = k;j < ndim;j++)
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
        ndim = k;
    }

    len.resize(ndim);
    stride.resize(ndim);
    idx.resize(ndim);
}

inline bool are_ordered()
{
    return true;
}

template <typename... Strides>
bool are_ordered(std::pair<stride_type,stride_type> s1, Strides... s2)
{
    return s1.first < s1.second || (s1.first == s1.second && are_ordered(s2...));
}

template <typename... Strides, size_t... I>
void fold(len_vector& len,
          label_vector& idx,
          std::tuple<Strides...> stride,
          std::index_sequence<I...>)
{
#if 0
    int ndim = len.size();

    /*
     * Sort indices by increasing stride
     */
    for (int i = 1; i < ndim; i++)
    for (int j = i; j > 0 && are_ordered(std::make_pair(std::get<I>(stride)[j-1], std::get<I>(stride)[j])...);j--)
    {
        std::swap(len[j-1], len[j]);
        std::swap(idx[j-1], idx[j]);
        (..., std::swap(std::get<I>(stride)[j-1], std::get<I>(stride)[j]));
    }

    /*
     * Find contiguous runs and replace with a single index
     */
    auto j = 0;
    for (int i = 0;i < ndim;j++)
    {
        len[j] = len[i];
        idx[j] = idx[i];
        (..., (std::get<I>(stride)[j] = std::get<I>(stride)[i]));

        while (((++i < ndim) && ... && (std::get<I>(stride)[i] == std::get<I>(stride)[i-1]*len[i-1])))
            len[j] *= len[i];
    }
    ndim = j;

    len.resize(ndim);
    idx.resize(ndim);
    (..., std::get<I>(stride).resize(ndim));
#endif
}

void fold(len_vector& len,
          stride_vector& stride,
          label_vector& idx)
{
    fold(len, idx, std::forward_as_tuple(stride), std::make_index_sequence<1>{});
}

void fold(len_vector& len,
          stride_vector& stride1,
          stride_vector& stride2,
          label_vector& idx)
{
    fold(len, idx, std::forward_as_tuple(stride1, stride2), std::make_index_sequence<2>{});
}

void fold(len_vector& len,
          stride_vector& stride1,
          stride_vector& stride2,
          stride_vector& stride3,
          label_vector& idx)
{
    fold(len, idx, std::forward_as_tuple(stride1, stride2, stride3), std::make_index_sequence<3>{});
}

}
}
