#ifndef _TBLIS_PARTITION_HPP_
#define _TBLIS_PARTITION_HPP_

#include <string>
#include <vector>

#include "../util/assert.h"
#include "../util/basic_types.h"
#include "../util/marray.hpp"

namespace tblis
{

template <typename T>
void matricize(const_tensor_view<T>  A,
               const_matrix_view<T>& AM, unsigned split)
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
void matricize(tensor_view<T>  A,
               matrix_view<T>& AM, unsigned split)
{
    matricize<T>(A, reinterpret_cast<const_matrix_view<T>&>(AM), split);
}

}

#endif
