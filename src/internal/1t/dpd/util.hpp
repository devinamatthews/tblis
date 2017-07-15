#ifndef _TBLIS_INTERNAL_1T_DPD_UTIL_HPP_
#define _TBLIS_INTERNAL_1T_DPD_UTIL_HPP_

#include "util/basic_types.h"
#include "internal/3t/dpd/mult.hpp"

namespace tblis
{
namespace internal
{

template <typename T, typename U>
void block_to_full(const dpd_varray_view<T>& A, varray<U>& A2)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();

    len_vector len_A(ndim_A);
    matrix<len_type> off_A({ndim_A, nirrep});
    for (unsigned i = 0;i < ndim_A;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            off_A[i][irrep] = len_A[i];
            len_A[i] += A.length(i, irrep);
        }
    }
    A2.reset(len_A);

    A.for_each_block(
    [&](const varray_view<T>& local_A, const irrep_vector& irreps_A)
    {
        varray_view<U> local_A2 = A2;

        for (unsigned i = 0;i < ndim_A;i++)
        {
            local_A2.length(i, local_A.length(i));
            local_A2.shift(i, off_A[i][irreps_A[i]]);
        }

        local_A2 = local_A;
    });
}

template <typename T, typename U>
void full_to_block(const varray<U>& A2, const dpd_varray_view<T>& A)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();

    len_vector len_A(ndim_A);
    matrix<len_type> off_A({ndim_A, nirrep});
    for (unsigned i = 0;i < ndim_A;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            off_A[i][irrep] = len_A[i];
            len_A[i] += A.length(i, irrep);
        }
    }

    A.for_each_block(
    [&](const varray_view<T>& local_A, const irrep_vector& irreps_A)
    {
        varray_view<const U> local_A2 = A2;

        for (unsigned i = 0;i < ndim_A;i++)
        {
            local_A2.length(i, local_A.length(i));
            local_A2.shift(i, off_A[i][irreps_A[i]]);
        }

        local_A = local_A2;
    });
}

template <typename T>
bool is_block_empty(const dpd_varray_view<T>& A, const irrep_vector& irreps)
{
    for (unsigned i = 0;i < A.dimension();i++)
    {
        if (!A.length(i, irreps[i])) return true;
    }
    return false;
}

inline unsigned assign_irrep(unsigned dim, unsigned irrep)
{
    return irrep;
}

template <typename... Args>
unsigned assign_irrep(unsigned dim, unsigned irrep,
                      irrep_vector& irreps,
                      const dim_vector& idx,
                      Args&... args)
{
    irreps[idx[dim]] = irrep;
    return assign_irrep(dim, irrep, args...);
}

template <typename... Args>
void assign_irreps(unsigned ndim, unsigned irrep, unsigned nirrep,
                   stride_type block, Args&... args)
{
    unsigned mask = nirrep-1;
    unsigned shift = (nirrep>1) + (nirrep>2) + (nirrep>4);

    unsigned irrep0 = irrep;
    for (unsigned i = 1;i < ndim;i++)
    {
        irrep0 ^= assign_irrep(i, block & mask, args...);
        block >>= shift;
    }
    if (ndim) assign_irrep(0, irrep0, args...);
}

}
}

#endif
