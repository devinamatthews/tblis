#ifndef _TBLIS_INTERNAL_1T_DPD_UTIL_HPP_
#define _TBLIS_INTERNAL_1T_DPD_UTIL_HPP_

#include "util/basic_types.h"
#include "util/tensor.hpp"
#include "internal/1t/dense/add.hpp"
#include "internal/3t/dpd/mult.hpp"

namespace tblis
{
namespace internal
{

class irrep_iterator
{
    protected:
        const unsigned irrep_;
        const unsigned irrep_bits_;
        const unsigned irrep_mask_;
        viterator<0> it_;

    public:
        irrep_iterator(unsigned irrep, unsigned nirrep, unsigned ndim)
        : irrep_(irrep), irrep_bits_(__builtin_popcount(nirrep-1)),
          irrep_mask_(nirrep-1), it_(irrep_vector(ndim ? ndim-1 : 0, nirrep)) {}

        bool next()
        {
            return it_.next();
        }

        unsigned nblock() const
        {
            return 1u << (irrep_bits_*it_.dimension());
        }

        void block(unsigned b)
        {
            irrep_vector irreps(it_.dimension());

            for (unsigned i = 0;i < it_.dimension();i++)
            {
                irreps[i] = b & irrep_mask_;
                b >>= irrep_bits_;
            }

            it_.position(irreps);
        }

        void reset()
        {
            it_.reset();
        }

        unsigned irrep(unsigned dim)
        {
            TBLIS_ASSERT(dim <= it_.dimension());

            if (dim == 0)
            {
                unsigned irr0 = irrep_;
                for (unsigned irr : it_.position()) irr0 ^= irr;
                return irr0;
            }

            return it_.position()[dim-1];
        }
};

template <typename T>
void block_to_full(const communicator& comm, const config& cfg,
                   const dpd_varray_view<T>& A, varray<T>& A2)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();

    len_vector len_A(ndim_A);
    matrix<len_type> off_A{{ndim_A, nirrep}};
    for (unsigned i = 0;i < ndim_A;i++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            off_A[i][irrep] = len_A[i];
            len_A[i] += A.length(i, irrep);
        }
    }

    if (comm.master()) A2.reset(len_A);
    comm.barrier();

    A.for_each_block(
    [&](const varray_view<T>& local_A, const irrep_vector& irreps_A)
    {
        auto data_A2 = A2.data();
        for (unsigned i = 0;i < ndim_A;i++)
            data_A2 += off_A[i][irreps_A[i]]*A2.stride(i);

        add(type_tag<T>::value, comm, cfg, {}, {}, local_A.lengths(),
            T(1), false, reinterpret_cast<char*>(local_A.data()), {}, local_A.strides(),
            T(0), false, reinterpret_cast<char*>(       data_A2), {},      A2.strides());
    });
}

template <typename T>
void full_to_block(const communicator& comm, const config& cfg,
                   varray<T>& A2, const dpd_varray_view<T>& A)
{
    unsigned nirrep = A.num_irreps();
    unsigned ndim_A = A.dimension();

    matrix<len_type> off_A{{ndim_A, nirrep}};
    for (unsigned i = 0;i < ndim_A;i++)
    {
        len_type off = 0;
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
        {
            off_A[i][irrep] = off;
            off += A.length(i, irrep);
        }
    }

    A.for_each_block(
    [&](const varray_view<T>& local_A, const irrep_vector& irreps_A)
    {
        auto data_A2 = A2.data();
        for (unsigned i = 0;i < ndim_A;i++)
            data_A2 += off_A[i][irreps_A[i]]*A2.stride(i);

        add(type_tag<T>::value, comm, cfg, {}, {}, local_A.lengths(),
            T(1), false, reinterpret_cast<char*>(       data_A2), {},      A2.strides(),
            T(0), false, reinterpret_cast<char*>(local_A.data()), {}, local_A.strides());
    });
}

template <unsigned I, size_t N>
void dense_total_lengths_and_strides_helper(std::array<len_vector,N>&,
                                            std::array<stride_vector,N>&) {}

template <unsigned I, size_t N, typename Array, typename... Args>
void dense_total_lengths_and_strides_helper(std::array<len_vector,N>& len,
                                            std::array<stride_vector,N>& stride,
                                            const Array& A,
                                            const dim_vector&, const Args&... args)
{
    unsigned ndim = A.permutation().size();
    unsigned nirrep = A.num_irreps();

    len[I].resize(ndim);
    stride[I].resize(ndim);

    for (unsigned j = 0;j < ndim;j++)
    {
        for (unsigned irrep = 0;irrep < nirrep;irrep++)
            len[I][j] += A.length(j, irrep);
    }

    auto iperm = detail::inverse_permutation(A.permutation());
    stride[I][iperm[0]] = 1;
    for (unsigned j = 1;j < ndim;j++)
    {
        stride[I][iperm[j]] = stride[I][iperm[j-1]] * len[I][iperm[j-1]];
    }

    dense_total_lengths_and_strides_helper<I+1>(len, stride, args...);
}

template <size_t N, typename... Args>
void dense_total_lengths_and_strides(std::array<len_vector,N>& len,
                                     std::array<stride_vector,N>& stride,
                                     const Args&... args)
{
    dense_total_lengths_and_strides_helper<0>(len, stride, args...);
}

template <typename T>
bool is_block_empty(const dpd_varray_view<T>& A, const irrep_vector& irreps)
{
    unsigned irrep = 0;

    for (unsigned i = 0;i < A.dimension();i++)
    {
        irrep ^= irreps[i];
        if (!A.length(i, irreps[i])) return true;
    }

    return irrep != A.irrep();
}

inline unsigned assign_irrep(unsigned, unsigned irrep)
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
