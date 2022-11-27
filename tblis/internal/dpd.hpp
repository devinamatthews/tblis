#ifndef TBLIS_INTERNAL_DPD_HPP
#define TBLIS_INTERNAL_DPD_HPP

#include <tblis/internal/dense.hpp>

#include <marray/dpd/dpd_marray_view.hpp>

namespace tblis
{

using MArray::matrix;
using MArray::irrep_iterator;
using MArray::dpd_marray_view;

using irrep_vector = dim_vector;

namespace internal
{

enum dpd_impl_t {BLIS, BLOCKED, FULL};
extern dpd_impl_t dpd_impl;

void add(type_t type, const communicator& comm, const config& cfg,
         const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
         const dim_vector& idx_A,
         const dim_vector& idx_A_AB,
         const scalar&  beta, bool conj_B, const dpd_marray_view<char>& B,
         const dim_vector& idx_B,
         const dim_vector& idx_B_AB);

void dot(type_t type, const communicator& comm, const config& cfg,
         bool conj_A, const dpd_marray_view<char>& A,
         const dim_vector& idx_A_AB,
         bool conj_B, const dpd_marray_view<char>& B,
         const dim_vector& idx_B_AB,
         char* result);

void reduce(type_t type, const communicator& comm, const config& cfg, reduce_t op,
            const dpd_marray_view<char>& A, const dim_vector& idx_A_A,
            char* result, len_type& idx);

void scale(type_t type, const communicator& comm, const config& cfg,
           const scalar& alpha, bool conj_A, const dpd_marray_view<char>& A,
           const dim_vector& idx_A_A);

void set(type_t type, const communicator& comm, const config& cfg,
         const scalar& alpha, const dpd_marray_view<char>& A, const dim_vector& idx_A);

void shift(type_t type, const communicator& comm, const config& cfg,
           const scalar& alpha, const scalar& beta, bool conj_A,
           const dpd_marray_view<char>& A, const dim_vector& idx_A_A);

void mult(type_t type, const communicator& comm, const config& cfg,
          const scalar& alpha,
          bool conj_A, const dpd_marray_view<char>& A,
          const dim_vector& idx_A_AB,
          const dim_vector& idx_A_AC,
          const dim_vector& idx_A_ABC,
          bool conj_B, const dpd_marray_view<char>& B,
          const dim_vector& idx_B_AB,
          const dim_vector& idx_B_BC,
          const dim_vector& idx_B_ABC,
          const scalar&  beta,
          bool conj_C, const dpd_marray_view<char>& C,
          const dim_vector& idx_C_AC,
          const dim_vector& idx_C_BC,
          const dim_vector& idx_C_ABC);

void canonicalize(dpd_marray_view<char>& A, label_vector& idx);

void fold(dpd_marray_view<char>& A, dim_vector& idx);

void fold(dpd_marray_view<char>& A, dim_vector& idx1,
          dpd_marray_view<char>& B, dim_vector& idx2);

void fold(dpd_marray_view<char>& A, dim_vector& idx1,
          dpd_marray_view<char>& B, dim_vector& idx2,
          dpd_marray_view<char>& C, dim_vector& idx3);

template <typename T>
void block_to_full(const communicator& comm, const config& cfg,
                   const dpd_marray_view<T>& A, marray<T>& A2)
{
    auto nirrep = A.num_irreps();
    auto ndim_A = A.dimension();

    len_vector len_A(ndim_A);
    matrix<len_type> off_A{ndim_A, nirrep};
    for (auto i : range(ndim_A))
    {
        for (auto irrep : range(nirrep))
        {
            off_A[i][irrep] = len_A[i];
            len_A[i] += A.length(i, irrep);
        }
    }

    if (comm.master()) A2.reset(len_A);
    comm.barrier();

    A.for_each_block(
    [&](const marray_view<T>& local_A, const irrep_vector& irreps_A)
    {
        auto data_A2 = A2.data();
        for (auto i : range(ndim_A))
            data_A2 += off_A[i][irreps_A[i]]*A2.stride(i);

        add(type_tag<T>::value, comm, cfg, {}, {}, local_A.lengths(),
            T(1), false, reinterpret_cast<char*>(local_A.data()), {}, local_A.strides(),
            T(0), false, reinterpret_cast<char*>(       data_A2), {},      A2.strides());
    });
}

template <typename T>
void full_to_block(const communicator& comm, const config& cfg,
                   marray<T>& A2, const dpd_marray_view<T>& A)
{
    auto nirrep = A.num_irreps();
    auto ndim_A = A.dimension();

    matrix<len_type> off_A{ndim_A, nirrep};
    for (auto i : range(ndim_A))
    {
        len_type off = 0;
        for (auto irrep : range(nirrep))
        {
            off_A[i][irrep] = off;
            off += A.length(i, irrep);
        }
    }

    A.for_each_block(
    [&](const marray_view<T>& local_A, const irrep_vector& irreps_A)
    {
        auto data_A2 = A2.data();
        for (auto i : range(ndim_A))
            data_A2 += off_A[i][irreps_A[i]]*A2.stride(i);

        add(type_tag<T>::value, comm, cfg, {}, {}, local_A.lengths(),
            T(1), false, reinterpret_cast<char*>(       data_A2), {},      A2.strides(),
            T(0), false, reinterpret_cast<char*>(local_A.data()), {}, local_A.strides());
    });
}

template <int I, size_t N>
void dense_total_lengths_and_strides_helper(std::array<len_vector,N>&,
                                            std::array<stride_vector,N>&) {}

template <int I, size_t N, typename Array, typename... Args>
void dense_total_lengths_and_strides_helper(std::array<len_vector,N>& len,
                                            std::array<stride_vector,N>& stride,
                                            const Array& A,
                                            const dim_vector&, const Args&... args)
{
    int ndim = A.permutation().size();
    auto nirrep = A.num_irreps();

    len[I].resize(ndim);
    stride[I].resize(ndim);

    for (auto j : range(ndim))
    {
        for (auto irrep : range(nirrep))
            len[I][j] += A.length(j, irrep);
    }

    auto iperm = inverse_permutation(A.permutation());
    stride[I][iperm[0]] = 1;
    for (auto j : range(1,ndim))
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
bool is_block_empty(const dpd_marray_view<T>& A, const irrep_vector& irreps)
{
    auto irrep = 0;

    for (auto i : range(A.dimension()))
    {
        irrep ^= irreps[i];
        if (!A.length(i, irreps[i])) return true;
    }

    return irrep != A.irrep();
}

inline int assign_irrep(int, int irrep)
{
    return irrep;
}

template <typename... Args>
int assign_irrep(int dim, int irrep,
                 irrep_vector& irreps,
                 const dim_vector& idx,
                 Args&... args)
{
    irreps[idx[dim]] = irrep;
    return assign_irrep(dim, irrep, args...);
}

template <typename... Args>
void assign_irreps(int ndim, int irrep, int nirrep,
                   stride_type block, Args&... args)
{
    int mask = nirrep-1;
    int shift = (nirrep>1) + (nirrep>2) + (nirrep>4);

    int irrep0 = irrep;
    for (auto i : range(1,ndim))
    {
        irrep0 ^= assign_irrep(i, block & mask, args...);
        block >>= shift;
    }
    if (ndim) assign_irrep(0, irrep0, args...);
}

}
}

#endif //TBLIS_INTERNAL_DPD_HPP
