#include "add.hpp"
#include "reduce.hpp"
#include "scale.hpp"
#include "shift.hpp"

#include "internal/1m/add.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void add(const communicator& comm, const config& cfg,
         const len_vector& len_A_,
         const len_vector& len_B_,
         const len_vector& len_AB_,
         T alpha, bool conj_A, const T* A,
         const stride_vector& stride_A_,
         const stride_vector& stride_A_AB_,
         T  beta, bool conj_B,       T* B,
         const stride_vector& stride_B_,
         const stride_vector& stride_B_AB_)
{
    auto perm_A = detail::sort_by_stride(stride_A_);
    auto perm_B = detail::sort_by_stride(stride_B_);
    auto perm_AB = detail::sort_by_stride(stride_B_AB_, stride_A_AB_);

    auto len_A = stl_ext::permuted(len_A_, perm_A);
    auto len_B = stl_ext::permuted(len_B_, perm_B);
    auto len_AB = stl_ext::permuted(len_AB_, perm_AB);

    auto stride_A = stl_ext::permuted(stride_A_, perm_A);
    auto stride_B = stl_ext::permuted(stride_B_, perm_B);
    auto stride_A_AB = stl_ext::permuted(stride_A_AB_, perm_AB);
    auto stride_B_AB = stl_ext::permuted(stride_B_AB_, perm_AB);

    len_type n_AB = stl_ext::prod(len_AB);
    len_type n_A = stl_ext::prod(len_A);
    len_type n_B = stl_ext::prod(len_B);

    if (n_AB == 0 || n_B == 0) return;

    if (n_A == 0)
    {
        scale(comm, cfg, len_B, beta, conj_B, B, stride_B);
        return;
    }

    //
    // Scalar intermediate
    //
    if (n_AB == 1)
    {
        if (n_A > 1)
        {
            T sum;
            len_type idx;
            reduce(comm, cfg, REDUCE_SUM, len_A, A, stride_A, sum, idx);

            if (comm.master())
            {
                if (beta == T(0))
                {
                    *B = alpha*(conj_A ? conj(sum) : sum);
                }
                else
                {
                    *B = alpha*(conj_A ? conj(sum) : sum) +
                          beta*(conj_B ? conj( *B) :  *B);
                }
            }
        }
        else if (n_B > 1)
        {
            shift(comm, cfg, len_B, alpha*(conj_A ? conj(*A) : *A),
                  beta, conj_B, B, stride_B);
        }
        else if (comm.master())
        {
            if (beta == T(0))
            {
                *B = alpha*(conj_A ? conj(*A) : *A);
            }
            else
            {
                *B = alpha*(conj_A ? conj(*A) : *A) +
                      beta*(conj_B ? conj(*B) : *B);
            }
        }

        comm.barrier();
        return;
    }

    if (n_A > 1)
    {
        //TODO sum (reduce?) ukr
        //TODO fused ukr

        comm.distribute_over_threads(n_AB,
        [&](len_type n_min, len_type n_max)
        {
            auto A1 = A;
            auto B1 = B;

            viterator<1> iter_A(len_A, stride_A);
            viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
            iter_AB.position(n_min, A1, B1);

            for (len_type i = n_min;i < n_max;i++)
            {
                iter_AB.next(A1, B1);

                T sum_A = T();
                while (iter_A.next(A1)) sum_A += *A1;
                sum_A = alpha*(conj_A ? conj(sum_A) : sum_A);

                if (beta == T(0)) *B1 = sum_A;
                else              *B1 = sum_A + beta*(conj_B ? conj(*B1) : *B1);
            }
        });
    }
    else if (n_B > 1)
    {
        //TODO replicate ukr
        //TODO fused ukr

        comm.distribute_over_threads(n_AB,
        [&](len_type n_min, len_type n_max)
        {
            auto A1 = A;
            auto B1 = B;

            viterator<1> iter_B(len_B, stride_B);
            viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
            iter_AB.position(n_min, A1, B1);

            for (len_type i = n_min;i < n_max;i++)
            {
                iter_AB.next(A1, B1);

                T tmp_A = alpha*(conj_A ? conj(*A1) : *A1);

                if (beta == T(0))
                {
                    while (iter_B.next(B1)) *B1 = tmp_A;
                }
                else
                {
                    TBLIS_SPECIAL_CASE(conj_B,
                    while (iter_B.next(B1))
                        *B1 = tmp_A + beta*(conj_B ? conj(*B1) : *B1);
                    )
                }
            }
        });
    }
    else
    {
        unsigned unit_A_AB = 0;
        unsigned unit_B_AB = 0;

        for (unsigned i = 1;i < len_AB.size();i++)
        {
            if (len_AB[i] == 1) continue;
            if (stride_A_AB[i] == 1 && unit_A_AB == 0) unit_A_AB = i;
            if (stride_B_AB[i] == 1 && unit_B_AB == 0) unit_B_AB = i;
        }

        if (unit_A_AB == unit_B_AB)
        {
            len_type n0 = len_AB[unit_A_AB];
            len_vector len1 = len_AB;
            len1.erase(len1.begin()+unit_A_AB);
            len_type n1 = stl_ext::prod(len1);

            stride_type stride_A0 = stride_A_AB[unit_A_AB];
            stride_vector stride_A1 = stride_A_AB;
            stride_A1.erase(stride_A1.begin()+unit_A_AB);

            stride_type stride_B0 = stride_B_AB[unit_A_AB];
            stride_vector stride_B1 = stride_B_AB;
            stride_B1.erase(stride_B1.begin()+unit_A_AB);

            comm.distribute_over_threads(n0, n1,
            [&](len_type n0_min, len_type n0_max, len_type n1_min, len_type n1_max)
            {
                auto A1 = A;
                auto B1 = B;

                viterator<2> iter_AB(len1, stride_A1, stride_B1);
                iter_AB.position(n1_min, A1, B1);

                A1 += n0_min*stride_A0;
                B1 += n0_min*stride_B0;

                for (len_type i = n1_min;i < n1_max;i++)
                {
                    iter_AB.next(A1, B1);
                    cfg.add_ukr.call<T>(n0_max-n0_min,
                                        alpha, conj_A, A1, stride_A0,
                                         beta, conj_B, B1, stride_B0);
                }
            });
        }
        else
        {
            // So that the two erase()'s work correctly
            if (unit_A_AB < unit_B_AB)
                std::swap(unit_A_AB, unit_B_AB);

            len_type m0 = len_AB[unit_A_AB];
            len_type n0 = len_AB[unit_B_AB];
            len_vector len1 = len_AB;
            len1.erase(len1.begin()+unit_A_AB);
            len1.erase(len1.begin()+unit_B_AB);
            len_type mn1 = stl_ext::prod(len1);

            stride_type stride_A_m = stride_A_AB[unit_A_AB];
            stride_type stride_A_n = stride_A_AB[unit_B_AB];
            stride_vector stride_A1 = stride_A_AB;
            stride_A1.erase(stride_A1.begin()+unit_A_AB);
            stride_A1.erase(stride_A1.begin()+unit_B_AB);

            stride_type stride_B_m = stride_B_AB[unit_A_AB];
            stride_type stride_B_n = stride_B_AB[unit_B_AB];
            stride_vector stride_B1 = stride_B_AB;
            stride_B1.erase(stride_B1.begin()+unit_A_AB);
            stride_B1.erase(stride_B1.begin()+unit_B_AB);

            unsigned nt_mn1, nt_mn;
            std::tie(nt_mn1, nt_mn) = partition_2x2(comm.num_threads(), mn1, m0*n0);

            auto subcomm = comm.gang(TCI_EVENLY, nt_mn1);

            subcomm.distribute_over_gangs(mn1,
            [&](len_type mn1_min, len_type mn1_max)
            {
                auto A1 = A;
                auto B1 = B;

                viterator<2> iter_AB(len1, stride_A1, stride_B1);
                iter_AB.position(mn1_min, A1, B1);

                for (len_type i = mn1_min;i < mn1_max;i++)
                {
                    iter_AB.next(A1, B1);

                    add(subcomm, cfg, m0, n0,
                        alpha, conj_A, A1, stride_A_m, stride_A_n,
                         beta, conj_B, B1, stride_B_m, stride_B_n);
                }
            });
        }
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void add(const communicator& comm, const config& cfg, \
                  const len_vector& len_A, \
                  const len_vector& len_B, \
                  const len_vector& len_AB, \
                  T alpha, bool conj_A, const T* A, \
                  const stride_vector& stride_A, \
                  const stride_vector& stride_A_AB, \
                  T  beta, bool conj_B,       T* B, \
                  const stride_vector& stride_B, \
                  const stride_vector& stride_B_AB);
#include "configs/foreach_type.h"

}
}
