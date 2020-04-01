#include "add.hpp"
#include "reduce.hpp"
#include "scale.hpp"
#include "shift.hpp"

#include "internal/0/add.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

void add(type_t type, const communicator& comm, const config& cfg,
         const len_vector& len_A,
         const scalar& alpha, bool conj_A, char* A,
         const stride_vector& stride_A,
         const scalar&  beta, bool conj_B, char* B)
{
    scalar sum(0.0, type);
    len_type idx;
    reduce(type, comm, cfg, REDUCE_SUM, len_A, A, stride_A, sum.raw(), idx);

    if (comm.master())
        add(type, alpha, conj_A, sum.raw(), beta, conj_B, B);
}

void add(type_t type, const communicator& comm, const config& cfg,
         const len_vector& len_B,
         const scalar& alpha, bool conj_A, char* A,
         const scalar&  beta, bool conj_B, char* B,
         const stride_vector& stride_B)
{
    scalar alpha_A(0, type);
    alpha_A.from(A);
    alpha_A *= alpha;

    shift(type, comm, cfg, len_B, alpha_A,
          beta, conj_B, B, stride_B);
}

void add(type_t type, const communicator& comm, const config& cfg,
         const len_vector& len_A,
         const len_vector& len_AB,
         const scalar& alpha, bool conj_A, char* A,
         const stride_vector& stride_A,
         const stride_vector& stride_A_AB_,
         const scalar&  beta, bool conj_B, char* B,
         const stride_vector& stride_B_AB_)
{
    bool empty = len_A.size() == 0;

    const len_type ts = type_size[type];

    len_type dummy;
    len_type n0 = (empty ? 1 : len_A[0]);
    len_vector len_A1(len_A.begin() + !empty, len_A.end());
    len_type n_AB = stl_ext::prod(len_AB);

    stride_type stride_A0 = (empty ? 1 : stride_A[0]);
    len_vector stride_A1;
    for (unsigned i = 1;i < stride_A.size();i++) stride_A1.push_back(stride_A[i]*ts);

    len_vector stride_A_AB, stride_B_AB;
    for (auto i : stride_A_AB_) stride_A_AB.push_back(i*ts);
    for (auto i : stride_B_AB_) stride_B_AB.push_back(i*ts);

    comm.distribute_over_threads(n_AB,
    [&](len_type n_min, len_type n_max)
    {
        auto A1 = A;
        auto B1 = B;

        viterator<1> iter_A(len_A1, stride_A1);
        viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
        iter_AB.position(n_min, A1, B1);

        for (len_type i = n_min;i < n_max;i++)
        {
            iter_AB.next(A1, B1);

            scalar sum_A(0.0, type);
            while (iter_A.next(A1))
                cfg.reduce_ukr.call(type, REDUCE_SUM, n0, A1, stride_A0, sum_A.raw(), dummy);

            add(type, alpha, conj_A, sum_A.raw(), beta, conj_B, B1);
        }
    });
}

void add(type_t type, const communicator& comm, const config& cfg,
         const len_vector& len_B,
         const len_vector& len_AB,
         const scalar& alpha, bool conj_A, char* A,
         const stride_vector& stride_A_AB_,
         const scalar&  beta, bool conj_B, char* B,
         const stride_vector& stride_B,
         const stride_vector& stride_B_AB_)
{
    bool empty = len_B.size() == 0;

    const len_type ts = type_size[type];

    len_type n0 = (empty ? 1 : len_B[0]);
    len_vector len_B1(len_B.begin() + !empty, len_B.end());
    len_type n_AB = stl_ext::prod(len_AB);

    stride_type stride_B0 = (empty ? 1 : stride_B[0]);
    len_vector stride_B1;
    for (unsigned i = 1;i < stride_B.size();i++) stride_B1.push_back(stride_B[i]*ts);

    len_vector stride_A_AB, stride_B_AB;
    for (auto i : stride_A_AB_) stride_A_AB.push_back(i*ts);
    for (auto i : stride_B_AB_) stride_B_AB.push_back(i*ts);

    comm.distribute_over_threads(n_AB,
    [&](len_type n_min, len_type n_max)
    {
        auto A1 = A;
        auto B1 = B;

        viterator<1> iter_B(len_B1, stride_B1);
        viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
        iter_AB.position(n_min, A1, B1);

        for (len_type i = n_min;i < n_max;i++)
        {
            iter_AB.next(A1, B1);

            scalar alpha_A(0, type);
            alpha_A.from(A1);
            alpha_A *= alpha;

            while (iter_B.next(B1))
                cfg.shift_ukr.call(type, n0, &alpha_A, &beta, conj_B, B1, stride_B0);
        }
    });
}

void add(type_t type, const communicator& comm, const config& cfg,
         const len_vector& len_AB,
         const scalar& alpha, bool conj_A, char* A,
         const stride_vector& stride_A_AB,
         const scalar&  beta, bool conj_B, char* B,
         const stride_vector& stride_B_AB)
{
    const len_type ts = type_size[type];

    const len_type MR = cfg.trans_mr.def(type);
    const len_type NR = cfg.trans_nr.def(type);

    unsigned unit_A_AB = 0;
    unsigned unit_B_AB = 0;

    for (unsigned i = 1;i < len_AB.size();i++)
    {
        if (len_AB[i] == 1) continue;
        if (stride_A_AB[i] == 1 && unit_A_AB == 0) unit_A_AB = i;
        if (stride_B_AB[i] == 1 && unit_B_AB == 0) unit_B_AB = i;
    }

    len_type m0 = len_AB[unit_A_AB];
    len_type n0 = len_AB[unit_B_AB];
    len_vector len1;
    for (unsigned i = 0;i < len_AB.size();i++)
        if (i != unit_A_AB && i != unit_B_AB)
            len1.push_back(len_AB[i]);
    len_type mn1 = stl_ext::prod(len1);

    stride_type stride_A_m = stride_A_AB[unit_A_AB];
    stride_type stride_A_n = stride_A_AB[unit_B_AB];
    stride_vector stride_A1;
    for (unsigned i = 0;i < stride_A_AB.size();i++)
        if (i != unit_A_AB && i != unit_B_AB)
            stride_A1.push_back(stride_A_AB[i]*ts);

    stride_type stride_B_m = stride_B_AB[unit_A_AB];
    stride_type stride_B_n = stride_B_AB[unit_B_AB];
    stride_vector stride_B1;
    for (unsigned i = 0;i < stride_B_AB.size();i++)
        if (i != unit_A_AB && i != unit_B_AB)
            stride_B1.push_back(stride_B_AB[i]*ts);

    if (unit_A_AB == unit_B_AB)
    {
        comm.distribute_over_threads(n0, mn1,
        [&](len_type n0_min, len_type n0_max, len_type n1_min, len_type n1_max)
        {
            auto A1 = A;
            auto B1 = B;

            viterator<2> iter_AB(len1, stride_A1, stride_B1);
            iter_AB.position(n1_min, A1, B1);
            A1 += n0_min*stride_A_m*ts;
            B1 += n0_min*stride_B_m*ts;

            for (len_type i = n1_min;i < n1_max;i++)
            {
                iter_AB.next(A1, B1);

                cfg.add_ukr.call(type, n0_max-n0_min,
                                 &alpha, conj_A, A1, stride_A_m,
                                  &beta, conj_B, B1, stride_B_m);
            }
        });
    }
    else
    {
        unsigned nt_mn1, nt_mn;
        std::tie(nt_mn1, nt_mn) = partition_2x2(comm.num_threads(), mn1, m0*n0);

        auto subcomm = comm.gang(TCI_EVENLY, nt_mn1);

        auto m = m0;
        auto n = n0;
        auto rs_A = stride_A_m;
        auto cs_A = stride_A_n;
        auto rs_B = stride_B_m;
        auto cs_B = stride_B_n;

        if (rs_B > cs_B)
        {
            std::swap(m, n);
            std::swap(rs_A, cs_A);
            std::swap(rs_B, cs_B);
        }

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

                subcomm.distribute_over_threads({m, MR}, {n, NR},
                [&](len_type m_min, len_type m_max, len_type n_min, len_type n_max)
                {
                    for (len_type i = m_min;i < m_max;i += MR)
                    for (len_type j = n_min;j < n_max;j += NR)
                    {
                        len_type m_loc = std::min(m_max-i, MR);
                        len_type n_loc = std::min(n_max-j, NR);

                        cfg.trans_ukr.call(type, m_loc, n_loc,
                            &alpha, conj_A, A1 + i*rs_A*ts + j*cs_A*ts, rs_A, cs_A,
                             &beta, conj_B, B1 + i*rs_B*ts + j*cs_B*ts, rs_B, cs_B);
                    }
                });
            }
        });
    }
}

void add(type_t type, const communicator& comm, const config& cfg,
         const len_vector& len_A_,
         const len_vector& len_B_,
         const len_vector& len_AB_,
         const scalar& alpha, bool conj_A, char* A,
         const stride_vector& stride_A_,
         const stride_vector& stride_A_AB_,
         const scalar&  beta, bool conj_B, char* B,
         const stride_vector& stride_B_,
         const stride_vector& stride_B_AB_)
{
    len_type n_AB = stl_ext::prod(len_AB_);
    len_type n_A = stl_ext::prod(len_A_);
    len_type n_B = stl_ext::prod(len_B_);

    if (n_AB == 0 || n_B == 0) return;

    if (n_A == 0)
    {
        scale(type, comm, cfg, len_B_, beta, conj_B, B, stride_B_);
        return;
    }

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

    if (n_AB == 1)
    {
        if (n_A > 1)
        {
            add(type, comm, cfg, len_A,
                alpha, conj_A, A, stride_A,
                 beta, conj_B, B);
        }
        else if (n_B > 1)
        {
            add(type, comm, cfg, len_B,
                alpha, conj_A, A,
                 beta, conj_B, B, stride_B);
        }
        else if (comm.master())
        {
            add(type, alpha, conj_A, A,
                       beta, conj_B, B);
        }
    }
    else
    {
        if (n_A > 1)
        {
            add(type, comm, cfg, len_A, len_AB,
                alpha, conj_A, A, stride_A, stride_A_AB,
                 beta, conj_B, B, stride_B_AB);
        }
        else if (n_B > 1)
        {
            add(type, comm, cfg, len_B, len_AB,
                alpha, conj_A, A, stride_A_AB,
                 beta, conj_B, B, stride_B, stride_B_AB);
        }
        else
        {
            add(type, comm, cfg, len_AB,
                alpha, conj_A, A, stride_A_AB,
                 beta, conj_B, B, stride_B_AB);
        }
    }

    comm.barrier();
}

}
}
