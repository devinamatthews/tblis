#include "mult.hpp"

#include "util/gemm_thread.hpp"
#include "util/tensor.hpp"

#include "nodes/gemm.hpp"

#include "internal/0/add.hpp"
#include "internal/0/mult.hpp"
#include "internal/1t/dense/add.hpp"
#include "internal/1t/dense/dot.hpp"
#include "internal/1t/dense/scale.hpp"
#include "internal/1t/dense/set.hpp"

#include "matrix/normal_matrix.hpp"
#include "matrix/tensor_matrix.hpp"

namespace tblis
{

MemoryPool BuffersForA, BuffersForB, BuffersForC;

namespace internal
{

impl_t impl = BLIS_BASED;

void mult_blis(type_t type, const communicator& comm, const config& cfg,
               const len_vector& len_AB,
               const len_vector& len_AC,
               const scalar& alpha, bool conj_A, char* A,
               const stride_vector& stride_A_AB_,
               const stride_vector& stride_A_AC_,
                                    bool conj_B, char* B,
               const stride_vector& stride_B_AB_,
               const scalar&  beta, bool conj_C, char* C,
               const stride_vector& stride_C_AC_)
{
    const len_type ts = type_size[type];

    auto stride_A_AB = stride_A_AB_;
    auto stride_A_AC = stride_A_AC_;
    auto stride_B_AB = stride_B_AB_;
    auto stride_C_AC = stride_C_AC_;

    auto reorder_AC = detail::sort_by_stride(stride_A_AC, stride_C_AC);
    auto reorder_AB = detail::sort_by_stride(stride_A_AB, stride_B_AB);

    auto m = len_AC[reorder_AC[0]];
    auto n = len_AB[reorder_AB[0]];
    auto rs_A = stride_A_AC[reorder_AC[0]];
    auto cs_A = stride_A_AB[reorder_AB[0]];
    auto inc_B = stride_B_AB[reorder_AB[0]];
    auto inc_C = stride_C_AC[reorder_AC[0]];

    reorder_AC.erase(reorder_AC.begin());
    reorder_AB.erase(reorder_AB.begin());

    for (auto& s : stride_A_AC) s *= ts;
    for (auto& s : stride_A_AB) s *= ts;
    for (auto& s : stride_B_AB) s *= ts;
    for (auto& s : stride_C_AC) s *= ts;

    auto m2 = stl_ext::prod(len_AC)/m;
    auto n2 = stl_ext::prod(len_AB)/n;

    if (comm.master()) flops += 2*m*m2*n*n2;

    unsigned nt_l, nt_m;
    std::tie(nt_l, nt_m) = partition_2x2(comm.num_threads(), m2, m2, m, m);

    auto subcomm = comm.gang(TCI_EVENLY, nt_l);

    subcomm.distribute_over_gangs(m2,
    [&](len_type l_min, len_type l_max)
    {
        viterator<2> iter_AB(stl_ext::permuted(len_AB, reorder_AB),
                             stl_ext::permuted(stride_A_AB, reorder_AB),
                             stl_ext::permuted(stride_B_AB, reorder_AB));

        viterator<2> iter_AC(stl_ext::permuted(len_AC, reorder_AC),
                             stl_ext::permuted(stride_A_AC, reorder_AC),
                             stl_ext::permuted(stride_C_AC, reorder_AC));

        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        iter_AC.position(l_min, A1, C1);

        for (len_type l = l_min;l < l_max;l++)
        {
            iter_AC.next(A1, C1);

            auto local_beta = beta;
            auto local_conj_C = conj_C;

            while (iter_AB.next(A1, B1))
            {
                if (rs_A <= cs_A)
                {
                    const len_type NF = cfg.addf_nf.def(type);

                    subcomm.distribute_over_threads(m,
                    [&,local_beta,local_conj_C](len_type m_min, len_type m_max) mutable
                    {
                        const void* As[16];

                        for (len_type j = 0;j < n;j += NF)
                        {
                            for (len_type k = 0;k < NF;k++)
                                As[k] = A1 + m_min*rs_A*ts + (j+k)*cs_A*ts;

                            cfg.addf_sum_ukr.call(type, m_max-m_min, std::min(NF, n-j),
                                                       &alpha,       conj_A, As, rs_A,
                                                                     conj_B, B1 + j*inc_B*ts, inc_B,
                                                  &local_beta, local_conj_C, C1 + m_min*inc_C*ts, inc_C);

                            local_beta = 1.0;
                            local_conj_C = false;
                        }
                    });
                }
                else
                {
                    const len_type NF = cfg.dotf_nf.def(type);

                    subcomm.distribute_over_threads({m, NF},
                    [&](len_type m_min, len_type m_max)
                    {
                        for (len_type i = m_min;i < m_max;i += NF)
                        {
                            cfg.dotf_ukr.call(type, std::min(NF, m_max-i), n,
                                                   &alpha,       conj_A, A1 + i*rs_A*ts, rs_A, cs_A,
                                                                 conj_B, B1, inc_B,
                                              &local_beta, local_conj_C, C1 + i*inc_C*ts, inc_C);
                        }
                    });
                }

                subcomm.barrier();

                local_beta = 1.0;
                local_conj_C = false;
            }
        }
    });
}

void mult_blis(type_t type, const communicator& comm, const config& cfg,
               const len_vector& len_AC,
               const len_vector& len_BC,
               const scalar& alpha, bool conj_A, char* A,
               const stride_vector& stride_A_AC_,
                                    bool conj_B, char* B,
               const stride_vector& stride_B_BC_,
               const scalar&  beta, bool conj_C, char* C,
               const stride_vector& stride_C_AC_,
               const stride_vector& stride_C_BC_)
{
    const len_type ts = type_size[type];

    auto stride_A_AC = stride_A_AC_;
    auto stride_B_BC = stride_B_BC_;
    auto stride_C_AC = stride_C_AC_;
    auto stride_C_BC = stride_C_BC_;

    auto reorder_AC = detail::sort_by_stride(stride_C_AC, stride_A_AC);
    auto reorder_BC = detail::sort_by_stride(stride_C_BC, stride_B_BC);

    auto m = len_AC[reorder_AC[0]];
    auto n = len_BC[reorder_BC[0]];
    auto rs_C = stride_C_AC[reorder_AC[0]];
    auto cs_C = stride_C_BC[reorder_BC[0]];
    auto inc_A = stride_A_AC[reorder_AC[0]];
    auto inc_B = stride_B_BC[reorder_BC[0]];

    reorder_AC.erase(reorder_AC.begin());
    reorder_BC.erase(reorder_BC.begin());

    for (auto& s : stride_A_AC) s *= ts;
    for (auto& s : stride_B_BC) s *= ts;
    for (auto& s : stride_C_AC) s *= ts;
    for (auto& s : stride_C_BC) s *= ts;

    len_type m2 = stl_ext::prod(len_AC)/m;
    len_type n2 = stl_ext::prod(len_BC)/n;

    if (comm.master()) flops += 2*m*m2*n*n2;

    unsigned nt_l, nt_m;
    std::tie(nt_l, nt_m) = partition_2x2(comm.num_threads(), m2*n2, m2*n2, m*n, m*n);

    auto subcomm = comm.gang(TCI_EVENLY, nt_l);

    subcomm.distribute_over_gangs(m2*n2,
    [&](len_type l_min, len_type l_max)
    {
        viterator<3> iter_ABC(stl_ext::appended(stl_ext::permuted(len_AC, reorder_AC),
                                                stl_ext::permuted(len_BC, reorder_BC)),
                              stl_ext::appended(stl_ext::permuted(stride_A_AC, reorder_AC),
                                                stride_vector(reorder_BC.size())),
                              stl_ext::appended(stride_vector(reorder_AC.size()),
                                                stl_ext::permuted(stride_B_BC, reorder_BC)),
                              stl_ext::appended(stl_ext::permuted(stride_C_AC, reorder_AC),
                                                stl_ext::permuted(stride_C_BC, reorder_BC)));

        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        iter_ABC.position(l_min, A1, B1, C1);

        for (len_type l = l_min;l < l_max;l++)
        {
            iter_ABC.next(A1, B1, C1);

            const len_type NF = cfg.addf_nf.def(type);

            if (rs_C <= cs_C)
            {
                subcomm.distribute_over_threads(m, n,
                [&](len_type m_min, len_type m_max, len_type n_min, len_type n_max)
                {
                    void* Cs[16];

                    for (len_type j = n_min;j < n_max;j += NF)
                    {
                        for (len_type k = 0;k < NF;k++)
                            Cs[k] = C1 + m_min*rs_C*ts + (j+k)*cs_C*ts;

                        cfg.addf_rep_ukr.call(type, m_max-m_min, std::min(NF, n_max-j),
                                              &alpha, conj_A, A1 + m_min*inc_A*ts, inc_A,
                                                      conj_B, B1 + j*inc_B*ts, inc_B,
                                               &beta, conj_C, Cs, rs_C);
                    }
                });
            }
            else
            {
                subcomm.distribute_over_threads(m, n,
                [&](len_type m_min, len_type m_max, len_type n_min, len_type n_max)
                {
                    void* Cs[16];

                    for (len_type j = m_min;j < m_max;j += NF)
                    {
                        for (len_type k = 0;k < NF;k++)
                            Cs[k] = C1 + (j+k)*rs_C*ts + n_min*cs_C*ts;

                        cfg.addf_rep_ukr.call(type, n_max-n_min, std::min(NF, m_max-j),
                                              &alpha, conj_B, B1 + n_min*inc_B*ts, inc_B,
                                                      conj_A, A1 + j*inc_A*ts, inc_A,
                                               &beta, conj_C, Cs, cs_C);
                    }
                });
            }
        }
    });
}

void mult_blis(type_t type, const communicator& comm, const config& cfg,
                   const len_vector& len_AB,
                   const len_vector& len_AC,
                   const len_vector& len_BC,
                   const scalar& alpha, bool conj_A, char* A,
                   const stride_vector& stride_A_AB,
                   const stride_vector& stride_A_AC,
                                        bool conj_B, char* B,
                   const stride_vector& stride_B_AB,
                   const stride_vector& stride_B_BC,
                   const scalar&  beta, bool conj_C, char* C,
                   const stride_vector& stride_C_AC,
                   const stride_vector& stride_C_BC)
{
    auto reorder_AC = detail::sort_by_stride(stride_C_AC, stride_A_AC);
    auto reorder_BC = detail::sort_by_stride(stride_C_BC, stride_B_BC);
    auto reorder_AB = detail::sort_by_stride(stride_A_AB, stride_B_AB);

    unsigned unit_A_AC = unit_dim(stride_A_AC, reorder_AC);
    unsigned unit_C_AC = unit_dim(stride_C_AC, reorder_AC);
    unsigned unit_B_BC = unit_dim(stride_B_BC, reorder_BC);
    unsigned unit_C_BC = unit_dim(stride_C_BC, reorder_BC);
    unsigned unit_A_AB = unit_dim(stride_A_AB, reorder_AB);
    unsigned unit_B_AB = unit_dim(stride_B_AB, reorder_AB);

    TBLIS_ASSERT(unit_C_AC == 0 || unit_C_AC == len_AC.size());
    TBLIS_ASSERT(unit_C_BC == 0 || unit_C_BC == len_BC.size());
    TBLIS_ASSERT(unit_A_AB == 0 || unit_B_AB == 0 ||
                 (unit_A_AB == len_AB.size() && unit_B_AB == len_AB.size()));

    bool pack_M_3d = unit_A_AC > 0 && unit_A_AC < len_AC.size();
    bool pack_N_3d = unit_B_BC > 0 && unit_B_BC < len_BC.size();
    bool pack_K_3d = (unit_A_AB > 0 && unit_A_AB < len_AB.size()) ||
                     (unit_B_AB > 0 && unit_B_AB < len_AB.size());

    if (pack_M_3d)
        std::rotate(reorder_AC.begin()+1, reorder_AC.begin()+unit_A_AC, reorder_AC.end());

    if (pack_N_3d)
        std::rotate(reorder_BC.begin()+1, reorder_BC.begin()+unit_B_BC, reorder_BC.end());

    if (pack_K_3d)
        std::rotate(reorder_AB.begin()+1, reorder_AB.begin()+std::max(unit_A_AB, unit_B_AB), reorder_AB.end());

    tensor_matrix at(alpha, conj_A,
                     stl_ext::permuted(len_AC, reorder_AC),
                     stl_ext::permuted(len_AB, reorder_AB),
                     A,
                     stl_ext::permuted(stride_A_AC, reorder_AC),
                     stl_ext::permuted(stride_A_AB, reorder_AB),
                     pack_M_3d, pack_K_3d);

    tensor_matrix bt({1, type}, conj_B,
                     stl_ext::permuted(len_AB, reorder_AB),
                     stl_ext::permuted(len_BC, reorder_BC),
                     B,
                     stl_ext::permuted(stride_B_AB, reorder_AB),
                     stl_ext::permuted(stride_B_BC, reorder_BC),
                     pack_K_3d, pack_N_3d);

    tensor_matrix ct(beta, conj_C,
                     stl_ext::permuted(len_AC, reorder_AC),
                     stl_ext::permuted(len_BC, reorder_BC),
                     C,
                     stl_ext::permuted(stride_C_AC, reorder_AC),
                     stl_ext::permuted(stride_C_BC, reorder_BC),
                     pack_M_3d, pack_N_3d);

    GotoGEMM{}(comm, cfg, at, bt, ct);
}

void mult_blis(type_t type, const communicator& comm, const config& cfg,
               const len_vector& len_AB,
               const len_vector& len_AC,
               const len_vector& len_ABC,
               const scalar& alpha, bool conj_A, char* A,
               const stride_vector& stride_A_AB_,
               const stride_vector& stride_A_AC_,
               const stride_vector& stride_A_ABC_,
                                    bool conj_B, char* B,
               const stride_vector& stride_B_AB_,
               const stride_vector& stride_B_ABC_,
               const scalar&  beta, bool conj_C, char* C,
               const stride_vector& stride_C_AC_,
               const stride_vector& stride_C_ABC_)
{
    const len_type ts = type_size[type];

    auto stride_A_AB = stride_A_AB_;
    auto stride_A_AC = stride_A_AC_;
    auto stride_A_ABC = stride_A_ABC_;
    auto stride_B_AB = stride_B_AB_;
    auto stride_B_ABC = stride_B_ABC_;
    auto stride_C_AC = stride_C_AC_;
    auto stride_C_ABC = stride_C_ABC_;

    auto reorder_AC = detail::sort_by_stride(stride_A_AC, stride_C_AC);
    auto reorder_AB = detail::sort_by_stride(stride_A_AB, stride_B_AB);
    auto reorder_ABC = detail::sort_by_stride(stride_C_ABC, stride_A_ABC, stride_B_ABC);

    auto m = len_AC[reorder_AC[0]];
    auto n = len_AB[reorder_AB[0]];
    auto rs_A = stride_A_AC[reorder_AC[0]];
    auto cs_A = stride_A_AB[reorder_AB[0]];
    auto inc_B = stride_B_AB[reorder_AB[0]];
    auto inc_C = stride_C_AC[reorder_AC[0]];

    reorder_AC.erase(reorder_AC.begin());
    reorder_AB.erase(reorder_AB.begin());

    for (auto& s : stride_A_AC) s *= ts;
    for (auto& s : stride_A_AB) s *= ts;
    for (auto& s : stride_B_AB) s *= ts;
    for (auto& s : stride_C_AC) s *= ts;
    for (auto& s : stride_A_ABC) s *= ts;
    for (auto& s : stride_B_ABC) s *= ts;
    for (auto& s : stride_C_ABC) s *= ts;

    len_type l = stl_ext::prod(len_ABC);
    len_type m2 = stl_ext::prod(len_AC)/m;
    len_type n2 = stl_ext::prod(len_AB)/n;

    if (comm.master()) flops += 2*m*m2*n*n2*l;

    unsigned nt_l, nt_m;
    std::tie(nt_l, nt_m) = partition_2x2(comm.num_threads(), l*m2, l*m2, m, m);

    auto subcomm = comm.gang(TCI_EVENLY, nt_l);

    subcomm.distribute_over_gangs(l*m2,
    [&](len_type l_min, len_type l_max)
    {
        viterator<2> iter_AB(stl_ext::permuted(len_AB, reorder_AB),
                             stl_ext::permuted(stride_A_AB, reorder_AB),
                             stl_ext::permuted(stride_B_AB, reorder_AB));

        viterator<3> iter_ABC(stl_ext::appended(stl_ext::permuted(len_ABC, reorder_ABC),
                                                stl_ext::permuted(len_AC, reorder_AC)),
                              stl_ext::appended(stl_ext::permuted(stride_A_ABC, reorder_ABC),
                                                stl_ext::permuted(stride_A_AC, reorder_AC)),
                              stl_ext::appended(stl_ext::permuted(stride_B_ABC, reorder_ABC),
                                                stride_vector(reorder_AC.size())),
                              stl_ext::appended(stl_ext::permuted(stride_C_ABC, reorder_ABC),
                                                stl_ext::permuted(stride_C_AC, reorder_AC)));

        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        iter_ABC.position(l_min, A1, B1, C1);

        for (len_type l = l_min;l < l_max;l++)
        {
            iter_ABC.next(A1, B1, C1);

            auto local_beta = beta;
            auto local_conj_C = conj_C;

            while (iter_AB.next(A1, B1))
            {
                if (rs_A <= cs_A)
                {
                    const len_type NF = cfg.addf_nf.def(type);

                    subcomm.distribute_over_threads(m,
                    [&,local_beta,local_conj_C](len_type m_min, len_type m_max) mutable
                    {
                        const void* As[16];

                        for (len_type j = 0;j < n;j += NF)
                        {
                            for (len_type k = 0;k < NF;k++)
                                As[k] = A1 + m_min*rs_A*ts + (j+k)*cs_A*ts;

                            cfg.addf_sum_ukr.call(type, m_max-m_min, std::min(NF, n-j),
                                                       &alpha,       conj_A, As, rs_A,
                                                                     conj_B, B1 + j*inc_B*ts, inc_B,
                                                  &local_beta, local_conj_C, C1 + m_min*inc_C*ts, inc_C);

                            local_beta = 1.0;
                            local_conj_C = false;
                        }
                    });
                }
                else
                {
                    const len_type NF = cfg.dotf_nf.def(type);

                    subcomm.distribute_over_threads({m, NF},
                    [&](len_type m_min, len_type m_max)
                    {
                        for (len_type i = m_min;i < m_max;i += NF)
                        {
                            cfg.dotf_ukr.call(type, std::min(NF, m_max-i), n,
                                                   &alpha,       conj_A, A1 + i*rs_A*ts, rs_A, cs_A,
                                                                 conj_B, B1, inc_B,
                                              &local_beta, local_conj_C, C1 + i*inc_C*ts, inc_C);
                        }
                    });
                }

                subcomm.barrier();

                local_beta = 1.0;
                local_conj_C = false;
            }
        }
    });
}

void mult_blis(type_t type, const communicator& comm, const config& cfg,
               const len_vector& len_AC,
               const len_vector& len_BC,
               const len_vector& len_ABC,
               const scalar& alpha,
               bool conj_A, char* A,
               const stride_vector& stride_A_AC_,
               const stride_vector& stride_A_ABC_,
               bool conj_B, char* B,
               const stride_vector& stride_B_BC_,
               const stride_vector& stride_B_ABC_,
               const scalar& beta,
               bool conj_C, char* C,
               const stride_vector& stride_C_AC_,
               const stride_vector& stride_C_BC_,
               const stride_vector& stride_C_ABC_)
{
    const len_type ts = type_size[type];

    auto stride_A_AC = stride_A_AC_;
    auto stride_A_ABC = stride_A_ABC_;
    auto stride_B_BC = stride_B_BC_;
    auto stride_B_ABC = stride_B_ABC_;
    auto stride_C_AC = stride_C_AC_;
    auto stride_C_BC = stride_C_BC_;
    auto stride_C_ABC = stride_C_ABC_;

    auto reorder_AC = detail::sort_by_stride(stride_C_AC, stride_A_AC);
    auto reorder_BC = detail::sort_by_stride(stride_C_BC, stride_B_BC);
    auto reorder_ABC = detail::sort_by_stride(stride_C_ABC, stride_A_ABC, stride_B_ABC);

    auto m = len_AC[reorder_AC[0]];
    auto n = len_BC[reorder_BC[0]];
    auto rs_C = stride_C_AC[reorder_AC[0]];
    auto cs_C = stride_C_BC[reorder_BC[0]];
    auto inc_A = stride_A_AC[reorder_AC[0]];
    auto inc_B = stride_B_BC[reorder_BC[0]];

    reorder_AC.erase(reorder_AC.begin());
    reorder_BC.erase(reorder_BC.begin());

    for (auto& s : stride_A_AC) s *= ts;
    for (auto& s : stride_B_BC) s *= ts;
    for (auto& s : stride_C_AC) s *= ts;
    for (auto& s : stride_C_BC) s *= ts;
    for (auto& s : stride_A_ABC) s *= ts;
    for (auto& s : stride_B_ABC) s *= ts;
    for (auto& s : stride_C_ABC) s *= ts;

    len_type l = stl_ext::prod(len_ABC);
    len_type m2 = stl_ext::prod(len_AC)/m;
    len_type n2 = stl_ext::prod(len_BC)/n;

    if (comm.master()) flops += 2*m*m2*n*n2*l;

    unsigned nt_l, nt_m;
    std::tie(nt_l, nt_m) = partition_2x2(comm.num_threads(), l*m2*n2, l*m2*n2, m*n, m*n);

    auto subcomm = comm.gang(TCI_EVENLY, nt_l);

    subcomm.distribute_over_gangs(l*m2*n2,
    [&](len_type l_min, len_type l_max)
    {
        viterator<3> iter_ABC(stl_ext::appended(stl_ext::permuted(len_ABC, reorder_ABC),
                                                stl_ext::permuted(len_AC, reorder_AC),
                                                stl_ext::permuted(len_BC, reorder_BC)),
                              stl_ext::appended(stl_ext::permuted(stride_A_ABC, reorder_ABC),
                                                stl_ext::permuted(stride_A_AC, reorder_AC),
                                                stride_vector(reorder_BC.size())),
                              stl_ext::appended(stl_ext::permuted(stride_B_ABC, reorder_ABC),
                                                stride_vector(reorder_AC.size()),
                                                stl_ext::permuted(stride_B_BC, reorder_BC)),
                              stl_ext::appended(stl_ext::permuted(stride_C_ABC, reorder_ABC),
                                                stl_ext::permuted(stride_C_AC, reorder_AC),
                                                stl_ext::permuted(stride_C_BC, reorder_BC)));

        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        iter_ABC.position(l_min, A1, B1, C1);

        for (len_type l = l_min;l < l_max;l++)
        {
            iter_ABC.next(A1, B1, C1);

            const len_type NF = cfg.addf_nf.def(type);

            if (rs_C <= cs_C)
            {
                subcomm.distribute_over_threads(m, n,
                [&](len_type m_min, len_type m_max, len_type n_min, len_type n_max)
                {
                    void* Cs[16];

                    for (len_type j = n_min;j < n_max;j += NF)
                    {
                        for (len_type k = 0;k < NF;k++)
                            Cs[k] = C1 + m_min*rs_C*ts + (j+k)*cs_C*ts;

                        cfg.addf_rep_ukr.call(type, m_max-m_min, std::min(NF, n_max-j),
                                              &alpha, conj_A, A1 + m_min*inc_A*ts, inc_A,
                                                      conj_B, B1 + j*inc_B*ts, inc_B,
                                               &beta, conj_C, Cs, rs_C);
                    }
                });
            }
            else
            {
                subcomm.distribute_over_threads(m, n,
                [&](len_type m_min, len_type m_max, len_type n_min, len_type n_max)
                {
                    void* Cs[16];

                    for (len_type j = m_min;j < m_max;j += NF)
                    {
                        for (len_type k = 0;k < NF;k++)
                            Cs[k] = C1 + (j+k)*rs_C*ts + n_min*cs_C*ts;

                        cfg.addf_rep_ukr.call(type, n_max-n_min, std::min(NF, m_max-j),
                                              &alpha, conj_B, B1 + n_min*inc_B*ts, inc_B,
                                                      conj_A, A1 + j*inc_A*ts, inc_A,
                                               &beta, conj_C, Cs, cs_C);
                    }
                });
            }
        }
    });
}

void mult_blis(type_t type, const communicator& comm, const config& cfg,
               const len_vector& len_AB,
               const len_vector& len_AC,
               const len_vector& len_BC,
               const len_vector& len_ABC,
               const scalar& alpha,
               bool conj_A, char* A,
               const stride_vector& stride_A_AB,
               const stride_vector& stride_A_AC,
               const stride_vector& stride_A_ABC,
               bool conj_B, char* B,
               const stride_vector& stride_B_AB,
               const stride_vector& stride_B_BC,
               const stride_vector& stride_B_ABC,
               const scalar& beta,
               bool conj_C, char* C,
               const stride_vector& stride_C_AC,
               const stride_vector& stride_C_BC,
               const stride_vector& stride_C_ABC)
{
    const len_type ts = type_size[type];

    auto reorder_AC = detail::sort_by_stride(stride_C_AC, stride_A_AC);
    auto reorder_BC = detail::sort_by_stride(stride_C_BC, stride_B_BC);
    auto reorder_AB = detail::sort_by_stride(stride_A_AB, stride_B_AB);
    auto reorder_ABC = detail::sort_by_stride(stride_C_ABC, stride_A_ABC, stride_B_ABC);

    unsigned unit_A_AC = unit_dim(stride_A_AC, reorder_AC);
    unsigned unit_C_AC = unit_dim(stride_C_AC, reorder_AC);
    unsigned unit_B_BC = unit_dim(stride_B_BC, reorder_BC);
    unsigned unit_C_BC = unit_dim(stride_C_BC, reorder_BC);
    unsigned unit_A_AB = unit_dim(stride_A_AB, reorder_AB);
    unsigned unit_B_AB = unit_dim(stride_B_AB, reorder_AB);

    TBLIS_ASSERT(unit_C_AC == 0 || unit_C_AC == len_AC.size());
    TBLIS_ASSERT(unit_C_BC == 0 || unit_C_BC == len_BC.size());
    TBLIS_ASSERT(unit_A_AB == 0 || unit_B_AB == 0 ||
                 (unit_A_AB == len_AB.size() && unit_B_AB == len_AB.size()));

    bool pack_M_3d = unit_A_AC > 0 && unit_A_AC < len_AC.size();
    bool pack_N_3d = unit_B_BC > 0 && unit_B_BC < len_BC.size();
    bool pack_K_3d = (unit_A_AB > 0 && unit_A_AB < len_AB.size()) ||
                     (unit_B_AB > 0 && unit_B_AB < len_AB.size());

    if (pack_M_3d)
        std::rotate(reorder_AC.begin()+1, reorder_AC.begin()+unit_A_AC, reorder_AC.end());

    if (pack_N_3d)
        std::rotate(reorder_BC.begin()+1, reorder_BC.begin()+unit_B_BC, reorder_BC.end());

    if (pack_K_3d)
        std::rotate(reorder_AB.begin()+1, reorder_AB.begin()+std::max(unit_A_AB, unit_B_AB), reorder_AB.end());

    scalar one(1.0, type);

    len_type m = stl_ext::prod(len_AC);
    len_type n = stl_ext::prod(len_BC);
    len_type k = stl_ext::prod(len_AB);
    len_type l = stl_ext::prod(len_ABC);

    if (comm.master()) flops += 2*m*n*k*l;

    unsigned nt_l, nt_mn;
    std::tie(nt_l, nt_mn) =
        partition_2x2(comm.num_threads(), l, l, m*n, m*n);

    auto subcomm = comm.gang(TCI_EVENLY, nt_l);

    auto len_AB_r = stl_ext::permuted(len_AB, reorder_AB);
    auto len_AC_r = stl_ext::permuted(len_AC, reorder_AC);
    auto len_BC_r = stl_ext::permuted(len_BC, reorder_BC);
    auto stride_A_AB_r = stl_ext::permuted(stride_A_AB, reorder_AB);
    auto stride_B_AB_r = stl_ext::permuted(stride_B_AB, reorder_AB);
    auto stride_A_AC_r = stl_ext::permuted(stride_A_AC, reorder_AC);
    auto stride_C_AC_r = stl_ext::permuted(stride_C_AC, reorder_AC);
    auto stride_B_BC_r = stl_ext::permuted(stride_B_BC, reorder_BC);
    auto stride_C_BC_r = stl_ext::permuted(stride_C_BC, reorder_BC);
    auto stride_A_ABC_r = stl_ext::permuted(stride_A_ABC, reorder_ABC);
    auto stride_B_ABC_r = stl_ext::permuted(stride_B_ABC, reorder_ABC);
    auto stride_C_ABC_r = stl_ext::permuted(stride_C_ABC, reorder_ABC);

    for (auto& s : stride_A_ABC_r) s *= ts;
    for (auto& s : stride_B_ABC_r) s *= ts;
    for (auto& s : stride_C_ABC_r) s *= ts;

    subcomm.distribute_over_gangs(l,
    [&](len_type l_min, len_type l_max)
    {
        viterator<3> iter_ABC(stl_ext::permuted(len_ABC, reorder_ABC),
                              stride_A_ABC_r, stride_B_ABC_r, stride_C_ABC_r);

        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        iter_ABC.position(l_min, A1, B1, C1);

        for (len_type l = l_min;l < l_max;l++)
        {
            iter_ABC.next(A1, B1, C1);

            tensor_matrix at(alpha, conj_A, len_AC_r, len_AB_r, A1,
                             stride_A_AC_r, stride_A_AB_r, pack_M_3d, pack_K_3d);

            tensor_matrix bt(  one, conj_B, len_AB_r, len_BC_r, B1,
                             stride_B_AB_r, stride_B_BC_r, pack_K_3d, pack_N_3d);

            tensor_matrix ct( beta, conj_C, len_AC_r, len_BC_r, C1,
                             stride_C_AC_r, stride_C_BC_r, pack_M_3d, pack_N_3d);

            GotoGEMM{}(subcomm, cfg, at, bt, ct);
        }
    });
}

template <typename T>
void mult_blas(const communicator& comm, const config& cfg,
               const len_vector& len_AB_,
               const len_vector& len_AC_,
               const len_vector& len_BC_,
               const len_vector& len_ABC_,
               T alpha, bool conj_A, T* A,
               const stride_vector& stride_A_AB_,
               const stride_vector& stride_A_AC_,
               const stride_vector& stride_A_ABC_,
                        bool conj_B, T* B,
               const stride_vector& stride_B_AB_,
               const stride_vector& stride_B_BC_,
               const stride_vector& stride_B_ABC_,
               T  beta, bool conj_C, T* C,
               const stride_vector& stride_C_AC_,
               const stride_vector& stride_C_BC_,
               const stride_vector& stride_C_ABC_)
{
    varray<T> ar, br, cr;

    auto len_AB = len_AB_; len_AB.push_back(1);
    auto len_AC = len_AC_; len_AC.push_back(1);
    auto len_BC = len_BC_; len_BC.push_back(1);
    auto len_ABC = len_ABC_; len_ABC.push_back(1);
    auto stride_A_AB = stride_A_AB_; stride_A_AB.push_back(1);
    auto stride_B_AB = stride_B_AB_; stride_B_AB.push_back(1);
    auto stride_A_AC = stride_A_AC_; stride_A_AC.push_back(1);
    auto stride_C_AC = stride_C_AC_; stride_C_AC.push_back(1);
    auto stride_B_BC = stride_B_BC_; stride_B_BC.push_back(1);
    auto stride_C_BC = stride_C_BC_; stride_C_BC.push_back(1);
    auto stride_A_ABC = stride_A_ABC_; stride_A_ABC.push_back(1);
    auto stride_B_ABC = stride_B_ABC_; stride_B_ABC.push_back(1);
    auto stride_C_ABC = stride_C_ABC_; stride_C_ABC.push_back(1);

    if (comm.master())
    {
        ar.reset(len_AC+len_AB);
        br.reset(len_AB+len_BC);
        cr.reset(len_AC+len_BC);
    }

    comm.broadcast(
    [&](varray<T>& ar, varray<T>& br, varray<T>& cr)
    {
        auto am = matricize(ar, len_AC.size());
        auto bm = matricize(br, len_AB.size());
        auto cm = matricize(cr, len_AC.size());

        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        viterator<3> it(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

        while (it.next(A1, B1, C1))
        {
            add(type_tag<T>::value, comm, cfg, {}, {}, ar.lengths(),
                alpha, conj_A, reinterpret_cast<char*>(       A1), {}, stride_A_AC+stride_A_AB,
                 T(0),  false, reinterpret_cast<char*>(ar.data()), {},            ar.strides());

            add(type_tag<T>::value, comm, cfg, {}, {}, br.lengths(),
                T(1), conj_B, reinterpret_cast<char*>(       B1), {}, stride_B_AB+stride_B_BC,
                T(0),  false, reinterpret_cast<char*>(br.data()), {},            br.strides());

            normal_matrix at(T(1), false, am.length(0), am.length(1),
                             reinterpret_cast<char*>(am.data()), am.stride(0), am.stride(1));

            normal_matrix bt(T(1), false, bm.length(0), bm.length(1),
                             reinterpret_cast<char*>(bm.data()), bm.stride(0), bm.stride(1));

            normal_matrix ct(T(0), false, cm.length(0), cm.length(1),
                             reinterpret_cast<char*>(cm.data()), cm.stride(0), cm.stride(1));

            GotoGEMM{}(comm, cfg, at, bt, ct);

            add(type_tag<T>::value, comm, cfg, {}, {}, cr.lengths(),
                T(1),  false, reinterpret_cast<char*>(cr.data()), {},             cr.strides(),
                beta, conj_C, reinterpret_cast<char*>(       C1), {}, stride_C_AC+stride_C_BC);
        }
    },
    ar, br, cr);
}

void mult_ref(type_t type, const communicator& comm, const config& cfg,
              const len_vector& len_AB,
              const len_vector& len_AC,
              const len_vector& len_BC,
              const len_vector& len_ABC,
              const scalar& alpha,
              bool conj_A, char* A,
              const stride_vector& stride_A_AB_,
              const stride_vector& stride_A_AC_,
              const stride_vector& stride_A_ABC_,
              bool conj_B, char* B,
              const stride_vector& stride_B_AB_,
              const stride_vector& stride_B_BC_,
              const stride_vector& stride_B_ABC_,
              const scalar& beta,
              bool conj_C, char* C,
              const stride_vector& stride_C_AC_,
              const stride_vector& stride_C_BC_,
              const stride_vector& stride_C_ABC_)
{
    (void)cfg;

    const len_type ts = type_size[type];

    len_type n = stl_ext::prod(len_ABC);

    auto stride_A_AB = stride_A_AB_;
    auto stride_B_AB = stride_B_AB_;
    auto stride_A_AC = stride_A_AC_;
    auto stride_C_AC = stride_C_AC_;
    auto stride_B_BC = stride_B_BC_;
    auto stride_C_BC = stride_C_BC_;
    auto stride_A_ABC = stride_A_ABC_;
    auto stride_B_ABC = stride_B_ABC_;
    auto stride_C_ABC = stride_C_ABC_;

    for (auto& s : stride_A_AC) s *= ts;
    for (auto& s : stride_B_BC) s *= ts;
    for (auto& s : stride_C_AC) s *= ts;
    for (auto& s : stride_C_BC) s *= ts;
    for (auto& s : stride_A_ABC) s *= ts;
    for (auto& s : stride_B_ABC) s *= ts;
    for (auto& s : stride_C_ABC) s *= ts;

    comm.distribute_over_threads(n,
    [&](len_type n_min, len_type n_max)
    {
        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);
        viterator<2> iter_BC(len_BC, stride_B_BC, stride_C_BC);
        viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);
        iter_ABC.position(n_min, A1, B1, C1);

        for (len_type i = n_min;i < n_max;i++)
        {
            iter_ABC.next(A1, B1, C1);

            while (iter_AC.next(A1, C1))
            {
                while (iter_BC.next(B1, C1))
                {
                    scalar sum(0, type);

                    dot(type, single, cfg, len_AB,
                        conj_A, reinterpret_cast<char*>(A1), stride_A_AB,
                        conj_B, reinterpret_cast<char*>(B1), stride_B_AB, sum.raw());

                    add(type, alpha, false, sum.raw(), beta, conj_C, C1);
                }
            }
        }
    });
}

void mult_vec(type_t type, const communicator& comm, const config& cfg,
              const len_vector& len_ABC,
              const scalar& alpha, bool conj_A, char* A,
              const stride_vector& stride_A_ABC,
                                   bool conj_B, char* B,
              const stride_vector& stride_B_ABC,
              const scalar&  beta, bool conj_C, char* C,
              const stride_vector& stride_C_ABC)
{
    bool empty = len_ABC.size() == 0;

    const len_type ts = type_size[type];

    len_type n0 = (empty ? 1 : len_ABC[0]);
    len_vector len1(len_ABC.begin() + !empty, len_ABC.end());
    len_type n1 = stl_ext::prod(len1);

    stride_type stride_A0 = (empty ? 1 : stride_A_ABC[0]);
    stride_type stride_B0 = (empty ? 1 : stride_B_ABC[0]);
    stride_type stride_C0 = (empty ? 1 : stride_C_ABC[0]);
    len_vector stride_A1, stride_B1, stride_C1;
    for (unsigned i = 1;i < stride_A_ABC.size();i++) stride_A1.push_back(stride_A_ABC[i]*ts);
    for (unsigned i = 1;i < stride_B_ABC.size();i++) stride_B1.push_back(stride_B_ABC[i]*ts);
    for (unsigned i = 1;i < stride_C_ABC.size();i++) stride_C1.push_back(stride_C_ABC[i]*ts);

    comm.distribute_over_threads(n0, n1,
    [&](len_type n0_min, len_type n0_max, len_type n1_min, len_type n1_max)
    {
        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        viterator<3> iter_ABC(len1, stride_A1, stride_B1, stride_C1);
        iter_ABC.position(n1_min, A1, B1, C1);
        A1 += n0_min*stride_A0*ts;
        B1 += n0_min*stride_B0*ts;
        C1 += n0_min*stride_C0*ts;

        for (len_type i = n1_min;i < n1_max;i++)
        {
            iter_ABC.next(A1, B1, C1);

            cfg.mult_ukr.call(type, n0_max-n0_min,
                              &alpha, conj_A, A1, stride_A0,
                                      conj_B, B1, stride_B0,
                               &beta, conj_C, C1, stride_C0);
        }
    });
}

void mult(type_t type, const communicator& comm, const config& cfg,
          const len_vector& len_AB,
          const len_vector& len_AC,
          const len_vector& len_BC,
          const len_vector& len_ABC,
          const scalar& alpha, bool conj_A, char* A,
          const stride_vector& stride_A_AB,
          const stride_vector& stride_A_AC,
          const stride_vector& stride_A_ABC,
                               bool conj_B, char* B,
          const stride_vector& stride_B_AB,
          const stride_vector& stride_B_BC,
          const stride_vector& stride_B_ABC,
          const scalar&  beta, bool conj_C, char* C,
          const stride_vector& stride_C_AC,
          const stride_vector& stride_C_BC,
          const stride_vector& stride_C_ABC)
{
    const len_type ts = type_size[type];
    auto n_AB = stl_ext::prod(len_AB);
    auto n_AC = stl_ext::prod(len_AC);
    auto n_BC = stl_ext::prod(len_BC);
    auto n_ABC = stl_ext::prod(len_ABC);

    if (n_AC == 0 || n_BC == 0 || n_ABC == 0) return;

    if (n_AB == 0)
    {
        if (beta.is_zero())
        {
            set(type, comm, cfg, len_AC+len_BC+len_ABC, beta, C,
                stride_C_AC+stride_C_BC+stride_C_ABC);
        }
        else if (!beta.is_one() || (beta.is_complex() && conj_C))
        {
            scale(type, comm, cfg, len_AC+len_BC+len_ABC, beta, conj_C, C,
                  stride_C_AC+stride_C_BC+stride_C_ABC);
        }

        return;
    }

    if (impl == REFERENCE)
    {
        mult_ref(type, comm, cfg,
                 len_AB, len_AC, len_BC, len_ABC,
                 alpha, conj_A, A, stride_A_AB, stride_A_AC, stride_A_ABC,
                        conj_B, B, stride_B_AB, stride_B_BC, stride_B_ABC,
                  beta, conj_C, C, stride_C_AC, stride_C_BC, stride_C_ABC);
        comm.barrier();
        return;
    }
    else if (impl == BLAS_BASED)
    {
        switch (type)
        {
            case TYPE_FLOAT:
                mult_blas(comm, cfg,
                          len_AB, len_AC, len_BC, len_ABC,
                          alpha.get<float>(), conj_A, reinterpret_cast<float*>(A),
                                              stride_A_AB, stride_A_AC, stride_A_ABC,
                                              conj_B, reinterpret_cast<float*>(B),
                                              stride_B_AB, stride_B_BC, stride_B_ABC,
                           beta.get<float>(), conj_C, reinterpret_cast<float*>(C),
                                              stride_C_AC, stride_C_BC, stride_C_ABC);
                break;
            case TYPE_DOUBLE:
                mult_blas(comm, cfg,
                          len_AB, len_AC, len_BC, len_ABC,
                          alpha.get<double>(), conj_A, reinterpret_cast<double*>(A),
                                               stride_A_AB, stride_A_AC, stride_A_ABC,
                                               conj_B, reinterpret_cast<double*>(B),
                                               stride_B_AB, stride_B_BC, stride_B_ABC,
                           beta.get<double>(), conj_C, reinterpret_cast<double*>(C),
                                               stride_C_AC, stride_C_BC, stride_C_ABC);
                break;
            case TYPE_SCOMPLEX:
                mult_blas(comm, cfg,
                          len_AB, len_AC, len_BC, len_ABC,
                          alpha.get<scomplex>(), conj_A, reinterpret_cast<scomplex*>(A),
                                                 stride_A_AB, stride_A_AC, stride_A_ABC,
                                                 conj_B, reinterpret_cast<scomplex*>(B),
                                                 stride_B_AB, stride_B_BC, stride_B_ABC,
                           beta.get<scomplex>(), conj_C, reinterpret_cast<scomplex*>(C),
                                                 stride_C_AC, stride_C_BC, stride_C_ABC);
                break;
            case TYPE_DCOMPLEX:
                mult_blas(comm, cfg,
                          len_AB, len_AC, len_BC, len_ABC,
                          alpha.get<dcomplex>(), conj_A, reinterpret_cast<dcomplex*>(A),
                                                 stride_A_AB, stride_A_AC, stride_A_ABC,
                                                 conj_B, reinterpret_cast<dcomplex*>(B),
                                                 stride_B_AB, stride_B_BC, stride_B_ABC,
                           beta.get<dcomplex>(), conj_C, reinterpret_cast<dcomplex*>(C),
                                                 stride_C_AC, stride_C_BC, stride_C_ABC);
                break;
        }
        comm.barrier();
        return;
    }

    enum
    {
        HAS_NONE = 0x0,
        HAS_AB   = 0x1,
        HAS_AC   = 0x2,
        HAS_BC   = 0x4,
        HAS_ABC  = 0x8
    };

    int groups = (n_AB  == 1 ? 0 : HAS_AB ) +
                 (n_AC  == 1 ? 0 : HAS_AC ) +
                 (n_BC  == 1 ? 0 : HAS_BC ) +
                 (n_ABC == 1 ? 0 : HAS_ABC);

    scalar zero(0, type);
    scalar sum(0, type);
    auto stride_A_ABC_ts = stride_A_ABC; for (auto& s : stride_A_ABC_ts) s *= ts;
    auto stride_B_ABC_ts = stride_B_ABC; for (auto& s : stride_B_ABC_ts) s *= ts;
    auto stride_C_ABC_ts = stride_C_ABC; for (auto& s : stride_C_ABC_ts) s *= ts;
    viterator<3> iter_ABC(len_ABC, stride_A_ABC_ts, stride_B_ABC_ts, stride_C_ABC_ts);

    switch (groups)
    {
        case HAS_NONE:
        {
            if (comm.master())
                mult(type, alpha, conj_A, A, conj_B, B, beta, conj_C, C);
        }
        break;
        case HAS_AB:
        {
            dot(type, comm, cfg, len_AB, conj_A, A, stride_A_AB,
                                         conj_B, B, stride_B_AB, sum.raw());

            if (comm.master())
                add(type, alpha, false, sum.raw(), beta, conj_C, C);
        }
        break;
        case HAS_AC:
        {
            add(type, alpha, conj_B, B, zero, false, sum.raw());

            add(type, comm, cfg, {}, {}, len_AC,
                 sum, conj_A, A, {}, stride_A_AC,
                beta, conj_C, C, {}, stride_C_AC);
        }
        break;
        case HAS_BC:
        {
            add(type, alpha, conj_A, A, zero, false, sum.raw());

            add(type, comm, cfg, {}, {}, len_BC,
                 sum, conj_B, B, {}, stride_B_BC,
                beta, conj_C, C, {}, stride_C_BC);
        }
        break;
        case HAS_ABC:
        {
            mult_vec(type, comm, cfg, len_ABC,
                     alpha, conj_A, A, stride_A_ABC,
                            conj_B, B, stride_B_ABC,
                      beta, conj_C, C, stride_C_ABC);
        }
        break;
        case HAS_AC+HAS_BC:
        {
            mult_blis(type, comm, cfg, len_AC, len_BC,
                      alpha, conj_A, A, stride_A_AC,
                             conj_B, B, stride_B_BC,
                       beta, conj_C, C, stride_C_AC, stride_C_BC);
        }
        break;
        case HAS_AB+HAS_AC:
        {
            mult_blis(type, comm, cfg, len_AB, len_AC,
                      alpha, conj_A, A, stride_A_AB, stride_A_AC,
                             conj_B, B, stride_B_AB,
                       beta, conj_C, C, stride_C_AC);
        }
        break;
        case HAS_AB+HAS_BC:
        {
            mult_blis(type, comm, cfg, len_AB, len_BC,
                      alpha, conj_B, B, stride_B_AB, stride_B_BC,
                             conj_A, A, stride_A_AB,
                       beta, conj_C, C, stride_C_BC);
        }
        break;
        case HAS_AB+HAS_AC+HAS_BC:
        {
            mult_blis(type, comm, cfg, len_AB, len_AC, len_BC,
                      alpha, conj_A, A, stride_A_AB, stride_A_AC,
                             conj_B, B, stride_B_AB, stride_B_BC,
                       beta, conj_C, C, stride_C_AC, stride_C_BC);
        }
        break;
        case HAS_AB+HAS_ABC:
        {
            while (iter_ABC.next(A, B, C))
            {
                dot(type, comm, cfg, len_AB, conj_A, A, stride_A_AB,
                                             conj_B, B, stride_B_AB, sum.raw());

                if (comm.master())
                    add(type, alpha, false, sum.raw(), beta, conj_C, C);
            }
        }
        break;
        case HAS_AC+HAS_ABC:
        {
            while (iter_ABC.next(A, B, C))
            {
                add(type, alpha, conj_B, B, zero, false, sum.raw());

                add(type, comm, cfg, {}, {}, len_AC,
                     sum, conj_A, A, {}, stride_A_AC,
                    beta, conj_C, C, {}, stride_C_AC);
            }
        }
        break;
        case HAS_BC+HAS_ABC:
        {
            while (iter_ABC.next(A, B, C))
            {
                add(type, alpha, conj_A, A, zero, false, sum.raw());

                add(type, comm, cfg, {}, {}, len_BC,
                     sum, conj_B, B, {}, stride_B_BC,
                    beta, conj_C, C, {}, stride_C_BC);
            }
        }
        break;
        case HAS_AC+HAS_BC+HAS_ABC:
        {
            mult_blis(type, comm, cfg, len_AC, len_BC, len_ABC,
                      alpha, conj_A, A, stride_A_AC, stride_A_ABC,
                             conj_B, B, stride_B_BC, stride_B_ABC,
                       beta, conj_C, C, stride_C_AC, stride_C_BC, stride_C_ABC);
        }
        break;
        case HAS_AB+HAS_AC+HAS_ABC:
        {
            mult_blis(type, comm, cfg,
                      len_AB, len_AC, len_ABC,
                      alpha, conj_A, A, stride_A_AB, stride_A_AC, stride_A_ABC,
                             conj_B, B, stride_B_AB, stride_B_ABC,
                       beta, conj_C, C, stride_C_AC, stride_C_ABC);
        }
        break;
        case HAS_AB+HAS_BC+HAS_ABC:
        {
            mult_blis(type, comm, cfg,
                      len_AB, len_BC, len_ABC,
                      alpha, conj_B, B, stride_B_AB, stride_B_BC, stride_B_ABC,
                             conj_A, A, stride_A_AB, stride_A_ABC,
                       beta, conj_C, C, stride_C_BC, stride_C_ABC);
        }
        break;
        case HAS_AB+HAS_AC+HAS_BC+HAS_ABC:
        {
            mult_blis(type, comm, cfg,
                      len_AB, len_AC, len_BC, len_ABC,
                      alpha, conj_A, A, stride_A_AB, stride_A_AC, stride_A_ABC,
                             conj_B, B, stride_B_AB, stride_B_BC, stride_B_ABC,
                       beta, conj_C, C, stride_C_AC, stride_C_BC, stride_C_ABC);
        }
        break;
    }

    comm.barrier();
}

}
}
