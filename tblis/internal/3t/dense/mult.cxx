#include <tblis/internal/dense.hpp>
#include <tblis/internal/memory_pool.hpp>
#include <tblis/internal/thread.hpp>

#include <tblis/matrix/normal_matrix.hpp>
#include <tblis/matrix/tensor_matrix.hpp>

#include <tblis/nodes/gemm.hpp>

namespace tblis
{

MemoryPool BuffersForA, BuffersForB, BuffersForC;

namespace internal
{

static
void mult_blis(type_t type, const communicator& comm, const config& cfg,
               const len_vector& len_AB_,
               const len_vector& len_AC_,
               const scalar& alpha, bool conj_A, char* A,
               const stride_vector& stride_A_AB_,
               const stride_vector& stride_A_AC_,
                                    bool conj_B, char* B,
               const stride_vector& stride_B_AB_,
               const scalar&  beta, bool conj_C, char* C,
               const stride_vector& stride_C_AC_)
{
    const len_type ts = type_size[type];

    len_vector len_AB(len_AB_.begin()+1, len_AB_.end());
    len_vector len_AC(len_AC_.begin()+1, len_AC_.end());
    stride_vector stride_A_AB(stride_A_AB_.begin()+1, stride_A_AB_.end());
    stride_vector stride_B_AB(stride_B_AB_.begin()+1, stride_B_AB_.end());
    stride_vector stride_A_AC(stride_A_AC_.begin()+1, stride_A_AC_.end());
    stride_vector stride_C_AC(stride_C_AC_.begin()+1, stride_C_AC_.end());

    auto m = len_AC_[0];
    auto n = len_AB_[0];
    auto rs_A = stride_A_AC_[0];
    auto cs_A = stride_A_AB_[0];
    auto inc_B = stride_B_AB_[0];
    auto inc_C = stride_C_AC_[0];

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
        viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
        viterator<2> iter_AC(len_AC, stride_A_AC, stride_C_AC);

        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        iter_AC.position(l_min, A1, C1);

        for (len_type l = l_min;l < l_max;l++)
        {
            iter_AC.next(A1, C1);

            auto beta1 = beta;
            auto conj_C1 = conj_C;

            while (iter_AB.next(A1, B1))
            {
                if (rs_A <= cs_A)
                {
                    const len_type NF = cfg.addf_nf.def(type);

                    subcomm.distribute_over_threads(m,
                    [&](len_type m_min, len_type m_max)
                    {
                        const void* As[16];
                        auto beta2 = beta1;
                        auto conj_C2 = conj_C1;

                        for (len_type j = 0;j < n;j += NF)
                        {
                            for (len_type k = 0;k < NF;k++)
                                As[k] = A1 + m_min*rs_A*ts + (j+k)*cs_A*ts;

                            cfg.addf_sum_ukr.call(type, m_max-m_min, std::min(NF, n-j),
                                                  &alpha, conj_A,  As,                  rs_A,
                                                          conj_B,  B1 +     j*inc_B*ts, inc_B,
                                                  &beta2, conj_C2, C1 + m_min*inc_C*ts, inc_C);

                            beta2 = 1.0;
                            conj_C2 = false;
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
                                              &alpha, conj_A,  A1 + i* rs_A*ts, rs_A, cs_A,
                                                      conj_B,  B1,              inc_B,
                                              &beta1, conj_C1, C1 + i*inc_C*ts, inc_C);
                        }
                    });
                }

                subcomm.barrier();

                beta1 = 1.0;
                conj_C1 = false;
            }
        }
    });
}

static
void mult_blis(type_t type, const communicator& comm, const config& cfg,
               const len_vector& len_AC_,
               const len_vector& len_BC_,
               const scalar& alpha, bool conj_A, char* A,
               const stride_vector& stride_A_AC_,
                                    bool conj_B, char* B,
               const stride_vector& stride_B_BC_,
               const scalar&  beta, bool conj_C, char* C,
               const stride_vector& stride_C_AC_,
               const stride_vector& stride_C_BC_)
{
    const len_type ts = type_size[type];

    len_vector len_AC(len_AC_.begin()+1, len_AC_.end());
    len_vector len_BC(len_BC_.begin()+1, len_BC_.end());
    stride_vector stride_A_AC(stride_A_AC_.begin()+1, stride_A_AC_.end());
    stride_vector stride_C_AC(stride_C_AC_.begin()+1, stride_C_AC_.end());
    stride_vector stride_B_BC(stride_B_BC_.begin()+1, stride_B_BC_.end());
    stride_vector stride_C_BC(stride_C_BC_.begin()+1, stride_C_BC_.end());

    auto m = len_AC_[0];
    auto n = len_BC_[0];
    auto rs_C = stride_C_AC_[0];
    auto cs_C = stride_C_BC_[0];
    auto inc_A = stride_A_AC_[0];
    auto inc_B = stride_B_BC_[0];

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
        len_vector len_ABC;
        stride_vector stride_A_ABC;
        stride_vector stride_B_ABC;
        stride_vector stride_C_ABC;

        if (rs_C <= cs_C)
        {
            len_ABC.insert(len_ABC.end(), len_AC.begin(), len_AC.end());
            stride_A_ABC.insert(stride_A_ABC.end(), stride_A_AC.begin(), stride_A_AC.end());
            stride_B_ABC.insert(stride_B_ABC.end(), len_AC.size(), 0);
            stride_C_ABC.insert(stride_C_ABC.end(), stride_C_AC.begin(), stride_C_AC.end());

            len_ABC.insert(len_ABC.end(), len_BC.begin(), len_BC.end());
            stride_A_ABC.insert(stride_A_ABC.end(), len_BC.size(), 0);
            stride_B_ABC.insert(stride_B_ABC.end(), stride_B_BC.begin(), stride_B_BC.end());
            stride_C_ABC.insert(stride_C_ABC.end(), stride_C_BC.begin(), stride_C_BC.end());
        }
        else
        {
            len_ABC.insert(len_ABC.end(), len_BC.begin(), len_BC.end());
            stride_A_ABC.insert(stride_A_ABC.end(), len_BC.size(), 0);
            stride_B_ABC.insert(stride_B_ABC.end(), stride_B_BC.begin(), stride_B_BC.end());
            stride_C_ABC.insert(stride_C_ABC.end(), stride_C_BC.begin(), stride_C_BC.end());

            len_ABC.insert(len_ABC.end(), len_AC.begin(), len_AC.end());
            stride_A_ABC.insert(stride_A_ABC.end(), stride_A_AC.begin(), stride_A_AC.end());
            stride_B_ABC.insert(stride_B_ABC.end(), len_AC.size(), 0);
            stride_C_ABC.insert(stride_C_ABC.end(), stride_C_AC.begin(), stride_C_AC.end());
        }

        viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

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

static
void mult_blis(type_t type, const communicator& comm, const config& cfg,
                   const len_vector& len_AB_,
                   const len_vector& len_AC_,
                   const len_vector& len_BC_,
                   const scalar& alpha, bool conj_A, char* A,
                   const stride_vector& stride_A_AB_,
                   const stride_vector& stride_A_AC_,
                                        bool conj_B, char* B,
                   const stride_vector& stride_B_AB_,
                   const stride_vector& stride_B_BC_,
                   const scalar&  beta, bool conj_C, char* C,
                   const stride_vector& stride_C_AC_,
                   const stride_vector& stride_C_BC_)
{
    auto len_AB = len_AB_;
    auto len_AC = len_AC_;
    auto len_BC = len_BC_;
    auto stride_A_AB = stride_A_AB_;
    auto stride_B_AB = stride_B_AB_;
    auto stride_A_AC = stride_A_AC_;
    auto stride_C_AC = stride_C_AC_;
    auto stride_B_BC = stride_B_BC_;
    auto stride_C_BC = stride_C_BC_;

    auto unit_A_AC = unit_dim(stride_A_AC);
    auto unit_C_AC = unit_dim(stride_C_AC);
    auto unit_B_BC = unit_dim(stride_B_BC);
    auto unit_C_BC = unit_dim(stride_C_BC);
    auto unit_A_AB = unit_dim(stride_A_AB);
    auto unit_B_AB = unit_dim(stride_B_AB);

    TBLIS_ASSERT(unit_C_AC == 0 || unit_C_AC == -1);
    TBLIS_ASSERT(unit_C_BC == 0 || unit_C_BC == -1);
    TBLIS_ASSERT(unit_A_AB == 0 || unit_B_AB == 0 ||
                 (unit_A_AB == -1 && unit_B_AB == -1));

    bool pack_M_3d = unit_A_AC > 0;
    bool pack_N_3d = unit_B_BC > 0;
    bool pack_K_3d = unit_A_AB > 0 || unit_B_AB > 0;

    if (pack_M_3d)
    {
        std::rotate(     len_AC.begin()+1,      len_AC.begin()+unit_A_AC,      len_AC.end());
        std::rotate(stride_A_AC.begin()+1, stride_A_AC.begin()+unit_A_AC, stride_A_AC.end());
        std::rotate(stride_C_AC.begin()+1, stride_C_AC.begin()+unit_A_AC, stride_C_AC.end());
    }

    if (pack_N_3d)
    {
        std::rotate(     len_BC.begin()+1,      len_BC.begin()+unit_B_BC,      len_BC.end());
        std::rotate(stride_B_BC.begin()+1, stride_B_BC.begin()+unit_B_BC, stride_B_BC.end());
        std::rotate(stride_C_BC.begin()+1, stride_C_BC.begin()+unit_B_BC, stride_C_BC.end());
    }

    if (pack_K_3d)
    {
        auto dim = std::max(unit_A_AB, unit_B_AB);
        std::rotate(     len_AB.begin()+1,      len_AB.begin()+dim,      len_AB.end());
        std::rotate(stride_A_AB.begin()+1, stride_A_AB.begin()+dim, stride_A_AB.end());
        std::rotate(stride_B_AB.begin()+1, stride_B_AB.begin()+dim, stride_B_AB.end());
    }

    tensor_matrix at(alpha, conj_A,
                     len_AC,
                     len_AB,
                     A,
                     stride_A_AC,
                     stride_A_AB,
                     pack_M_3d, pack_K_3d);

    tensor_matrix bt({1, type}, conj_B,
                     len_AB,
                     len_BC,
                     B,
                     stride_B_AB,
                     stride_B_BC,
                     pack_K_3d, pack_N_3d);

    tensor_matrix ct(beta, conj_C,
                     len_AC,
                     len_BC,
                     C,
                     stride_C_AC,
                     stride_C_BC,
                     pack_M_3d, pack_N_3d);

    GotoGEMM{}(comm, cfg, at, bt, ct);
}

static
void mult_blis(type_t type, const communicator& comm, const config& cfg,
               const len_vector& len_AB_,
               const len_vector& len_AC_,
               const len_vector& len_ABC_,
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

    len_vector len_AB(len_AB_.begin()+1, len_AB_.end());
    len_vector len_AC(len_AC_.begin()+1, len_AC_.end());
    stride_vector stride_A_AB(stride_A_AB_.begin()+1, stride_A_AB_.end());
    stride_vector stride_B_AB(stride_B_AB_.begin()+1, stride_B_AB_.end());
    stride_vector stride_A_AC(stride_A_AC_.begin()+1, stride_A_AC_.end());
    stride_vector stride_C_AC(stride_C_AC_.begin()+1, stride_C_AC_.end());

    auto m = len_AC[0];
    auto n = len_AB[0];
    auto rs_A = stride_A_AC[0];
    auto cs_A = stride_A_AB[0];
    auto inc_B = stride_B_AB[0];
    auto inc_C = stride_C_AC[0];

    for (auto& s : stride_A_AC) s *= ts;
    for (auto& s : stride_A_AB) s *= ts;
    for (auto& s : stride_B_AB) s *= ts;
    for (auto& s : stride_C_AC) s *= ts;

    len_type l = stl_ext::prod(len_ABC_);
    len_type m2 = stl_ext::prod(len_AC)/m;
    len_type n2 = stl_ext::prod(len_AB)/n;

    if (comm.master()) flops += 2*m*m2*n*n2*l;

    unsigned nt_l, nt_m;
    std::tie(nt_l, nt_m) = partition_2x2(comm.num_threads(), l*m2, l*m2, m, m);

    auto subcomm = comm.gang(TCI_EVENLY, nt_l);

    subcomm.distribute_over_gangs(l*m2,
    [&](len_type l_min, len_type l_max)
    {
        len_vector len_ABC = len_ABC_;
        stride_vector stride_A_ABC = stride_A_ABC_;
        stride_vector stride_B_ABC = stride_B_ABC_;
        stride_vector stride_C_ABC = stride_C_ABC_;

        for (auto& s : stride_A_ABC) s *= ts;
        for (auto& s : stride_B_ABC) s *= ts;
        for (auto& s : stride_C_ABC) s *= ts;

        len_ABC.insert(len_ABC.end(), len_AC.begin(), len_AC.end());
        stride_A_ABC.insert(stride_A_ABC.end(), stride_A_AC.begin(), stride_A_AC.end());
        stride_B_ABC.insert(stride_B_ABC.end(), len_AC.size(), 0);
        stride_C_ABC.insert(stride_C_ABC.end(), stride_C_AC.begin(), stride_C_AC.end());

        viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
        viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        iter_ABC.position(l_min, A1, B1, C1);

        for (len_type l = l_min;l < l_max;l++)
        {
            iter_ABC.next(A1, B1, C1);

            auto beta1 = beta;
            auto conj_C1 = conj_C;

            while (iter_AB.next(A1, B1))
            {
                if (rs_A <= cs_A)
                {
                    const len_type NF = cfg.addf_nf.def(type);

                    subcomm.distribute_over_threads(m,
                    [&](len_type m_min, len_type m_max)
                    {
                        const void* As[16];
                        auto beta2 = beta1;
                        auto conj_C2 = conj_C1;

                        for (len_type j = 0;j < n;j += NF)
                        {
                            for (len_type k = 0;k < NF;k++)
                                As[k] = A1 + m_min*rs_A*ts + (j+k)*cs_A*ts;

                            cfg.addf_sum_ukr.call(type, m_max-m_min, std::min(NF, n-j),
                                                  &alpha, conj_A,  As,                  rs_A,
                                                          conj_B,  B1 +     j*inc_B*ts, inc_B,
                                                  &beta2, conj_C2, C1 + m_min*inc_C*ts, inc_C);

                            beta2 = 1.0;
                            conj_C2 = false;
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
                                              &alpha, conj_A,  A1 + i* rs_A*ts, rs_A, cs_A,
                                                      conj_B,  B1,              inc_B,
                                              &beta1, conj_C1, C1 + i*inc_C*ts, inc_C);
                        }
                    });
                }

                subcomm.barrier();

                beta1 = 1.0;
                conj_C1 = false;
            }
        }
    });
}

static
void mult_blis(type_t type, const communicator& comm, const config& cfg,
               const len_vector& len_AC_,
               const len_vector& len_BC_,
               const len_vector& len_ABC_,
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

    len_vector len_AC(len_AC_.begin()+1, len_AC_.end());
    len_vector len_BC(len_BC_.begin()+1, len_BC_.end());
    stride_vector stride_A_AC(stride_A_AC_.begin()+1, stride_A_AC_.end());
    stride_vector stride_C_AC(stride_C_AC_.begin()+1, stride_C_AC_.end());
    stride_vector stride_B_BC(stride_B_BC_.begin()+1, stride_B_BC_.end());
    stride_vector stride_C_BC(stride_C_BC_.begin()+1, stride_C_BC_.end());

    auto m = len_AC[0];
    auto n = len_BC[0];
    auto rs_C = stride_C_AC[0];
    auto cs_C = stride_C_BC[0];
    auto inc_A = stride_A_AC[0];
    auto inc_B = stride_B_BC[0];

    for (auto& s : stride_A_AC) s *= ts;
    for (auto& s : stride_B_BC) s *= ts;
    for (auto& s : stride_C_AC) s *= ts;
    for (auto& s : stride_C_BC) s *= ts;

    len_type l = stl_ext::prod(len_ABC_);
    len_type m2 = stl_ext::prod(len_AC)/m;
    len_type n2 = stl_ext::prod(len_BC)/n;

    if (comm.master()) flops += 2*m*m2*n*n2*l;

    unsigned nt_l, nt_m;
    std::tie(nt_l, nt_m) = partition_2x2(comm.num_threads(), l*m2*n2, l*m2*n2, m*n, m*n);

    auto subcomm = comm.gang(TCI_EVENLY, nt_l);

    subcomm.distribute_over_gangs(l*m2*n2,
    [&](len_type l_min, len_type l_max)
    {
        len_vector len_ABC = len_ABC_;
        stride_vector stride_A_ABC = stride_A_ABC_;
        stride_vector stride_B_ABC = stride_B_ABC_;
        stride_vector stride_C_ABC = stride_C_ABC_;

        for (auto& s : stride_A_ABC) s *= ts;
        for (auto& s : stride_B_ABC) s *= ts;
        for (auto& s : stride_C_ABC) s *= ts;

        if (rs_C <= cs_C)
        {
            len_ABC.insert(len_ABC.end(), len_AC.begin(), len_AC.end());
            stride_A_ABC.insert(stride_A_ABC.end(), stride_A_AC.begin(), stride_A_AC.end());
            stride_B_ABC.insert(stride_B_ABC.end(), len_AC.size(), 0);
            stride_C_ABC.insert(stride_C_ABC.end(), stride_C_AC.begin(), stride_C_AC.end());

            len_ABC.insert(len_ABC.end(), len_BC.begin(), len_BC.end());
            stride_A_ABC.insert(stride_A_ABC.end(), len_BC.size(), 0);
            stride_B_ABC.insert(stride_B_ABC.end(), stride_B_BC.begin(), stride_B_BC.end());
            stride_C_ABC.insert(stride_C_ABC.end(), stride_C_BC.begin(), stride_C_BC.end());
        }
        else
        {
            len_ABC.insert(len_ABC.end(), len_BC.begin(), len_BC.end());
            stride_A_ABC.insert(stride_A_ABC.end(), len_BC.size(), 0);
            stride_B_ABC.insert(stride_B_ABC.end(), stride_B_BC.begin(), stride_B_BC.end());
            stride_C_ABC.insert(stride_C_ABC.end(), stride_C_BC.begin(), stride_C_BC.end());

            len_ABC.insert(len_ABC.end(), len_AC.begin(), len_AC.end());
            stride_A_ABC.insert(stride_A_ABC.end(), stride_A_AC.begin(), stride_A_AC.end());
            stride_B_ABC.insert(stride_B_ABC.end(), len_AC.size(), 0);
            stride_C_ABC.insert(stride_C_ABC.end(), stride_C_AC.begin(), stride_C_AC.end());
        }

        viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

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

static
void mult_blis(type_t type, const communicator& comm, const config& cfg,
               const len_vector& len_AB_,
               const len_vector& len_AC_,
               const len_vector& len_BC_,
               const len_vector& len_ABC_,
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
    const len_type ts = type_size[type];

    auto len_AB = len_AB_;
    auto len_AC = len_AC_;
    auto len_BC = len_BC_;
    auto len_ABC = len_ABC_;
    auto stride_A_AB = stride_A_AB_;
    auto stride_B_AB = stride_B_AB_;
    auto stride_A_AC = stride_A_AC_;
    auto stride_C_AC = stride_C_AC_;
    auto stride_B_BC = stride_B_BC_;
    auto stride_C_BC = stride_C_BC_;
    auto stride_A_ABC = stride_A_ABC_;
    auto stride_B_ABC = stride_B_ABC_;
    auto stride_C_ABC = stride_C_ABC_;

    auto unit_A_AC = unit_dim(stride_A_AC);
    auto unit_C_AC = unit_dim(stride_C_AC);
    auto unit_B_BC = unit_dim(stride_B_BC);
    auto unit_C_BC = unit_dim(stride_C_BC);
    auto unit_A_AB = unit_dim(stride_A_AB);
    auto unit_B_AB = unit_dim(stride_B_AB);

    TBLIS_ASSERT(unit_C_AC == 0 || unit_C_AC == -1);
    TBLIS_ASSERT(unit_C_BC == 0 || unit_C_BC == -1);
    TBLIS_ASSERT(unit_A_AB == 0 || unit_B_AB == 0 ||
                 (unit_A_AB == -1 && unit_B_AB == -1));

    bool pack_M_3d = unit_A_AC > 0;
    bool pack_N_3d = unit_B_BC > 0;
    bool pack_K_3d = unit_A_AB > 0 || unit_B_AB > 0;

    if (pack_M_3d)
    {
        std::rotate(     len_AC.begin()+1,      len_AC.begin()+unit_A_AC,      len_AC.end());
        std::rotate(stride_A_AC.begin()+1, stride_A_AC.begin()+unit_A_AC, stride_A_AC.end());
        std::rotate(stride_C_AC.begin()+1, stride_C_AC.begin()+unit_A_AC, stride_C_AC.end());
    }

    if (pack_N_3d)
    {
        std::rotate(     len_BC.begin()+1,      len_BC.begin()+unit_B_BC,      len_BC.end());
        std::rotate(stride_B_BC.begin()+1, stride_B_BC.begin()+unit_B_BC, stride_B_BC.end());
        std::rotate(stride_C_BC.begin()+1, stride_C_BC.begin()+unit_B_BC, stride_C_BC.end());
    }

    if (pack_K_3d)
    {
        auto dim = std::max(unit_A_AB, unit_B_AB);
        std::rotate(     len_AB.begin()+1,      len_AB.begin()+dim,      len_AB.end());
        std::rotate(stride_A_AB.begin()+1, stride_A_AB.begin()+dim, stride_A_AB.end());
        std::rotate(stride_B_AB.begin()+1, stride_B_AB.begin()+dim, stride_B_AB.end());
    }

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

    for (auto& s : stride_A_ABC) s *= ts;
    for (auto& s : stride_B_ABC) s *= ts;
    for (auto& s : stride_C_ABC) s *= ts;

    subcomm.distribute_over_gangs(l,
    [&](len_type l_min, len_type l_max)
    {
        viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        iter_ABC.position(l_min, A1, B1, C1);

        for (len_type l = l_min;l < l_max;l++)
        {
            iter_ABC.next(A1, B1, C1);

            tensor_matrix at(alpha, conj_A, len_AC, len_AB, A1,
                             stride_A_AC, stride_A_AB, pack_M_3d, pack_K_3d);

            tensor_matrix bt(  one, conj_B, len_AB, len_BC, B1,
                             stride_B_AB, stride_B_BC, pack_K_3d, pack_N_3d);

            tensor_matrix ct( beta, conj_C, len_AC, len_BC, C1,
                             stride_C_AC, stride_C_BC, pack_M_3d, pack_N_3d);

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
    marray<T> ar, br, cr;

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

    auto len_a = len_AC; len_a.insert(len_a.end(), len_AB.begin(), len_AB.end());
    auto len_b = len_AB; len_b.insert(len_a.end(), len_BC.begin(), len_BC.end());
    auto len_c = len_AC; len_c.insert(len_a.end(), len_BC.begin(), len_BC.end());
    auto stride_a = stride_A_AC; stride_a.insert(len_a.end(), stride_A_AB.begin(), stride_A_AB.end());
    auto stride_b = stride_B_AB; stride_b.insert(len_a.end(), stride_B_BC.begin(), stride_B_BC.end());
    auto stride_c = stride_C_AC; stride_c.insert(len_a.end(), stride_C_BC.begin(), stride_C_BC.end());

    if (comm.master())
    {
        ar.reset(len_a);
        br.reset(len_b);
        cr.reset(len_c);
    }

    comm.broadcast(
    [&](marray<T>& ar, marray<T>& br, marray<T>& cr)
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
                alpha, conj_A, reinterpret_cast<char*>(       A1), {},     stride_a,
                 T(0),  false, reinterpret_cast<char*>(ar.data()), {}, ar.strides());

            add(type_tag<T>::value, comm, cfg, {}, {}, br.lengths(),
                T(1), conj_B, reinterpret_cast<char*>(       B1), {},     stride_b,
                T(0),  false, reinterpret_cast<char*>(br.data()), {}, br.strides());

            normal_matrix at(T(1), false, am.length(0), am.length(1),
                             reinterpret_cast<char*>(am.data()), am.stride(0), am.stride(1));

            normal_matrix bt(T(1), false, bm.length(0), bm.length(1),
                             reinterpret_cast<char*>(bm.data()), bm.stride(0), bm.stride(1));

            normal_matrix ct(T(0), false, cm.length(0), cm.length(1),
                             reinterpret_cast<char*>(cm.data()), cm.stride(0), cm.stride(1));

            GotoGEMM{}(comm, cfg, at, bt, ct);

            add(type_tag<T>::value, comm, cfg, {}, {}, cr.lengths(),
                T(1),  false, reinterpret_cast<char*>(cr.data()), {}, cr.strides(),
                beta, conj_C, reinterpret_cast<char*>(       C1), {},     stride_c);
        }
    },
    ar, br, cr);
}

static
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

static
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
    for (auto i : range(1,stride_A_ABC.size())) stride_A1.push_back(stride_A_ABC[i]*ts);
    for (auto i : range(1,stride_B_ABC.size())) stride_B1.push_back(stride_B_ABC[i]*ts);
    for (auto i : range(1,stride_C_ABC.size())) stride_C1.push_back(stride_C_ABC[i]*ts);

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
        auto len_C = len_ABC;
        auto stride_C = stride_C_ABC;

        len_C.insert(len_C.end(), len_AC.begin(), len_AC.end());
        len_C.insert(len_C.end(), len_BC.begin(), len_BC.end());
        stride_C.insert(stride_C.end(), stride_C_AC.begin(), stride_C_AC.end());
        stride_C.insert(stride_C.end(), stride_C_BC.begin(), stride_C_BC.end());

        if (beta.is_zero())
        {
            set(type, comm, cfg, len_C, beta, C, stride_C);
        }
        else if (!beta.is_one() || (beta.is_complex() && conj_C))
        {
            scale(type, comm, cfg, len_C, beta, conj_C, C, stride_C);
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
            case FLOAT:
                mult_blas(comm, cfg,
                          len_AB, len_AC, len_BC, len_ABC,
                          alpha.get<float>(), conj_A, reinterpret_cast<float*>(A),
                                              stride_A_AB, stride_A_AC, stride_A_ABC,
                                              conj_B, reinterpret_cast<float*>(B),
                                              stride_B_AB, stride_B_BC, stride_B_ABC,
                           beta.get<float>(), conj_C, reinterpret_cast<float*>(C),
                                              stride_C_AC, stride_C_BC, stride_C_ABC);
                break;
            case DOUBLE:
                mult_blas(comm, cfg,
                          len_AB, len_AC, len_BC, len_ABC,
                          alpha.get<double>(), conj_A, reinterpret_cast<double*>(A),
                                               stride_A_AB, stride_A_AC, stride_A_ABC,
                                               conj_B, reinterpret_cast<double*>(B),
                                               stride_B_AB, stride_B_BC, stride_B_ABC,
                           beta.get<double>(), conj_C, reinterpret_cast<double*>(C),
                                               stride_C_AC, stride_C_BC, stride_C_ABC);
                break;
            case SCOMPLEX:
                mult_blas(comm, cfg,
                          len_AB, len_AC, len_BC, len_ABC,
                          alpha.get<scomplex>(), conj_A, reinterpret_cast<scomplex*>(A),
                                                 stride_A_AB, stride_A_AC, stride_A_ABC,
                                                 conj_B, reinterpret_cast<scomplex*>(B),
                                                 stride_B_AB, stride_B_BC, stride_B_ABC,
                           beta.get<scomplex>(), conj_C, reinterpret_cast<scomplex*>(C),
                                                 stride_C_AC, stride_C_BC, stride_C_ABC);
                break;
            case DCOMPLEX:
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
