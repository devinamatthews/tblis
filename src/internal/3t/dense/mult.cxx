#include "mult.hpp"

#include "util/gemm_thread.hpp"
#include "util/tensor.hpp"

#include "nodes/gemm.hpp"

#include "internal/1t/dense/add.hpp"
#include "internal/1t/dense/dot.hpp"
#include "internal/1t/dense/scale.hpp"
#include "internal/1t/dense/set.hpp"
#include "internal/1v/mult.hpp"
#include "internal/2m/mult.hpp"
#include "internal/3m/mult.hpp"

namespace tblis
{

MemoryPool BuffersForScatter(4096);

namespace internal
{

impl_t impl = BLIS_BASED;

template <typename T>
void mult_blis(const communicator& comm, const config& cfg,
               const len_vector& len_AB,
               const len_vector& len_AC,
               T alpha, bool conj_A, const T* A,
               const stride_vector& stride_A_AB,
               const stride_vector& stride_A_AC,
                        bool conj_B, const T* B,
               const stride_vector& stride_B_AB,
               T  beta, bool conj_C,       T* C,
               const stride_vector& stride_C_AC)
{
    auto reorder_AC = detail::sort_by_stride(stride_A_AC, stride_C_AC);
    auto reorder_AB = detail::sort_by_stride(stride_A_AB, stride_B_AB);

    unsigned unit_AC = 0;
    for (unsigned i : reorder_AC)
    {
        if (len_AC[i] == 1) continue;
        unit_AC = i;
        break;
    }

    unsigned unit_AB = 0;
    for (unsigned i : reorder_AB)
    {
        if (len_AB[i] == 1) continue;
        unit_AB = i;
        break;
    }

    auto m = len_AC[unit_AC];
    auto n = len_AB[unit_AB];
    auto rs_A = stride_A_AC[unit_AC];
    auto cs_A = stride_A_AB[unit_AB];
    auto inc_B = stride_B_AB[unit_AB];
    auto inc_C = stride_C_AC[unit_AC];

    stl_ext::erase(reorder_AC, unit_AC);
    stl_ext::erase(reorder_AB, unit_AB);

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
                mult(subcomm, cfg, m, n,
                     alpha, conj_A, A1, rs_A, cs_A,
                            conj_B, B1, inc_B,
                     local_beta, local_conj_C, C1, inc_C);

                local_beta = T(1);
                local_conj_C = false;
            }
        }
    });
}

template <typename T>
void mult_blis(const communicator& comm, const config& cfg,
               const len_vector& len_AC,
               const len_vector& len_BC,
               T alpha, bool conj_A, const T* A,
               const stride_vector& stride_A_AC,
                        bool conj_B, const T* B,
               const stride_vector& stride_B_BC,
               T  beta, bool conj_C,       T* C,
               const stride_vector& stride_C_AC,
               const stride_vector& stride_C_BC)
{
    auto reorder_AC = detail::sort_by_stride(stride_C_AC, stride_A_AC);
    auto reorder_BC = detail::sort_by_stride(stride_C_BC, stride_B_BC);

    unsigned unit_AC = 0;
    for (unsigned i : reorder_AC)
    {
        if (len_AC[i] == 1) continue;
        unit_AC = i;
        break;
    }

    unsigned unit_BC = 0;
    for (unsigned i : reorder_BC)
    {
        if (len_BC[i] == 1) continue;
        unit_BC = i;
        break;
    }

    auto m = len_AC[unit_AC];
    auto n = len_BC[unit_BC];
    auto rs_C = stride_C_AC[unit_AC];
    auto cs_C = stride_C_BC[unit_BC];
    auto inc_A = stride_A_AC[unit_AC];
    auto inc_B = stride_B_BC[unit_BC];

    stl_ext::erase(reorder_AC, unit_AC);
    stl_ext::erase(reorder_BC, unit_BC);

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

            mult(subcomm, cfg, m, n,
                 alpha, conj_A, A1, inc_A,
                        conj_B, B1, inc_B,
                  beta, conj_C, C1, rs_C, cs_C);
        }
    });
}

template <typename T>
void mult_blis(const communicator& comm, const config& cfg,
                   const len_vector& len_AB,
                   const len_vector& len_AC,
                   const len_vector& len_BC,
                   T alpha, bool conj_A, const T* A,
                   const stride_vector& stride_A_AB,
                   const stride_vector& stride_A_AC,
                            bool conj_B, const T* B,
                   const stride_vector& stride_B_AB,
                   const stride_vector& stride_B_BC,
                   T  beta, bool conj_C,       T* C,
                   const stride_vector& stride_C_AC,
                   const stride_vector& stride_C_BC)
{
    //TODO
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);

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

    tensor_matrix<T> at(stl_ext::permuted(len_AC, reorder_AC),
                        stl_ext::permuted(len_AB, reorder_AB),
                        const_cast<T*>(A),
                        stl_ext::permuted(stride_A_AC, reorder_AC),
                        stl_ext::permuted(stride_A_AB, reorder_AB),
                        pack_M_3d, pack_K_3d);

    tensor_matrix<T> bt(stl_ext::permuted(len_AB, reorder_AB),
                        stl_ext::permuted(len_BC, reorder_BC),
                        const_cast<T*>(B),
                        stl_ext::permuted(stride_B_AB, reorder_AB),
                        stl_ext::permuted(stride_B_BC, reorder_BC),
                        pack_K_3d, pack_N_3d);

    tensor_matrix<T> ct(stl_ext::permuted(len_AC, reorder_AC),
                        stl_ext::permuted(len_BC, reorder_BC),
                        C,
                        stl_ext::permuted(stride_C_AC, reorder_AC),
                        stl_ext::permuted(stride_C_BC, reorder_BC),
                        pack_M_3d, pack_N_3d);

    TensorGEMM{}(comm, cfg, alpha, at, bt, beta, ct);
}

template <typename T>
void mult_blis(const communicator& comm, const config& cfg,
               const len_vector& len_AB,
               const len_vector& len_AC,
               const len_vector& len_ABC,
               T alpha, bool conj_A, const T* A,
               const stride_vector& stride_A_AB,
               const stride_vector& stride_A_AC,
               const stride_vector& stride_A_ABC,
                        bool conj_B, const T* B,
               const stride_vector& stride_B_AB,
               const stride_vector& stride_B_ABC,
               T  beta, bool conj_C,       T* C,
               const stride_vector& stride_C_AC,
               const stride_vector& stride_C_ABC)
{
    auto reorder_AC = detail::sort_by_stride(stride_A_AC, stride_C_AC);
    auto reorder_AB = detail::sort_by_stride(stride_A_AB, stride_B_AB);
    auto reorder_ABC = detail::sort_by_stride(stride_C_ABC, stride_A_ABC, stride_B_ABC);

    unsigned unit_AC = 0;
    for (unsigned i : reorder_AC)
    {
        if (len_AC[i] == 1) continue;
        unit_AC = i;
        break;
    }

    unsigned unit_AB = 0;
    for (unsigned i : reorder_AB)
    {
        if (len_AB[i] == 1) continue;
        unit_AB = i;
        break;
    }

    auto m = len_AC[unit_AC];
    auto n = len_AB[unit_AB];
    auto rs_A = stride_A_AC[unit_AC];
    auto cs_A = stride_A_AB[unit_AB];
    auto inc_B = stride_B_AB[unit_AB];
    auto inc_C = stride_C_AC[unit_AC];

    stl_ext::erase(reorder_AC, unit_AC);
    stl_ext::erase(reorder_AB, unit_AB);

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
                mult(subcomm, cfg, m, n,
                     alpha, conj_A, A1, rs_A, cs_A,
                            conj_B, B1, inc_B,
                     local_beta, local_conj_C, C1, inc_C);

                local_beta = T(1);
                local_conj_C = false;
            }
        }
    });
}

template <typename T>
void mult_blis(const communicator& comm, const config& cfg,
               const len_vector& len_AC,
               const len_vector& len_BC,
               const len_vector& len_ABC,
               T alpha, bool conj_A, const T* A,
               const stride_vector& stride_A_AC,
               const stride_vector& stride_A_ABC,
                        bool conj_B, const T* B,
               const stride_vector& stride_B_BC,
               const stride_vector& stride_B_ABC,
               T  beta, bool conj_C,       T* C,
               const stride_vector& stride_C_AC,
               const stride_vector& stride_C_BC,
               const stride_vector& stride_C_ABC)
{
    auto reorder_AC = detail::sort_by_stride(stride_C_AC, stride_A_AC);
    auto reorder_BC = detail::sort_by_stride(stride_C_BC, stride_B_BC);
    auto reorder_ABC = detail::sort_by_stride(stride_C_ABC, stride_A_ABC, stride_B_ABC);

    unsigned unit_AC = 0;
    for (unsigned i : reorder_AC)
    {
        if (len_AC[i] == 1) continue;
        unit_AC = i;
        break;
    }

    unsigned unit_BC = 0;
    for (unsigned i : reorder_BC)
    {
        if (len_BC[i] == 1) continue;
        unit_BC = i;
        break;
    }

    auto m = len_AC[unit_AC];
    auto n = len_BC[unit_BC];
    auto rs_C = stride_C_AC[unit_AC];
    auto cs_C = stride_C_BC[unit_BC];
    auto inc_A = stride_A_AC[unit_AC];
    auto inc_B = stride_B_BC[unit_BC];

    stl_ext::erase(reorder_AC, unit_AC);
    stl_ext::erase(reorder_BC, unit_BC);

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

            mult(subcomm, cfg, m, n,
                 alpha, conj_A, A1, inc_A,
                        conj_B, B1, inc_B,
                  beta, conj_C, C1, rs_C, cs_C);
        }
    });
}

template <typename T>
void mult_blis(const communicator& comm, const config& cfg,
               const len_vector& len_AB,
               const len_vector& len_AC,
               const len_vector& len_BC,
               const len_vector& len_ABC,
               T alpha, bool conj_A, const T* A,
               const stride_vector& stride_A_AB,
               const stride_vector& stride_A_AC,
               const stride_vector& stride_A_ABC,
                        bool conj_B, const T* B,
               const stride_vector& stride_B_AB,
               const stride_vector& stride_B_BC,
               const stride_vector& stride_B_ABC,
               T  beta, bool conj_C,       T* C,
               const stride_vector& stride_C_AC,
               const stride_vector& stride_C_BC,
               const stride_vector& stride_C_ABC)
{
    //TODO
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);

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

    len_type m = stl_ext::prod(len_AC);
    len_type n = stl_ext::prod(len_BC);
    len_type k = stl_ext::prod(len_AB);
    len_type l = stl_ext::prod(len_ABC);

    if (comm.master()) flops += 2*m*n*k*l;

    unsigned nt_l, nt_mn;
    std::tie(nt_l, nt_mn) =
        partition_2x2(comm.num_threads(), l, l, m*n, m*n);

    auto subcomm = comm.gang(TCI_EVENLY, nt_l);

    subcomm.distribute_over_gangs(l,
    [&](len_type l_min, len_type l_max)
    {
        tensor_matrix<T> at(stl_ext::permuted(len_AC, reorder_AC),
                            stl_ext::permuted(len_AB, reorder_AB),
                            nullptr,
                            stl_ext::permuted(stride_A_AC, reorder_AC),
                            stl_ext::permuted(stride_A_AB, reorder_AB),
                            pack_M_3d, pack_K_3d);

        tensor_matrix<T> bt(stl_ext::permuted(len_AB, reorder_AB),
                            stl_ext::permuted(len_BC, reorder_BC),
                            nullptr,
                            stl_ext::permuted(stride_B_AB, reorder_AB),
                            stl_ext::permuted(stride_B_BC, reorder_BC),
                            pack_K_3d, pack_N_3d);

        tensor_matrix<T> ct(stl_ext::permuted(len_AC, reorder_AC),
                            stl_ext::permuted(len_BC, reorder_BC),
                            nullptr,
                            stl_ext::permuted(stride_C_AC, reorder_AC),
                            stl_ext::permuted(stride_C_BC, reorder_BC),
                            pack_M_3d, pack_N_3d);

        viterator<3> iter_ABC(stl_ext::permuted(len_ABC, reorder_ABC),
                              stl_ext::permuted(stride_A_ABC, reorder_ABC),
                              stl_ext::permuted(stride_B_ABC, reorder_ABC),
                              stl_ext::permuted(stride_C_ABC, reorder_ABC));

        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        iter_ABC.position(l_min, A1, B1, C1);

        for (len_type l = l_min;l < l_max;l++)
        {
            iter_ABC.next(A1, B1, C1);

            at.data(const_cast<T*>(A1));
            bt.data(const_cast<T*>(B1));
            ct.data(C1);

            TensorGEMM{}(subcomm, cfg, alpha, at, bt, beta, ct);
        }
    });
}

template <typename T>
void mult_blas(const communicator& comm, const config& cfg,
               const len_vector& len_AB_,
               const len_vector& len_AC_,
               const len_vector& len_BC_,
               const len_vector& len_ABC_,
               T alpha, bool conj_A, const T* A,
               const stride_vector& stride_A_AB_,
               const stride_vector& stride_A_AC_,
               const stride_vector& stride_A_ABC_,
                        bool conj_B, const T* B,
               const stride_vector& stride_B_AB_,
               const stride_vector& stride_B_BC_,
               const stride_vector& stride_B_ABC_,
               T  beta, bool conj_C,       T* C,
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
        matrix_view<T> am, bm, cm;
        matricize<T>(ar, am, static_cast<unsigned>(len_AC.size()));
        matricize<T>(br, bm, static_cast<unsigned>(len_AB.size()));
        matricize<T>(cr, cm, static_cast<unsigned>(len_AC.size()));

        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        viterator<3> it(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

        while (it.next(A1, B1, C1))
        {
            add(comm, cfg, {}, {}, ar.lengths(),
                T(1), conj_A,        A1, {}, stride_A_AC+stride_A_AB,
                T(0),  false, ar.data(), {},            ar.strides());

            add(comm, cfg, {}, {}, br.lengths(),
                T(1), conj_B,        B1, {}, stride_B_AB+stride_B_BC,
                T(0),  false, br.data(), {},            br.strides());

            mult(comm, cfg, cm.length(0), cm.length(1), am.length(1),
                 alpha, false, am.data(), am.stride(0), am.stride(1),
                        false, bm.data(), bm.stride(0), bm.stride(1),
                  T(0), false, cm.data(), cm.stride(0), cm.stride(1));

            add(comm, cfg, {}, {}, cr.lengths(),
                T(1),  false, cr.data(), {},             cr.strides(),
                beta, conj_C,        C1, {}, stride_C_AC+stride_C_BC);
        }
    },
    ar, br, cr);
}

template <typename T>
void mult_ref(const communicator& comm, const config& cfg,
              const len_vector& len_AB,
              const len_vector& len_AC,
              const len_vector& len_BC,
              const len_vector& len_ABC,
              T alpha, bool conj_A, const T* A,
              const stride_vector& stride_A_AB,
              const stride_vector& stride_A_AC,
              const stride_vector& stride_A_ABC,
                       bool conj_B, const T* B,
              const stride_vector& stride_B_AB,
              const stride_vector& stride_B_BC,
              const stride_vector& stride_B_ABC,
              T  beta, bool conj_C,       T* C,
              const stride_vector& stride_C_AC,
              const stride_vector& stride_C_BC,
              const stride_vector& stride_C_ABC)
{
    (void)cfg;

    len_type n = stl_ext::prod(len_ABC);

    comm.distribute_over_threads(n,
    [&](len_type n_min, len_type n_max)
    {
        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
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
                    T temp = T();

                    TBLIS_SPECIAL_CASE(conj_A,
                    TBLIS_SPECIAL_CASE(conj_B,
                    while (iter_AB.next(A1, B1))
                    {
                        temp += (conj_A ? conj(*A1) : *A1)*
                                (conj_B ? conj(*B1) : *B1);
                    }
                    ))
                    temp *= alpha;

                    if (beta == T(0))
                    {
                        *C1 = temp;
                    }
                    else
                    {
                        *C1 = temp + beta*(conj_C ? conj(*C1) : *C1);
                    }
                }
            }
        }
    });
}

template <typename T>
void mult_vec(const communicator& comm, const config& cfg,
              const len_vector& len_ABC,
              T alpha, bool conj_A, const T* A,
              const stride_vector& stride_A_ABC,
                       bool conj_B, const T* B,
              const stride_vector& stride_B_ABC,
              T  beta, bool conj_C,       T* C,
              const stride_vector& stride_C_ABC)
{
    (void)cfg;

    len_type n = stl_ext::prod(len_ABC);

    comm.distribute_over_threads(n,
    [&](len_type n_min, len_type n_max)
    {
        auto A1 = A;
        auto B1 = B;
        auto C1 = C;

        viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);
        iter_ABC.position(n_min, A1, B1, C1);

        if (beta == T(0))
        {
            TBLIS_SPECIAL_CASE(conj_A,
            TBLIS_SPECIAL_CASE(conj_B,
            for (len_type i = n_min;i < n_max;i++)
            {
                iter_ABC.next(A1, B1, C1);

                *C1 = alpha*(conj_A ? conj(*A1) : *A1)*
                            (conj_B ? conj(*B1) : *B1);
            }
            ))
        }
        else
        {
            TBLIS_SPECIAL_CASE(conj_A,
            TBLIS_SPECIAL_CASE(conj_B,
            for (len_type i = n_min;i < n_max;i++)
            {
                iter_ABC.next(A1, B1, C1);

                *C1 = alpha*(conj_A ? conj(*A1) : *A1)*
                            (conj_B ? conj(*B1) : *B1) +
                       beta*(conj_C ? conj(*C1) : *C1);
            }
            ))
        }
    });
}

template <typename T>
void mult(const communicator& comm, const config& cfg,
          const len_vector& len_AB,
          const len_vector& len_AC,
          const len_vector& len_BC,
          const len_vector& len_ABC,
          T alpha, bool conj_A, const T* A,
          const stride_vector& stride_A_AB,
          const stride_vector& stride_A_AC,
          const stride_vector& stride_A_ABC,
                   bool conj_B, const T* B,
          const stride_vector& stride_B_AB,
          const stride_vector& stride_B_BC,
          const stride_vector& stride_B_ABC,
          T  beta, bool conj_C,       T* C,
          const stride_vector& stride_C_AC,
          const stride_vector& stride_C_BC,
          const stride_vector& stride_C_ABC)
{
    auto n_AB = stl_ext::prod(len_AB);
    auto n_AC = stl_ext::prod(len_AC);
    auto n_BC = stl_ext::prod(len_BC);
    auto n_ABC = stl_ext::prod(len_ABC);

    if (n_AC == 0 || n_BC == 0 || n_ABC == 0) return;

    if (n_AB == 0)
    {
        if (beta == T(0))
        {
            set(comm, cfg, len_AC+len_BC+len_ABC, beta, C,
                stride_C_AC+stride_C_BC+stride_C_ABC);
        }
        else if (beta != T(1) || (is_complex<T>::value && conj_C))
        {
            scale(comm, cfg, len_AC+len_BC+len_ABC, beta, conj_C, C,
                  stride_C_AC+stride_C_BC+stride_C_ABC);
        }

        return;
    }

    if (impl == REFERENCE)
    {
        mult_ref(comm, cfg,
                 len_AB, len_AC, len_BC, len_ABC,
                 alpha, conj_A, A, stride_A_AB, stride_A_AC, stride_A_ABC,
                        conj_B, B, stride_B_AB, stride_B_BC, stride_B_ABC,
                  beta, conj_C, C, stride_C_AC, stride_C_BC, stride_C_ABC);
        comm.barrier();
        return;
    }
    else if (impl == BLAS_BASED)
    {
        mult_blas(comm, cfg,
                  len_AB, len_AC, len_BC, len_ABC,
                  alpha, conj_A, A, stride_A_AB, stride_A_AC, stride_A_ABC,
                         conj_B, B, stride_B_AB, stride_B_BC, stride_B_ABC,
                   beta, conj_C, C, stride_C_AC, stride_C_BC, stride_C_ABC);
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

    T sum;
    viterator<3> iter_ABC(len_ABC, stride_A_ABC, stride_B_ABC, stride_C_ABC);

    switch (groups)
    {
        case HAS_NONE:
        {
            if (comm.master())
            {
                if (beta == T(0))
                {
                    *C = alpha*(conj_A ? conj(*A) : *A)*
                               (conj_B ? conj(*B) : *B);
                }
                else
                {
                    *C = alpha*(conj_A ? conj(*A) : *A)*
                               (conj_B ? conj(*B) : *B) +
                          beta*(conj_C ? conj(*C) : *C);
                }
            }
        }
        break;
        case HAS_AB:
        {
            dot(comm, cfg, len_AB, conj_A, A, stride_A_AB,
                                   conj_B, B, stride_B_AB, sum);

            if (comm.master())
            {
                if (beta == T(0))
                {
                    *C = alpha*sum;
                }
                else
                {
                    *C = alpha*sum + beta*(conj_C ? conj(*C) : *C);
                }
            }
        }
        break;
        case HAS_AC:
        {
            add(comm, cfg, {}, {}, len_AC, alpha*(conj_B ? conj(*B) : *B),
                conj_A, A, {}, stride_A_AC, beta, conj_C, C, {}, stride_C_AC);
        }
        break;
        case HAS_BC:
        {
            add(comm, cfg, {}, {}, len_BC, alpha*(conj_A ? conj(*A) : *A),
                conj_B, B, {}, stride_B_BC, beta, conj_C, C, {}, stride_C_BC);
        }
        break;
        case HAS_ABC:
        {
            mult_vec(comm, cfg, len_ABC,
                     alpha, conj_A, A, stride_A_ABC,
                            conj_B, B, stride_B_ABC,
                      beta, conj_C, C, stride_C_ABC);
        }
        break;
        case HAS_AC+HAS_BC:
        {
            mult_blis(comm, cfg, len_AC, len_BC,
                      alpha, conj_A, A, stride_A_AC,
                             conj_B, B, stride_B_BC,
                       beta, conj_C, C, stride_C_AC, stride_C_BC);
        }
        break;
        case HAS_AB+HAS_AC:
        {
            mult_blis(comm, cfg, len_AB, len_AC,
                      alpha, conj_A, A, stride_A_AB, stride_A_AC,
                             conj_B, B, stride_B_AB,
                       beta, conj_C, C, stride_C_AC);
        }
        break;
        case HAS_AB+HAS_BC:
        {
            mult_blis(comm, cfg, len_AB, len_BC,
                      alpha, conj_B, B, stride_B_AB, stride_B_BC,
                             conj_A, A, stride_A_AB,
                       beta, conj_C, C, stride_C_BC);
        }
        break;
        case HAS_AB+HAS_AC+HAS_BC:
        {
            mult_blis(comm, cfg, len_AB, len_AC, len_BC,
                      alpha, conj_A, A, stride_A_AB, stride_A_AC,
                             conj_B, B, stride_B_AB, stride_B_BC,
                       beta, conj_C, C, stride_C_AC, stride_C_BC);
        }
        break;
        case HAS_AB+HAS_ABC:
        {
            while (iter_ABC.next(A, B, C))
            {
                dot(comm, cfg, len_AB, conj_A, A, stride_A_AB,
                                       conj_B, B, stride_B_AB, sum);

                if (comm.master())
                {
                    if (beta == T(0))
                    {
                        *C = alpha*sum;
                    }
                    else
                    {
                        *C = alpha*sum + beta*(conj_C ? conj(*C) : *C);
                    }
                }
            }
        }
        break;
        case HAS_AC+HAS_ABC:
        {
            while (iter_ABC.next(A, B, C))
            {
                add(comm, cfg, {}, {}, len_AC, alpha*(conj_B ? conj(*B) : *B),
                    conj_A, A, {}, stride_A_AC, beta, conj_C, C, {}, stride_C_AC);
            }
        }
        break;
        case HAS_BC+HAS_ABC:
        {
            while (iter_ABC.next(A, B, C))
            {
                add(comm, cfg, {}, {}, len_BC, alpha*(conj_A ? conj(*A) : *A),
                    conj_B, B, {}, stride_B_BC, beta, conj_C, C, {}, stride_C_BC);
            }
        }
        break;
        case HAS_AC+HAS_BC+HAS_ABC:
        {
            mult_blis(comm, cfg, len_AC, len_BC, len_ABC,
                      alpha, conj_A, A, stride_A_AC, stride_A_ABC,
                             conj_B, B, stride_B_BC, stride_B_ABC,
                       beta, conj_C, C, stride_C_AC, stride_C_BC, stride_C_ABC);
        }
        break;
        case HAS_AB+HAS_AC+HAS_ABC:
        {
            mult_blis(comm, cfg,
                      len_AB, len_AC, len_ABC,
                      alpha, conj_A, A, stride_A_AB, stride_A_AC, stride_A_ABC,
                             conj_B, B, stride_B_AB, stride_B_ABC,
                       beta, conj_C, C, stride_C_AC, stride_C_ABC);
        }
        break;
        case HAS_AB+HAS_BC+HAS_ABC:
        {
            mult_blis(comm, cfg,
                      len_AB, len_BC, len_ABC,
                      alpha, conj_B, B, stride_B_AB, stride_B_BC, stride_B_ABC,
                             conj_A, A, stride_A_AB, stride_A_ABC,
                       beta, conj_C, C, stride_C_BC, stride_C_ABC);
        }
        break;
        case HAS_AB+HAS_AC+HAS_BC+HAS_ABC:
        {
            mult_blis(comm, cfg,
                      len_AB, len_AC, len_BC, len_ABC,
                      alpha, conj_A, A, stride_A_AB, stride_A_AC, stride_A_ABC,
                             conj_B, B, stride_B_AB, stride_B_BC, stride_B_ABC,
                       beta, conj_C, C, stride_C_AC, stride_C_BC, stride_C_ABC);
        }
        break;
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, const config& cfg, \
                   const len_vector& len_AB, \
                   const len_vector& len_AC, \
                   const len_vector& len_BC, \
                   const len_vector& len_ABC, \
                   T alpha, bool conj_A, const T* A, \
                   const stride_vector& stride_A_AB, \
                   const stride_vector& stride_A_AC, \
                   const stride_vector& stride_A_ABC, \
                            bool conj_B, const T* B, \
                   const stride_vector& stride_B_AB, \
                   const stride_vector& stride_B_BC, \
                   const stride_vector& stride_B_ABC, \
                   T  beta, bool conj_C,       T* C, \
                   const stride_vector& stride_C_AC, \
                   const stride_vector& stride_C_BC, \
                   const stride_vector& stride_C_ABC);
#include "configs/foreach_type.h"

}
}
