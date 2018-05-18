#include "mult.hpp"

#include "util/gemm_thread.hpp"

#include "nodes/gemm.hpp"

namespace tblis
{

MemoryPool BuffersForA(4096);
MemoryPool BuffersForB(4096);

namespace internal
{

template <typename T>
void mult(const communicator& comm, const config& cfg_,
          len_type m, len_type n, len_type k,
          T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
                   bool conj_B, const T* B, stride_type rs_B, stride_type cs_B,
          T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C)
{
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);
    
    config cfg = cfg_;

    const bool row_major = cfg.gemm_row_major.value<T>();

    //if ((row_major ? rs_C : cs_C) == 1)
    if (n > m)
    {
        /*
         * Compute C^T = B^T * A^T instead
         */
        std::swap(m, n);
        std::swap(A, B);
        std::swap(rs_A, cs_B);
        std::swap(rs_B, cs_A);
        std::swap(rs_C, cs_C);
    }

    if ((row_major ? rs_C : cs_C) == 1)
    {
        cfg.gemm_row_major.value<T>() ^= true;
        cfg.gemm_flip_ukr.value<T>() ^= true;
        std::swap(cfg.gemm_mr, cfg.gemm_nr);
        std::swap(cfg.pack_nn_mr_ukr, cfg.pack_nn_nr_ukr);
        cfg.gemm_mc.iota<T>() = cfg.gemm_mr.def<T>();
        cfg.gemm_nc.iota<T>() = cfg.gemm_nr.def<T>();
        cfg.gemm_mc.def<T>() = round_up(cfg.gemm_mc.def<T>(),
                                        cfg.gemm_mr.def<T>());
        cfg.gemm_nc.def<T>() = round_up(cfg.gemm_nc.def<T>(),
                                        cfg.gemm_nr.def<T>());
        cfg.gemm_mc.max<T>() = round_up(cfg.gemm_mc.max<T>(),
                                        cfg.gemm_mr.def<T>());
        cfg.gemm_nc.max<T>() = round_up(cfg.gemm_nc.max<T>(),
                                        cfg.gemm_nr.def<T>());
    }

    matrix_view<T> Av({m, k}, const_cast<T*>(A), {rs_A, cs_A});
    matrix_view<T> Bv({k, n}, const_cast<T*>(B), {rs_B, cs_B});
    matrix_view<T> Cv({m, n},                C , {rs_C, cs_C});

    int nt = comm.num_threads();
    gemm_thread_config tc = make_gemm_thread_config<T>(cfg, nt, m, n, k);

    communicator comm_nc =    comm.gang(TCI_EVENLY, tc.jc_nt);
    communicator comm_kc = comm_nc.gang(TCI_EVENLY,        1);
    communicator comm_mc = comm_kc.gang(TCI_EVENLY, tc.ic_nt);
    communicator comm_nr = comm_mc.gang(TCI_EVENLY, tc.jr_nt);
    communicator comm_mr = comm_nr.gang(TCI_EVENLY, tc.ir_nt);

    GotoGEMM gemm;
    step<0>(gemm).subcomm = &comm_nc;
    step<1>(gemm).subcomm = &comm_kc;
    step<3>(gemm).subcomm = &comm_mc;
    step<5>(gemm).subcomm = &comm_nr;
    step<6>(gemm).subcomm = &comm_mr;

    gemm(comm, cfg, alpha, Av, Bv, beta, Cv);

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, const config& cfg, \
                   len_type m, len_type n, len_type k, \
                   T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A, \
                            bool conj_B, const T* B, stride_type rs_B, stride_type cs_B, \
                   T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C);
#include "configs/foreach_type.h"

template <typename T>
void mult(const communicator& comm, const config& cfg,
          len_type m, len_type n, len_type k,
          T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
                   bool conj_D, const T* D, stride_type inc_D,
                   bool conj_B, const T* B, stride_type rs_B, stride_type cs_B,
          T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C)
{
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C && !conj_D);

    const bool row_major = cfg.gemm_row_major.value<T>();

    if ((row_major ? rs_C : cs_C) == 1)
    {
        /*
         * Compute C^T = B^T * A^T instead
         */
        std::swap(m, n);
        std::swap(A, B);
        std::swap(rs_A, cs_B);
        std::swap(rs_B, cs_A);
        std::swap(rs_C, cs_C);
    }

    matrix_view<T> Av({m, k}, const_cast<T*>(A), {rs_A, cs_A});
    diag_scaled_matrix_view<T,0> Bv(k, n, const_cast<T*>(B), rs_B, cs_B, const_cast<T*>(D), inc_D);
    matrix_view<T> Cv({m, n}, C, {rs_C, cs_C});

    int nt = comm.num_threads();
    gemm_thread_config tc = make_gemm_thread_config<T>(cfg, nt, m, n, k);

    communicator comm_nc =    comm.gang(TCI_EVENLY, tc.jc_nt);
    communicator comm_kc = comm_nc.gang(TCI_EVENLY,        1);
    communicator comm_mc = comm_kc.gang(TCI_EVENLY, tc.ic_nt);
    communicator comm_nr = comm_mc.gang(TCI_EVENLY, tc.jr_nt);
    communicator comm_mr = comm_nr.gang(TCI_EVENLY, tc.ir_nt);

    GotoGEMM gemm;
    step<0>(gemm).subcomm = &comm_nc;
    step<1>(gemm).subcomm = &comm_kc;
    step<3>(gemm).subcomm = &comm_mc;
    step<5>(gemm).subcomm = &comm_nr;
    step<6>(gemm).subcomm = &comm_mr;

    gemm(comm, cfg, alpha, Av, Bv, beta, Cv);

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, const config& cfg, \
                   len_type m, len_type n, len_type k, \
                   T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A, \
                            bool conj_D, const T* D, stride_type inc_D, \
                            bool conj_B, const T* B, stride_type rs_B, stride_type cs_B, \
                   T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C);
#include "configs/foreach_type.h"

}
}
