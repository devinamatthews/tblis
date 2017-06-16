#include "mult.hpp"

#include "util/gemm_thread.hpp"

#include "nodes/partm.hpp"
#include "nodes/packm.hpp"
#include "nodes/gemm_mkr.hpp"

namespace tblis
{
namespace internal
{

extern MemoryPool BuffersForA, BuffersForB;
MemoryPool BuffersForA(4096);
MemoryPool BuffersForB(4096);

using GotoGEMM = partition_gemm_nc<
                   partition_gemm_kc<
                     pack_b<BuffersForB,
                       partition_gemm_mc<
                         pack_a<BuffersForA,
                           partition_gemm_nr<
                             partition_gemm_mr<
                               gemm_micro_kernel>>>>>>>;

using GotoGEMM2 = partition_gemm_nc<
                    partition_gemm_kc<
                      pack_b<BuffersForB,
                        partition_gemm_mc<
                          pack_a<BuffersForA,
                           gemm_macro_kernel>>>>>;

template <typename T>
void mult(const communicator& comm, const config& cfg,
          len_type m, len_type n, len_type k,
          T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
                   bool conj_B, const T* B, stride_type rs_B, stride_type cs_B,
          T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C)
{
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);

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
    matrix_view<T> Bv({k, n}, const_cast<T*>(B), {rs_B, cs_B});
    matrix_view<T> Cv({m, n},                C , {rs_C, cs_C});

#if TBLIS_ENABLE_TBB
#if TBB_INTERFACE_VERSION >= 9100
    int nt = tbb::this_task_arena::max_concurrency();
#else //TBB_INTERFACE_VERSION >= 9100
    int nt = tblis_get_num_threads();
#endif //TBB_INTERFACE_VERSION >= 9100
#else //TBLIS_ENABLE_TBB
    int nt = comm.num_threads();
#endif //TBLIS_ENABLE_TBB

    gemm_thread_config tc = make_gemm_thread_config<T>(cfg, nt, m, n, k);

#if TBLIS_ENABLE_TBB && !TBLIS_SIMPLE_TBB
    GotoGEMM2 gemm;
    step<0>(gemm).distribute = tc.jc_nt;
    step<3>(gemm).distribute = tc.ic_nt;
    step<5>(gemm).distribute_n = tc.jr_nt;
    step<5>(gemm).distribute_m = tc.ir_nt;
#else
    GotoGEMM gemm;
    step<0>(gemm).distribute = tc.jc_nt;
    step<3>(gemm).distribute = tc.ic_nt;
    step<5>(gemm).distribute = tc.jr_nt;
    step<6>(gemm).distribute = tc.ir_nt;
#endif

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

}
}
