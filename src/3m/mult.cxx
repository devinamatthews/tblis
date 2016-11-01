#include "mult.h"

#include "nodes/partm.hpp"
#include "nodes/packm.hpp"
#include "nodes/gemm_ukr.hpp"

#include "util/env.hpp"

namespace tblis
{

namespace
{
    MemoryPool BuffersForA(4096);
    MemoryPool BuffersForB(4096);
}

struct gemm_thread_config
{
    gemm_thread_config(int jc_nt, int ic_nt, int jr_nt, int ir_nt)
    : jc_nt(jc_nt), ic_nt(ic_nt), jr_nt(jr_nt), ir_nt(ir_nt) {}

    int jc_nt = 1;
    int ic_nt = 1;
    int jr_nt = 1;
    int ir_nt = 1;
};

gemm_thread_config make_gemm_thread_config(int nthread, len_type m, len_type n, len_type k)
{
    int ic_nt, jc_nt;
    std::tie(ic_nt, jc_nt) = partition_2x2(nthread, m*2, n);

    int ir_nt = 1, jr_nt;
    for (jr_nt = 3;jr_nt > 1;jr_nt--)
    {
        if (jc_nt%jr_nt == 0)
        {
            jc_nt /= jr_nt;
            break;
        }
    }

    jc_nt = envtol("BLIS_JC_NT", jc_nt);
    ic_nt = envtol("BLIS_IC_NT", ic_nt);
    jr_nt = envtol("BLIS_JR_NT", jr_nt);
    ir_nt = envtol("BLIS_IR_NT", ir_nt);

    TBLIS_ASSERT(ir_nt*jr_nt*ic_nt*jc_nt == nthread);

    return {jc_nt, ic_nt, jr_nt, ir_nt};
}

using GotoGEMM = partition_gemm_nc<
                   partition_gemm_kc<
                     pack_b<BuffersForB,
                       partition_gemm_mc<
                         pack_a<BuffersForA,
                           partition_gemm_nr<
                             partition_gemm_mr<
                               gemm_micro_kernel
                             >
                           >
                         >
                       >
                     >
                   >
                 >;

template <int N> struct step_helper;

template <>
struct step_helper<0>
{
    template <typename T>
    T& operator()(T& tree) const { return tree; }
};

template <int N>
struct step_helper
{
    template <typename T>
    auto operator()(T& tree) const -> decltype(step_helper<N-1>()(tree.child))
    {
        return step_helper<N-1>()(tree.child);
    }
};

template <int N, typename T>
auto step(T& tree) -> decltype(step_helper<N>()(tree))
{
    return step_helper<N>()(tree);
}

void mult_int(const communicator& comm, const config& cfg,
              const tblis_matrix& A, const tblis_matrix& B, tblis_matrix& C)
{
    TBLIS_ASSERT(A.m == C.m);
    TBLIS_ASSERT(B.n == C.n);
    TBLIS_ASSERT(A.n == B.m);
    TBLIS_ASSERT(A.type == B.type);
    TBLIS_ASSERT(A.type == C.type);

    TBLIS_ASSERT(!A.conj);
    TBLIS_ASSERT(!B.conj);
    TBLIS_ASSERT(!C.conj);

    GotoGEMM gemm;

    int nt = comm.num_threads();
    len_type m = C.m;
    len_type n = C.m;
    len_type k = A.n;

    TBLIS_WITH_TYPE_AS(A.type, T,
    {
        const bool row_major = cfg.gemm_row_major<T>();

        matrix_view<T> Av, Bv, Cv;

        if ((row_major ? C.rs : C.cs) == 1)
        {
            /*
             * Compute C^T = B^T * A^T instead
             */
            std::swap(m, n);
            Av.reset({m, k}, (T*)B.data, {B.cs, B.rs});
            Bv.reset({k, n}, (T*)A.data, {A.cs, A.rs});
            Cv.reset({m, n}, (T*)C.data, {C.cs, C.rs});
        }
        else
        {
            Av.reset({m, k}, (T*)A.data, {A.rs, A.cs});
            Bv.reset({k, n}, (T*)B.data, {B.rs, B.cs});
            Cv.reset({m, n}, (T*)C.data, {C.rs, C.cs});
        }

        auto tc = make_gemm_thread_config(nt, m, n, k);
        step<0>(gemm).distribute = tc.jc_nt;
        step<3>(gemm).distribute = tc.ic_nt;
        step<5>(gemm).distribute = tc.jr_nt;
        step<6>(gemm).distribute = tc.ir_nt;

        T alpha = A.alpha<T>()*B.alpha<T>();
        T beta = C.alpha<T>();

        gemm(comm, cfg, alpha, Av, Bv, beta, Cv);

        if (comm.master())
        {
            C.alpha<T>() = T(1);
            C.conj = false;
        }
    })
}

extern "C"
{

void tblis_matrix_mult(const tblis_comm* comm, const tblis_config* cfg,
                       const tblis_matrix* A, const tblis_matrix* B,
                       tblis_matrix* C)
{
    parallelize_if(mult_int, comm, get_config(cfg),
                   *A, *B, *C);
}

}

}
