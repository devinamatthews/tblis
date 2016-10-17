#ifndef _TBLIS_GEMM_TEMPLATE_HPP_
#define _TBLIS_GEMM_TEMPLATE_HPP_

#include "../../external/tci/src/tci.hpp"
#include "../../tblis_config.h"
#include "../configs/configs.hpp.in"
#include "../util/assert.h"
#include "../util/basic_types.h"

namespace tblis
{

struct gemm_thread_config
{
    gemm_thread_config(int jc_nt, int ic_nt, int jr_nt, int ir_nt)
    : jc_nt(jc_nt), ic_nt(ic_nt), jr_nt(jr_nt), ir_nt(ir_nt) {}

    int jc_nt = 1;
    int ic_nt = 1;
    int jr_nt = 1;
    int ir_nt = 1;
};

template <typename Config>
gemm_thread_config make_gemm_thread_config(int nthread, len_type m, len_type n, len_type k)
{
    int ic_nt, jc_nt;
    partition_2x2(nthread, m*2, n, ic_nt, jc_nt);

    int ir_nt = 1, jr_nt;
    for (jr_nt = 3;jr_nt > 1;jr_nt--)
    {
        if (jc_nt%jr_nt == 0)
        {
            jc_nt /= jr_nt;
            break;
        }
    }

    int jc_nt_ev = envtol("BLIS_JC_NT", 0);
    int ic_nt_ev = envtol("BLIS_IC_NT", 0);
    int jr_nt_ev = envtol("BLIS_JR_NT", 0);
    int ir_nt_ev = envtol("BLIS_IR_NT", 0);
    if (jc_nt_ev) jc_nt = jc_nt_ev;
    if (ic_nt_ev) ic_nt = ic_nt_ev;
    if (jr_nt_ev) jr_nt = jr_nt_ev;
    if (ir_nt_ev) ir_nt = ir_nt_ev;

    return {jc_nt, ic_nt, jr_nt, ir_nt};
}

template <typename Config, template <typename> class Child, template <typename> class... Children>
struct GEMM
{
    template <typename T>
    struct run
    {
        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(thread_communicator& comm, T alpha, MatrixA A, MatrixB B, T beta, MatrixC C)
        {
            constexpr bool row_major = Config::template gemm_row_major<T>::value;

            TBLIS_ASSERT(A.length(0) == C.length(0));
            TBLIS_ASSERT(A.length(1) == B.length(0));
            TBLIS_ASSERT(B.length(1) == C.length(1));

            typename Child<Config>::template run<T, Children...> child;

            if (C.stride(!row_major) == 1)
            {
                A.transpose();
                B.transpose();
                C.transpose();
                child(make_gemm_thread_config<Config>(
                    comm.num_threads(), C.length(0), C.length(1), B.length(1)),
                    comm, alpha, B, A, beta, C);
            }
            else
            {
                child(make_gemm_thread_config<Config>(
                    comm.num_threads(), C.length(0), C.length(1), A.length(1)),
                    comm, alpha, A, B, beta, C);
            }
        }

        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
        {
            parallelize
            (
                [&](thread_communicator& comm)
                {
                    operator()(comm, alpha, A, B, beta, C);
                },
                0, Config::tree_barrier_arity
            );
        }
    };
};

}

#endif
