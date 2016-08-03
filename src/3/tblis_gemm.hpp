#ifndef _TENSOR_TBLIS_GEMM_HPP_
#define _TENSOR_TBLIS_GEMM_HPP_

#include "tblis.hpp"

namespace tblis
{

    /*
namespace detail
{
    template <int I, int J=0>
    struct get_child;

    template <int I>
    struct get_child<I, I>
    {
        template <typename T>
        T& operator()(T& child) const
        {
            return child;
        }
    };

    template <int I, int J>
    struct get_child
    {
        template <typename T>
        auto operator()(T& child) const
        -> decltype(get_child<I, J+1>()(child.child))
        {
            return get_child<I, J+1>()(child.child);
        }
    };
}
*/

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
gemm_thread_config make_gemm_thread_config(int nthread, idx_type m, idx_type n, idx_type k)
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

template <typename Config, typename Child, typename... Children>
struct GEMM
{
    template <typename T>
    struct run
    {
        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(thread_communicator& comm, T alpha, MatrixA A, MatrixB B, T beta, MatrixC C)
        {
            constexpr bool row_major = Config::template gemm_row_major<T>::value;

            assert(A.length(0) == C.length(0));
            assert(A.length(1) == B.length(0));
            assert(B.length(1) == C.length(1));

            typename Child::template run<T, Children...> child;

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

template <typename T, typename Config=TBLIS_DEFAULT_CONFIG>
void tblis_gemm(thread_communicator& comm,
                T alpha, const_matrix_view<T> A,
                         const_matrix_view<T> B,
                T  beta,       matrix_view<T> C);

template <typename T, typename Config=TBLIS_DEFAULT_CONFIG>
void tblis_gemm(T alpha, const_matrix_view<T> A,
                         const_matrix_view<T> B,
                T  beta,       matrix_view<T> C);

}

#endif
