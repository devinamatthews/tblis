#ifndef _TENSOR_TBLIS_GEMM_HPP_
#define _TENSOR_TBLIS_GEMM_HPP_

#include "tblis.hpp"

namespace tblis
{

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

template <typename Config, int IC, int JC, int KC, int IR, int JR, int KR,
          typename Child, typename... Children>
struct GEMM
{
    template <typename T>
    struct run
    {
        typename Child::template run<T, Children...> child;

        template <int I>
        auto step() -> decltype(detail::get_child<I>()(child))
        {
            return detail::get_child<I>()(child);
        }

        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
        {
            idx_type jc_way = envtol("BLIS_JC_NT");
            idx_type ic_way = envtol("BLIS_IC_NT");
            idx_type jr_way = envtol("BLIS_JR_NT");
            idx_type ir_way = envtol("BLIS_IR_NT");
            idx_type nthread = jc_way*ic_way*jr_way*ir_way;

            step<IC>().distribute = ic_way;
            step<JC>().distribute = jc_way;
            step<KC>().distribute = 1; //kc_way
            step<IR>().distribute = ir_way;
            step<JR>().distribute = jr_way;

            parallelize
            (
                [=](ThreadCommunicator& comm) mutable
                {
                    child(comm, alpha, A, B, beta, C);
                },
                nthread, Config::tree_barrier_arity
            );
        }
    };
};

template <typename T, typename Config=TBLIS_DEFAULT_CONFIG>
void tblis_gemm(T alpha, const_matrix_view<T> A,
                         const_matrix_view<T> B,
                T  beta,       matrix_view<T> C);

}

#endif
