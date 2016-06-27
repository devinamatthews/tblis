#ifndef _TENSOR_TBLIS_GEMM_HPP_
#define _TENSOR_TBLIS_GEMM_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
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
            dim_t jc_way = bli_read_nway_from_env( "BLIS_JC_NT" );
            dim_t ic_way = bli_read_nway_from_env( "BLIS_IC_NT" );
            dim_t jr_way = bli_read_nway_from_env( "BLIS_JR_NT" );
            dim_t ir_way = bli_read_nway_from_env( "BLIS_IR_NT" );
            dim_t nthread = jc_way*ic_way*jr_way*ir_way;

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

template <typename T, typename Config>
using GotoGEMM = GEMM<Config, 3, 0, 1, 6, 5, -1,
                      PartitionN<Config::NC>,
                      PartitionK<Config::KC>,
                      PackB<Config::KR,Config::NR>,
                      PartitionM<Config::MC>,
                      PackA<Config::MR,Config::KR>,
                      PartitionN<Config::NR>,
                      PartitionM<Config::MR>,
                      MicroKernel<Config>>::run<T>;

template <typename Config, typename T, typename MatrixA, typename MatrixB, typename MatrixC>
void tblis_gemm_int(T alpha, const MatrixA& A, const MatrixB& B, T beta, MatrixC&& C)
{
    assert(A.length() == C.length(), "m dimension does not match");
    assert(A.width() == B.length(), "k dimension does not match");
    assert(B.width() == C.width(), "n dimension does not match");

    GotoGEMM<T,Config>()(alpha, A, B, beta, C);
}

template <typename Config, typename T, typename MatrixA, typename MatrixB, typename MatrixC>
void tblis_gemm(T alpha, const MatrixA& A, const MatrixB& B, T beta, MatrixC&& C)
{
    if ((Config::template gemm_row_major<T>::value ? C.row_stride() : C.col_stride()) == 1)
    {
        tblis_gemm_int<Config>(alpha, B.transposed(), A.transposed(), beta, C.transposed());
    }
    else
    {
        tblis_gemm_int<Config>(alpha, A, B, beta, C);
    }
}

}
}

#endif
