#ifndef _TENSOR_TBLIS_GEMM_HPP_
#define _TENSOR_TBLIS_GEMM_HPP_

#include <type_traits>
#include <utility>

#include "tblis.hpp"

#include "util/util.hpp"

namespace tblis
{
namespace blis_like
{

template <typename Child, typename... Children>
struct GEMM;

namespace detail
{
    template <int I, int J>
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
        -> decltype(get_child<I, J+1>()(std::declval<T&>().child))
        {
            return get_child<I, J+1>()(child.child);
        }
    };
}

template <typename Child, typename... Children>
struct GEMM
{
    template <typename T>
    struct run
    {
        typename Child::template run<T, Children...> child;

        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
        {
            child(comm, alpha, A, B, beta, C);
        }

        template <int I>
        auto step() -> decltype(detail::get_child<I, 0>()(child))
        {
            return detail::get_child<I, 0>()(child);
        }
    };
};

template <template <typename> class MT, template <typename> class NT>
struct MacroKernel
{
    template <typename T>
    using run = typename PartitionN<NT>::template run<T, PartitionM<MT>, MicroKernel<MT,NT>>;
};

template <typename T>
using DefaultGEMM = GEMM<PartitionN<NC>,
                         PartitionK<KC>,
                         PackB<KR,NR>,
                         PartitionM<MC>,
                         PackA<MR,KR>,
                         MacroKernel<MR,NR>>::run<T>;

template <typename T, typename MatrixA, typename MatrixB, typename MatrixC>
void tblis_gemm_int(T alpha, MatrixA&& A, MatrixB&& B, T beta, MatrixC&& C)
{
    ASSERT(A.length() == C.length(), "m dimension does not match");
    ASSERT(A.width() == B.length(), "k dimension does not match");
    ASSERT(B.width() == C.width(), "n dimension does not match");

    dim_t jc_way = bli_read_nway_from_env( "BLIS_JC_NT" );
    dim_t ic_way = bli_read_nway_from_env( "BLIS_IC_NT" );
    dim_t jr_way = bli_read_nway_from_env( "BLIS_JR_NT" );
    dim_t ir_way = bli_read_nway_from_env( "BLIS_IR_NT" );
    dim_t nthread = jc_way*ic_way*jr_way*ir_way;

    //printf_locked("%d %d %d %d\n", jc_way, ic_way, jr_way, ir_way);

    DefaultGEMM<T> gemm;
    gemm.template step<0>().distribute = jc_way;
    gemm.template step<1>().distribute = 1; //kc_way
    gemm.template step<3>().distribute = ic_way;
    gemm.template step<5>().distribute = jr_way;
    gemm.template step<6>().distribute = ir_way;

    parallelize(nthread,
    [=](ThreadCommunicator& comm) mutable
    {
        gemm(comm, alpha, A, B, beta, C);
    });
}

template <typename U, typename MatrixA, typename MatrixB, typename MatrixC>
void tblis_gemm(U alpha, const MatrixA& A, const MatrixB& B, U beta, MatrixC&& C)
{
    MatrixA Av;
    MatrixB Bv;
    typename std::decay<MatrixC>::type Cv;

    //printf("%d %ld %ld\n", (int)C.is_transposed(), C.row_stride(), C.col_stride());

    if ((C.is_transposed() ? C.row_stride() : C.col_stride()) == 1)
    {
        using namespace blis::transpose;
        ViewNoTranspose(const_cast<MatrixB&>(B)^T, Bv);
        ViewNoTranspose(const_cast<MatrixA&>(A)^T, Av);
        ViewNoTranspose(                     C ^T, Cv);

        tblis_gemm_int(alpha, Bv, Av, beta, Cv);
    }
    else
    {
        ViewNoTranspose(const_cast<MatrixA&>(A), Av);
        ViewNoTranspose(const_cast<MatrixB&>(B), Bv);
        ViewNoTranspose(                     C , Cv);

        tblis_gemm_int(alpha, Av, Bv, beta, Cv);
    }
}

}
}

#endif
