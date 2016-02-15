#ifndef _TENSOR_TBLIS_GEMM_HPP_
#define _TENSOR_TBLIS_GEMM_HPP_

#include <type_traits>

#include "pack.hpp"
#include "partition.hpp"
#include "ukernel.hpp"
#include "normfm.hpp"

namespace tblis
{

template <typename Child, typename... Children>
struct GEMM
{
    template <typename T, typename MatrixA, typename MatrixB, typename MatrixC>
    static void run(T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        typename Child::template run<T, Children...>()(alpha, A, B, beta, C);
    }
};

typedef GEMM<PartitionN<NC>,
             PartitionK<KC>,
             PackB<NR,KR>,
             PartitionM<MC>,
             PackA<MR,KR>,
             PartitionN<NR>,
             PartitionM<MR>,
             MicroKernel<MR,NR>> DefaultGEMM;

template <typename T, typename MatrixA, typename MatrixB, typename MatrixC>
void tblis_gemm(T alpha, const MatrixA& A, const MatrixB& B, T beta, MatrixC&& C)
{
    ASSERT(A.length() == C.length(), "m dimension does not match");
    ASSERT(A.width() == B.length(), "k dimension does not match");
    ASSERT(B.width() == C.width(), "n dimension does not match");

    MatrixA Av;
    MatrixB Bv;
    typename std::decay<MatrixC>::type Cv;

    ViewNoTranspose(const_cast<MatrixA&>(A), Av);
    ViewNoTranspose(const_cast<MatrixB&>(B), Bv);
    ViewNoTranspose(                     C , Cv);

    DefaultGEMM::run(alpha, Av, Bv, beta, Cv);
}

}

#endif
