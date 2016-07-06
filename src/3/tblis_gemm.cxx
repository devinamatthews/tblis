#include "tblis.hpp"

namespace tblis
{

namespace detail
{
    MemoryPool BuffersForA(4096);
    MemoryPool BuffersForB(4096);
}

template <typename T, typename Config>
using GotoGEMM = typename GEMM<Config, 3, 0, 1, 6, 5, -1,
                               PartitionN<Config::template NC>,
                               PartitionK<Config::template KC>,
                               PackB<Config::template KR, Config::template NR>,
                               PartitionM<Config::template MC>,
                               PackA<Config::template MR, Config::template KR>,
                               PartitionN<Config::template NR>,
                               PartitionM<Config::template MR>,
                               MicroKernel<Config>>::template run<T>;

template <typename T, typename Config>
void tblis_gemm(T alpha, const_matrix_view<T> A_,
                         const_matrix_view<T> B_,
                T  beta,       matrix_view<T> C_)
{
    constexpr bool row_major = Config::template gemm_row_major<T>::value;

    matrix_view<T> A(reinterpret_cast<matrix_view<T>&>(A_));
    matrix_view<T> B(reinterpret_cast<matrix_view<T>&>(B_));
    matrix_view<T> C(C_);

    assert(A.length(0) == C.length(0));
    assert(A.length(1) == B.length(0));
    assert(B.length(1) == C.length(1));

    if (C.stride(!row_major) == 1)
    {
        A.transpose();
        B.transpose();
        C.transpose();
        GotoGEMM<T,Config>()(alpha, B, A, beta, C);
    }
    else
    {
        GotoGEMM<T,Config>()(alpha, A, B, beta, C);
    }
}

#define INSTANTIATE_FOR_CONFIG(T,config) \
template \
void tblis_gemm<T,config>(T alpha, const_matrix_view<T> A, \
                                   const_matrix_view<T> B, \
                          T  beta,       matrix_view<T> C);
#include "tblis_instantiate_for_configs.hpp"

}
