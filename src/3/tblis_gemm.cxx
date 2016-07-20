#include "tblis.hpp"

namespace tblis
{

namespace detail
{
    MemoryPool BuffersForA(4096);
    MemoryPool BuffersForB(4096);
}

template <typename T, typename Config>
using GotoGEMM = typename GEMM<Config,
                               PartitionNC<Config>,
                               PartitionKC<Config>,
                               PackB<Config::template KR, Config::template NR>,
                               PartitionMC<Config>,
                               PackA<Config::template MR, Config::template KR>,
                               PartitionNR<Config>,
                               PartitionMR<Config>,
                               MicroKernel<Config>>::template run<T>;

template <typename T, typename Config>
void tblis_gemm(thread_communicator& comm,
                T alpha, const_matrix_view<T> A,
                         const_matrix_view<T> B,
                T  beta,       matrix_view<T> C)
{
    GotoGEMM<T,Config>()(comm, alpha,
                         reinterpret_cast<matrix_view<T>&>(A),
                         reinterpret_cast<matrix_view<T>&>(B),
                         beta, C);
}

template <typename T, typename Config>
void tblis_gemm(T alpha, const_matrix_view<T> A,
                         const_matrix_view<T> B,
                T  beta,       matrix_view<T> C)
{
    GotoGEMM<T,Config>()(alpha,
                         reinterpret_cast<matrix_view<T>&>(A),
                         reinterpret_cast<matrix_view<T>&>(B),
                         beta, C);
}

#define INSTANTIATE_FOR_CONFIG(T,config) \
template \
void tblis_gemm<T,config>(thread_communicator& comm, \
                          T alpha, const_matrix_view<T> A, \
                                   const_matrix_view<T> B, \
                          T  beta,       matrix_view<T> C); \
template \
void tblis_gemm<T,config>(T alpha, const_matrix_view<T> A, \
                                   const_matrix_view<T> B, \
                          T  beta,       matrix_view<T> C);
#include "tblis_instantiate_for_configs.hpp"

}
