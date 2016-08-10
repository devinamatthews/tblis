#include "tblis_gemm.hpp"

#include "tblis_gemm_template.hpp"

namespace tblis
{

template <typename T, typename Config=TBLIS_DEFAULT_CONFIG>
using GotoGEMM = typename GEMM<Config,
                               PartitionNC,
                               PartitionKC,
                               PackB,
                               PartitionMC,
                               PackA,
                               PartitionNR,
                               PartitionMR,
                               MicroKernel>::template run<T>;

template <typename T>
void tblis_gemm(thread_communicator& comm,
                T alpha, const_matrix_view<T> A,
                         const_matrix_view<T> B,
                T  beta,       matrix_view<T> C)
{
    GotoGEMM<T>()(comm, alpha,
                  reinterpret_cast<matrix_view<T>&>(A),
                  reinterpret_cast<matrix_view<T>&>(B),
                  beta, C);
}

template <typename T>
void tblis_gemm(T alpha, const_matrix_view<T> A,
                         const_matrix_view<T> B,
                T  beta,       matrix_view<T> C)
{
    GotoGEMM<T>()(alpha,
                  reinterpret_cast<matrix_view<T>&>(A),
                  reinterpret_cast<matrix_view<T>&>(B),
                  beta, C);
}

#define INSTANTIATE_FOR_TYPE(T) \
template \
void tblis_gemm<T>(thread_communicator& comm, \
                   T alpha, const_matrix_view<T> A, \
                            const_matrix_view<T> B, \
                   T  beta,       matrix_view<T> C); \
template \
void tblis_gemm<T>(T alpha, const_matrix_view<T> A, \
                            const_matrix_view<T> B, \
                   T  beta,       matrix_view<T> C);
#include "tblis_instantiate_for_types.hpp"

}
