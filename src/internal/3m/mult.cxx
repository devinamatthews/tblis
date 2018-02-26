#include "mult.hpp"

#include "util/gemm_thread.hpp"

#include "matrix/normal_matrix.hpp"
#include "matrix/diag_scaled_matrix.hpp"

#include "nodes/gemm.hpp"

namespace tblis
{

MemoryPool BuffersForA(4096);
MemoryPool BuffersForB(4096);

namespace internal
{

template <typename T>
void mult(const communicator& comm, const config& cfg,
          len_type m, len_type n, len_type k,
          T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
                   bool conj_B, const T* B, stride_type rs_B, stride_type cs_B,
          T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C)
{
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C);

    normal_matrix<T> Av(m, k, const_cast<T*>(A), rs_A, cs_A);
    normal_matrix<T> Bv(k, n, const_cast<T*>(B), rs_B, cs_B);
    normal_matrix<T> Cv(m, n,                C , rs_C, cs_C);

    GotoGEMM{}(comm, cfg, alpha, Av, Bv, beta, Cv);

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, const config& cfg, \
                   len_type m, len_type n, len_type k, \
                   T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A, \
                            bool conj_B, const T* B, stride_type rs_B, stride_type cs_B, \
                   T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C);
#include "configs/foreach_type.h"

template <typename T>
void mult(const communicator& comm, const config& cfg,
          len_type m, len_type n, len_type k,
          T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
                   bool conj_D, const T* D, stride_type inc_D,
                   bool conj_B, const T* B, stride_type rs_B, stride_type cs_B,
          T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C)
{
    TBLIS_ASSERT(!conj_A && !conj_B && !conj_C && !conj_D);

         normal_matrix<T  > Av(m, k, const_cast<T*>(A), rs_A, cs_A);
    diag_scaled_matrix<T,0> Bv(k, n, const_cast<T*>(B), rs_B, cs_B,
                                     const_cast<T*>(D), inc_D);
         normal_matrix<T  > Cv(m, n,                C , rs_C, cs_C);

    GotoGEMM{}(comm, cfg, alpha, Av, Bv, beta, Cv);

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, const config& cfg, \
                   len_type m, len_type n, len_type k, \
                   T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A, \
                            bool conj_D, const T* D, stride_type inc_D, \
                            bool conj_B, const T* B, stride_type rs_B, stride_type cs_B, \
                   T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C);
#include "configs/foreach_type.h"

}
}
