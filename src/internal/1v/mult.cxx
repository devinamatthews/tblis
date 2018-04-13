#include "add.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void mult(const communicator& comm, const config& cfg, len_type n,
          T alpha, bool conj_A, const T* A, stride_type inc_A,
                   bool conj_B, const T* B, stride_type inc_B,
          T  beta, bool conj_C,       T* C, stride_type inc_C)
{
    comm.distribute_over_threads(n,
    [&](len_type n_min, len_type n_max)
    {
        cfg.mult_ukr.call<T>(n_max-n_min,
                             alpha, conj_A, A + n_min*inc_A, inc_A,
                                    conj_B, B + n_min*inc_B, inc_B,
                              beta, conj_C, C + n_min*inc_C, inc_C);
    });
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, const config& cfg, len_type n, \
                   T alpha, bool conj_A, const T* A, stride_type inc_A, \
                            bool conj_B, const T* B, stride_type inc_B, \
                   T  beta, bool conj_C,       T* C, stride_type inc_C);
#include "configs/foreach_type.h"

}
}
