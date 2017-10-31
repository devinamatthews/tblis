#include "dot.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dot(const communicator& comm, const config& cfg,
         const len_vector& len_AB,
         bool conj_A, const T* A, const stride_vector& stride_A_AB,
         bool conj_B, const T* B, const stride_vector& stride_B_AB,
         T& result)
{
    (void)cfg;

    len_type n = stl_ext::prod(len_AB);

    atomic_accumulator<T> local_result;

    if (conj_A) conj_B = !conj_B;

    comm.distribute_over_threads(n,
    [&](len_type n_min, len_type n_max)
    {
        auto A1 = A;
        auto B1 = B;

        viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
        iter_AB.position(n_min, A1, B1);

        T micro_result = T();

        for (len_type i = n_min;i < n_max;i++)
        {
            iter_AB.next(A1, B1);
            micro_result += (*A1)*(conj_B ? conj(*B1) : *B1);
        }

        local_result += micro_result;
    });

    reduce(comm, local_result);
    if (comm.master())
        result = (conj_A ? conj((T)local_result) : (T)local_result);

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void dot(const communicator& comm, const config& cfg, \
                  const len_vector& len_AB, \
                  bool conj_A, const T* A, const stride_vector& stride_A_AB, \
                  bool conj_B, const T* B, const stride_vector& stride_B_AB, \
                  T& result);
#include "configs/foreach_type.h"

}
}
