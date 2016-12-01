#include "dot.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void dot(const communicator& comm, const config& cfg,
         const std::vector<len_type>& len_A,
         const std::vector<len_type>& len_B,
         const std::vector<len_type>& len_AB,
         bool conj_A, const T* A, const std::vector<stride_type>& stride_A,
                                  const std::vector<stride_type>& stride_A_AB,
         bool conj_B, const T* B, const std::vector<stride_type>& stride_B,
                                  const std::vector<stride_type>& stride_B_AB,
         T& result)
{
    (void)cfg;

    MArray::viterator<1> iter_A(len_A, stride_A);
    MArray::viterator<1> iter_B(len_B, stride_B);
    MArray::viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);

    len_type n = stl_ext::prod(len_AB);

    len_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(n);

    T local_result = T();

    if (conj_A) conj_B = !conj_B;

    iter_AB.position(n_min, A, B);

    for (len_type i = n_min;i < n_max;i++)
    {
        iter_AB.next(A, B);

        T sum_A = T();
        T sum_B = T();
        while (iter_A.next(A)) sum_A += *A;
        while (iter_B.next(B)) sum_B += *B;

        if (conj_B)
        {
            local_result += sum_A*conj(sum_B);
        }
        else
        {
            local_result += sum_A*sum_B;
        }
    }

    len_type dummy = 0;
    reduce(comm, REDUCE_SUM, local_result, dummy);
    if (comm.master())
        result = (conj_A ? conj(local_result) : local_result);

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void dot(const communicator& comm, const config& cfg, \
                  const std::vector<len_type>& len_A, \
                  const std::vector<len_type>& len_B, \
                  const std::vector<len_type>& len_AB, \
                  bool conj_A, const T* A, const std::vector<stride_type>& stride_A, \
                                           const std::vector<stride_type>& stride_A_AB, \
                  bool conj_B, const T* B, const std::vector<stride_type>& stride_B, \
                                           const std::vector<stride_type>& stride_B_AB, \
                  T& result);
#include "configs/foreach_type.h"

}
}
