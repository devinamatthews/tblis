#include "tblis_sumv.hpp"

#include "tblis_config.hpp"

namespace tblis
{

template <typename T>
void tblis_sumv_ref(thread_communicator& comm, len_type n,
                    const T* TBLIS_RESTRICT A, stride_type inc_A,
                    T& TBLIS_RESTRICT sum)
{
    sum = T();

    if (n == 0) return;

    T subsum = T();

    len_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(n);

    if (inc_A == 1)
    {
        for (len_type i = n_min;i < n_max;i++)
        {
            subsum += A[i];
        }
    }
    else
    {
        A += n_min*inc_A;
        for (len_type i = n_min;i < n_max;i++)
        {
            subsum += (*A);
            A += inc_A;
        }
    }

    comm.reduce(subsum);
    sum = subsum;
}

template <typename T>
void tblis_sumv(const_row_view<T> A, T& sum)
{
    tblis_sumv(A.length(), A.data(), A.stride(), sum);
}

template <typename T>
T tblis_sumv(const_row_view<T> A)
{
    T sum;
    tblis_sumv(A, sum);
    return sum;
}

template <typename T>
void tblis_sumv(len_type n, const T* A, stride_type inc_A, T& sum)
{
    parallelize
    (
        [&](thread_communicator& comm)
        {
            tblis_sumv_ref(comm, n, A, inc_A, sum);
        }
    );
}

template <typename T>
T tblis_sumv(len_type n, const T* A, stride_type inc_A)
{
    T sum;
    tblis_sumv(n, A, inc_A, sum);
    return sum;
}

#define INSTANTIATE_FOR_TYPE(T) \
template void tblis_sumv_ref(thread_communicator& comm, idx_type n, const T* A, stride_type inc_A, T& sum); \
template void tblis_sumv(idx_type n, const T* A, stride_type inc_A, T& sum); \
template    T tblis_sumv(idx_type n, const T* A, stride_type inc_A); \
template void tblis_sumv(const_row_view<T> A, T& sum); \
template    T tblis_sumv(const_row_view<T> A);
#include "tblis_instantiate_for_types.hpp"

}
