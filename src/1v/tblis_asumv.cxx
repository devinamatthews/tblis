#include "tblis.hpp"

namespace tblis
{

template <typename T>
void tblis_asumv_ref(ThreadCommunicator& comm,
                     idx_type n, const T* restrict A, stride_type inc_A, T& restrict sum)
{
    sum = T();

    if (n == 0) return;

    T subsum = T();

    idx_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(n);

    if (inc_A == 1)
    {
        for (idx_type i = n_min;i < n_max;i++)
        {
            subsum += std::abs(A[i]);
        }
    }
    else
    {
        A += n_min*inc_A;
        for (idx_type i = n_min;i < n_max;i++)
        {
            subsum += std::abs(*A);
            A += inc_A;
        }
    }

    comm.reduce(subsum);
    sum = subsum;
}

template <typename T>
void tblis_asumv(const_row_view<T> A, T& sum)
{
    tblis_asumv(A.length(), A.data(), A.stride(), sum);
}

template <typename T>
T tblis_asumv(const_row_view<T> A)
{
    T sum;
    tblis_asumv(A, sum);
    return sum;
}

template <typename T>
void tblis_asumv(idx_type n, const T* A, stride_type inc_A, T& sum)
{
    parallelize
    (
        [&](ThreadCommunicator& comm)
        {
            tblis_asumv_ref(comm, n, A, inc_A, sum);
        }
    );
}

template <typename T>
T tblis_asumv(idx_type n, const T* A, stride_type inc_A)
{
    T sum;
    tblis_asumv(n, A, inc_A, sum);
    return sum;
}

#define INSTANTIATE_FOR_TYPE(T) \
template void tblis_asumv_ref(ThreadCommunicator& comm, idx_type n, const T* A, stride_type inc_A, T& sum); \
template void tblis_asumv(idx_type n, const T* A, stride_type inc_A, T& sum); \
template    T tblis_asumv(idx_type n, const T* A, stride_type inc_A); \
template void tblis_asumv(const_row_view<T> A, T& sum); \
template    T tblis_asumv(const_row_view<T> A);
#include "tblis_instantiate_for_types.hpp"

}
