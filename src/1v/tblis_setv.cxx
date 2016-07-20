#include "tblis.hpp"

namespace tblis
{

template <typename T>
void tblis_setv_ref(thread_communicator& comm, idx_type n, T alpha, T* A, stride_type inc_A)
{
    if (n == 0) return;

    idx_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(n);

    if (inc_A == 1)
    {
        for (idx_type i = n_min;i < n_max;i++)
        {
            A[i] = alpha;
        }
    }
    else
    {
        A += n_min*inc_A;
        for (idx_type i = n_min;i < n_max;i++)
        {
            (*A) = alpha;
            A += inc_A;
        }
    }
}

template <typename T>
void tblis_setv(T alpha, row_view<T> A)
{
    tblis_setv(A.length(), alpha, A.data(), A.stride());
}

template <typename T>
void tblis_setv(idx_type n, T alpha, T* A, stride_type inc_A)
{
    if (alpha == T(0))
    {
        tblis_zerov(n, A, inc_A);
    }
    else
    {
        parallelize
        (
            [&](thread_communicator& comm)
            {
                tblis_setv_ref(comm, n, alpha, A, inc_A);
            }
        );
    }
}

#define INSTANTIATE_FOR_TYPE(T) \
template void tblis_setv_ref(thread_communicator& comm, idx_type n, T alpha, T* A, stride_type inc_A); \
template void tblis_setv(idx_type n, T alpha, T* A, stride_type inc_A); \
template void tblis_setv(T alpha, row_view<T> A);
#include "tblis_instantiate_for_types.hpp"

}
