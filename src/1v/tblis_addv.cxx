#include "tblis_addv.hpp"

#include "tblis_config.hpp"
#include "tblis_assert.hpp"

namespace tblis
{

template <typename T>
void tblis_addv_ref(thread_communicator& comm,
                    bool conj_A, len_type n,
                    const T* TBLIS_RESTRICT A, stride_type inc_A,
                          T* TBLIS_RESTRICT B, stride_type inc_B)
{
    if (n == 0) return;

    len_type n_min, n_max;
    std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(n);

    if (inc_A == 1 && inc_B == 1)
    {
        if (conj_A)
        {
            for (len_type i = n_min;i < n_max;i++)
            {
                B[i] += conj(A[i]);
            }
        }
        else
        {
            for (len_type i = n_min;i < n_max;i++)
            {
                B[i] += A[i];
            }
        }
    }
    else
    {
        A += n_min*inc_A;
        B += n_min*inc_B;

        if (conj_A)
        {
            for (len_type i = n_min;i < n_max;i++)
            {
                (*B) += conj(*A);
                A += inc_A;
                B += inc_B;
            }
        }
        else
        {
            for (len_type i = n_min;i < n_max;i++)
            {
                (*B) += (*A);
                A += inc_A;
                B += inc_B;
            }
        }
    }
}

template <typename T>
void tblis_addv(const_row_view<T> A, row_view<T> B)
{
    TBLIS_ASSERT(A.length() == B.length());
    tblis_addv(false, A.length(), A.data(), A.stride(),
                                  B.data(), B.stride());
}

template <typename T>
void tblis_addv(bool conj_A, len_type n,
                const T* A, stride_type inc_A,
                      T* B, stride_type inc_B)
{
    parallelize
    (
        [&](thread_communicator& comm)
        {
            tblis_addv_ref(comm, conj_A, n, A, inc_A, B, inc_B);
        }
    );
}

#define INSTANTIATE_FOR_TYPE(T) \
template void tblis_addv_ref(thread_communicator& comm, bool conj_A, idx_type n, const T* A, stride_type inc_A, T* B, stride_type inc_B); \
template void tblis_addv(bool conj_A, idx_type n, const T* A, stride_type inc_A, T* B, stride_type inc_B); \
template void tblis_addv(const_row_view<T> A, row_view<T> B);
#include "tblis_instantiate_for_types.hpp"

}
