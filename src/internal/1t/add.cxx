#include "add.hpp"

#include "util/tensor.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void add(const communicator& comm, const config& cfg,
         const std::vector<len_type>& len_A,
         const std::vector<len_type>& len_B,
         const std::vector<len_type>& len_AB,
         T alpha, bool conj_A, const T* A,
         const std::vector<stride_type>& stride_A,
         const std::vector<stride_type>& stride_A_AB,
         T  beta, bool conj_B,       T* B,
         const std::vector<stride_type>& stride_B,
         const std::vector<stride_type>& stride_B_AB)
{
    if (len_A.empty() && len_B.empty() && !len_AB.empty())
    {
        len_type len0 = len_AB[0];
        std::vector<len_type> len1(len_AB.begin()+1, len_AB.end());

        stride_type stride_A0 = stride_A_AB[0];
        std::vector<stride_type> stride_A1(stride_A_AB.begin()+1,
                                           stride_A_AB.end());

        stride_type stride_B0 = stride_B_AB[0];
        std::vector<stride_type> stride_B1(stride_B_AB.begin()+1,
                                           stride_B_AB.end());

        MArray::viterator<2> iter_AB(len1, stride_A1, stride_B1);
        len_type n = stl_ext::prod(len1);

        len_type m_min, m_max, n_min, n_max;
        std::tie(m_min, m_max, std::ignore,
                 n_min, n_max, std::ignore) =
            comm.distribute_over_threads_2d(len0, n);

        iter_AB.position(n_min, A, B);
        A += m_min*stride_A0;
        B += m_min*stride_B0;

        if (beta == T(0))
        {
            for (len_type i = n_min;i < n_max;i++)
            {
                iter_AB.next(A, B);
                cfg.copy_ukr.call<T>(m_max-m_min,
                                     alpha, conj_A, A, stride_A0,
                                                    B, stride_B0);
            }
        }
        else
        {
            for (len_type i = n_min;i < n_max;i++)
            {
                iter_AB.next(A, B);
                cfg.add_ukr.call<T>(m_max-m_min,
                                    alpha, conj_A, A, stride_A0,
                                     beta, conj_B, B, stride_B0);
            }
        }
    }
    else
    {
        MArray::viterator<1> iter_A(len_A, stride_A);
        MArray::viterator<1> iter_B(len_B, stride_B);
        MArray::viterator<2> iter_AB(len_AB, stride_A_AB, stride_B_AB);
        len_type n = stl_ext::prod(len_AB);

        len_type n_min, n_max;
        std::tie(n_min, n_max, std::ignore) = comm.distribute_over_threads(n);

        iter_AB.position(n_min, A, B);

        for (len_type i = n_min;i < n_max;i++)
        {
            iter_AB.next(A, B);

            T sum_A = T();
            while (iter_A.next(A)) sum_A += *A;
            sum_A = alpha*(conj_A ? conj(sum_A) : sum_A);

            TBLIS_SPECIAL_CASE(is_complex<T>::value && conj_B,
            TBLIS_SPECIAL_CASE(beta == T(0),
            {
                while (iter_B.next(B))
                {
                    *B = sum_A + beta*(conj_B ? conj(*B) : *B);
                }
            }
            ))
        }
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void add(const communicator& comm, const config& cfg, \
                  const std::vector<len_type>& len_A, \
                  const std::vector<len_type>& len_B, \
                  const std::vector<len_type>& len_AB, \
                  T alpha, bool conj_A, const T* A, \
                  const std::vector<stride_type>& stride_A, \
                  const std::vector<stride_type>& stride_A_AB, \
                  T  beta, bool conj_B,       T* B, \
                  const std::vector<stride_type>& stride_B, \
                  const std::vector<stride_type>& stride_B_AB);
#include "configs/foreach_type.h"

}
}
