#include "add.hpp"

namespace tblis
{
namespace internal
{

template <typename T>
void add(const communicator& comm, const config& cfg, len_type m, len_type n,
         T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
         T  beta, bool conj_B,       T* B, stride_type rs_B, stride_type cs_B)
{
    const bool TWOD = (rs_A < cs_A && rs_B > cs_B) ||
                      (rs_A > cs_A && rs_B < cs_B);

    const len_type MR = (TWOD ? cfg.trans_mr.def<T>() : 1);
    const len_type NR = (TWOD ? cfg.trans_nr.def<T>() : 1);

    const bool trans = (TWOD ? cfg.trans_row_major.value<T>() && rs_B < cs_B
                             : rs_B > cs_B);

    if (trans)
    {
        std::swap(m, n);
        std::swap(rs_A, cs_A);
        std::swap(rs_B, cs_B);
    }

    len_type m_min, m_max, n_min, n_max;
    std::tie(m_min, m_max, std::ignore,
             n_min, n_max, std::ignore) =
        comm.distribute_over_threads_2d(m, n, MR, NR);

    /*
     * If A is row-major and B is column-major or vice versa, use
     * the transpose microkernel.
     */
    if (TWOD)
    {
        if (beta == T(0))
        {
            for (len_type i = m_min;i < m_max;i += MR)
            {
                len_type m_loc = std::min(m-i, MR);
                for (len_type j = n_min;j < n_max;j += NR)
                {
                    len_type n_loc = std::min(n-j, NR);
                    cfg.trans_copy_ukr.call<T>(m_loc, n_loc,
                        alpha, conj_A, A + i*rs_A + j*cs_A, rs_A, cs_A,
                                       B + i*rs_B + j*cs_B, rs_B, cs_B);
                }
            }
        }
        else
        {
            for (len_type i = m_min;i < m_max;i += MR)
            {
                len_type m_loc = std::min(m-i, MR);
                for (len_type j = n_min;j < n_max;j += NR)
                {
                    len_type n_loc = std::min(n-j, NR);
                    cfg.trans_add_ukr.call<T>(m_loc, n_loc,
                        alpha, conj_A, A + i*rs_A + j*cs_A, rs_A, cs_A,
                         beta, conj_B, B + i*rs_B + j*cs_B, rs_B, cs_B);
                }
            }
        }
    }
    /*
     * Otherwise, A can be added to B column-by-column or row-by-row
     */
    else
    {
        if (beta == T(0))
        {
            for (len_type j = n_min;j < n_max;j++)
            {
                cfg.copy_ukr.call<T>(m_max-m_min,
                    alpha, conj_A, A + m_min*rs_A + j*cs_A, rs_A,
                                   B + m_min*rs_B + j*cs_B, rs_B);
            }
        }
        else
        {
            for (len_type j = n_min;j < n_max;j++)
            {
                cfg.add_ukr.call<T>(m_max-m_min,
                    alpha, conj_A, A + m_min*rs_A + j*cs_A, rs_A,
                     beta, conj_B, B + m_min*rs_B + j*cs_B, rs_B);
            }
        }
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void add(const communicator& comm, const config& cfg, len_type m, len_type n, \
                  T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A, \
                  T  beta, bool conj_B,       T* B, stride_type rs_B, stride_type cs_B);
#include "configs/foreach_type.h"

}
}
