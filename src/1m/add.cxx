#include "add.h"

#include "configs/configs.hpp"

namespace tblis
{

void add_int(const communicator& comm, const config& cfg,
             const tblis_matrix& A, tblis_matrix& B)
{
    TBLIS_ASSERT(A.m == B.m);
    TBLIS_ASSERT(A.n == B.n);
    TBLIS_ASSERT(A.type == B.type);

    TBLIS_WITH_TYPE_AS(A.type, T,
    {
        const bool TWOD = ((A.rs < A.cs && B.rs > B.cs) ||
                           (A.rs > A.cs && B.rs < B.cs)) &&
                          A.alpha<T>() != T(0);
        const bool trans = cfg.trans_row_major<T>() && B.rs < B.cs;
        const len_type MR = cfg.trans_mr.def<T>();
        const len_type NR = cfg.trans_nr.def<T>();

        /*
         * If A is row-major and B is column-major or vice versa, use
         * the transpose microkernel.
         */
        if (TWOD)
        {
            /*
             * Logically transpose A and B if storage of B does not match
             * microkernel preference.
             */
            if (trans)
            {
                len_type m_min, m_max, n_min, n_max;
                std::tie(m_min, m_max, std::ignore,
                         n_min, n_max, std::ignore) =
                    comm.distribute_over_threads_2d(A.n, A.m, MR, NR);

                if (B.alpha<T>() == T(0))
                {
                    for (len_type i = m_min;i < m_max;i += MR)
                    {
                        len_type i_loc = std::min(A.n-i, MR);
                        for (len_type j = n_min;j < n_max;j += NR)
                        {
                            len_type j_loc = std::min(A.m-j, NR);
                            cfg.trans_copy_ukr.call<T>(i_loc, j_loc,
                                A.alpha<T>(), A.conj, (const T*)A.data + i*A.cs + j*A.rs, A.cs, A.rs,
                                                            (T*)B.data + i*B.cs + j*B.rs, B.cs, B.rs);
                        }
                    }
                }
                else
                {
                    for (len_type i = m_min;i < m_max;i += MR)
                    {
                        len_type i_loc = std::min(A.n-i, MR);
                        for (len_type j = n_min;j < n_max;j += NR)
                        {
                            len_type j_loc = std::min(A.m-j, NR);
                            cfg.trans_add_ukr.call<T>(i_loc, j_loc,
                                A.alpha<T>(), A.conj, (const T*)A.data + i*A.cs + j*A.rs, A.cs, A.rs,
                                B.alpha<T>(), B.conj,       (T*)B.data + i*B.cs + j*B.rs, B.cs, B.rs);
                        }
                    }
                }
            }
            else
            {
                len_type m_min, m_max, n_min, n_max;
                std::tie(m_min, m_max, std::ignore,
                         n_min, n_max, std::ignore) =
                    comm.distribute_over_threads_2d(A.m, A.n, MR, NR);

                if (B.alpha<T>() == T(0))
                {
                    for (len_type i = m_min;i < m_max;i += MR)
                    {
                        len_type i_loc = std::min(A.m-i, MR);
                        for (len_type j = n_min;j < n_max;j += NR)
                        {
                            len_type j_loc = std::min(A.n-j, NR);
                            cfg.trans_copy_ukr.call<T>(i_loc, j_loc,
                                A.alpha<T>(), A.conj, (const T*)A.data+ i*A.rs + n_min*A.cs+ m_min*A.rs + j*A.cs, A.rs, A.cs,
                                                            (T*)B.data+ i*B.rs + n_min*B.cs+ m_min*B.rs + j*B.cs, B.rs, B.cs);
                        }
                    }
                }
                else
                {
                    for (len_type i = m_min;i < m_max;i += MR)
                    {
                        len_type i_loc = std::min(A.m-i, MR);
                        for (len_type j = n_min;j < n_max;j += NR)
                        {
                            len_type j_loc = std::min(A.n-j, NR);
                            cfg.trans_add_ukr.call<T>(i_loc, j_loc,
                                A.alpha<T>(), A.conj, (const T*)A.data+ i*A.rs + n_min*A.cs+ m_min*A.rs + j*A.cs, A.rs, A.cs,
                                B.alpha<T>(), B.conj,       (T*)B.data+ i*B.rs + n_min*B.cs+ m_min*B.rs + j*B.cs, B.rs, B.cs);
                        }
                    }
                }
            }
        }
        /*
         * Otherwise, A can be added to B column-by-column or row-by-row
         */
        else
        {
            len_type m_min, m_max, n_min, n_max;
            std::tie(m_min, m_max, std::ignore,
                     n_min, n_max, std::ignore) =
                comm.distribute_over_threads_2d(A.m, A.n);

            /*
             * Add columns if column-major
             */
            if (B.rs < B.cs)
            {
                if (A.alpha<T>() == T(0))
                {
                    if (B.alpha<T>() == T(0))
                    {
                        for (len_type j = n_min;j < n_max;j++)
                        {
                            cfg.set_ukr.call<T>(m_max-m_min,
                                T(0), (T*)B.data+ m_min*B.rs + j*B.cs, B.rs);
                        }
                    }
                    else
                    {
                        for (len_type j = n_min;j < n_max;j++)
                        {
                            cfg.scale_ukr.call<T>(m_max-m_min,
                                B.alpha<T>(), B.conj, (T*)B.data+ m_min*B.rs + j*B.cs, B.rs);
                        }
                    }
                }
                else
                {
                    if (B.alpha<T>() == T(0))
                    {
                        for (len_type j = n_min;j < n_max;j++)
                        {
                            cfg.copy_ukr.call<T>(m_max-m_min,
                                A.alpha<T>(), A.conj, (const T*)A.data+ m_min*A.rs + j*A.cs, A.rs,
                                                            (T*)B.data+ m_min*B.rs + j*B.cs, B.rs);
                        }
                    }
                    else
                    {
                        for (len_type j = n_min;j < n_max;j++)
                        {
                            cfg.add_ukr.call<T>(m_max-m_min,
                                A.alpha<T>(), A.conj, (const T*)A.data+ m_min*A.rs + j*A.cs, A.rs,
                                B.alpha<T>(), B.conj,       (T*)B.data+ m_min*B.rs + j*B.cs, B.rs);
                        }
                    }
                }
            }
            /*
             * Add rows if row-major
             */
            else
            {
                if (A.alpha<T>() == T(0))
                {
                    if (B.alpha<T>() == T(0))
                    {
                        for (len_type i = m_min;i < m_max;i++)
                        {
                            cfg.set_ukr.call<T>(n_max-n_min,
                                T(0), (T*)B.data+ i*B.rs + n_min*B.cs, B.cs);
                        }
                    }
                    else
                    {
                        for (len_type i = m_min;i < m_max;i++)
                        {
                            cfg.scale_ukr.call<T>(n_max-n_min,
                                B.alpha<T>(), B.conj, (T*)B.data+ i*B.rs + n_min*B.cs, B.cs);
                        }
                    }
                }
                else
                {
                    if (B.alpha<T>() == T(0))
                    {
                        for (len_type i = m_min;i < m_max;i++)
                        {
                            cfg.copy_ukr.call<T>(n_max-n_min,
                                A.alpha<T>(), A.conj, (const T*)A.data+ i*A.rs + n_min*A.cs, A.cs,
                                                            (T*)B.data+ i*B.rs + n_min*B.cs, B.cs);
                        }
                    }
                    else
                    {
                        for (len_type i = m_min;i < m_max;i++)
                        {
                            cfg.add_ukr.call<T>(n_max-n_min,
                                A.alpha<T>(), A.conj, (const T*)A.data+ i*A.rs + n_min*A.cs, A.cs,
                                B.alpha<T>(), B.conj,       (T*)B.data+ i*B.rs + n_min*B.cs, B.cs);
                        }
                    }
                }
            }
        }

        B.alpha<T>() = T(1);
        B.conj = false;
    })
}

extern "C"
{

void tblis_matrix_add(const tblis_comm* comm, const tblis_config* cfg,
                      const tblis_matrix* A, tblis_matrix* B)
{
    parallelize_if(add_int, comm, get_config(cfg),
                   *A, *B);
}

}

}
