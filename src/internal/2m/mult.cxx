#include "mult.hpp"

namespace tblis
{

namespace internal
{

template <typename T>
void mult(const communicator& comm, const config& cfg,
          len_type m, len_type n,
          T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A,
                   bool conj_B, const T* B, stride_type inc_B,
          T  beta, bool conj_C,       T* C, stride_type inc_C)
{
    if (comm.master()) flops += 2*m*n;

    if (rs_A <= cs_A)
    {
        const len_type NF = cfg.addf_nf.def<T>();

        comm.distribute_over_threads(m,
        [&](len_type m_min, len_type m_max)
        {
            auto local_beta = beta;
            auto local_conj_C = conj_C;

            const T* As[16];

            for (len_type j = 0;j < n;j += NF)
            {
                for (len_type k = 0;k < NF;k++)
                    As[k] = A + m_min*rs_A + (j+k)*cs_A;

                cfg.addf_sum_ukr.call<T>(m_max-m_min, std::min(NF, n-j),
                                         alpha, conj_A, As, rs_A,
                                                conj_B, B + j*inc_B, inc_B,
                                         local_beta, local_conj_C, C + m_min*inc_C, inc_C);

                local_beta = T(1);
                local_conj_C = false;
            }
        });
    }
    else
    {
        const len_type NF = cfg.dotf_nf.def<T>();

        comm.distribute_over_threads({m, NF},
        [&](len_type m_min, len_type m_max)
        {
            for (len_type i = m_min;i < m_max;i += NF)
            {
                cfg.dotf_ukr.call<T>(std::min(NF, m_max-i), n,
                                     alpha, conj_A, A + i*rs_A, rs_A, cs_A,
                                            conj_B, B, inc_B,
                                      beta, conj_C, C + i*inc_C, inc_C);
            }
        });
    }

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, const config& cfg, \
                   len_type m, len_type n, \
                   T alpha, bool conj_A, const T* A, stride_type rs_A, stride_type cs_A, \
                            bool conj_B, const T* B, stride_type inc_B, \
                   T  beta, bool conj_C,       T* C, stride_type inc_C);
#include "configs/foreach_type.h"

template <typename T>
void mult(const communicator& comm, const config& cfg,
          len_type m, len_type n,
          T alpha, bool conj_A, const T* A, stride_type inc_A,
                   bool conj_B, const T* B, stride_type inc_B,
          T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C)
{
    if (comm.master()) flops += 2*m*n;

    if (rs_C > cs_C)
    {
        std::swap(m, n);
        std::swap(conj_A, conj_B);
        std::swap(A, B);
        std::swap(inc_A, inc_B);
        std::swap(rs_C, cs_C);
    }

    const len_type NF = cfg.addf_nf.def<T>();

    comm.distribute_over_threads(m, n,
    [&](len_type m_min, len_type m_max, len_type n_min, len_type n_max)
    {
        T* Cs[16];

        for (len_type j = n_min;j < n_max;j += NF)
        {
            for (len_type k = 0;k < NF;k++)
                Cs[k] = C + m_min*rs_C + (j+k)*cs_C;

            cfg.addf_rep_ukr.call<T>(m_max-m_min, std::min(NF, n_max-j),
                                     alpha, conj_A, A + m_min*inc_A, inc_A,
                                            conj_B, B + j*inc_B, inc_B,
                                      beta, conj_C, Cs, rs_C);
        }
    });

    comm.barrier();
}

#define FOREACH_TYPE(T) \
template void mult(const communicator& comm, const config& cfg, \
                   len_type m, len_type n, \
                   T alpha, bool conj_A, const T* A, stride_type inc_A, \
                            bool conj_B, const T* B, stride_type inc_B, \
                   T  beta, bool conj_C,       T* C, stride_type rs_C, stride_type cs_C);
#include "configs/foreach_type.h"

}
}
