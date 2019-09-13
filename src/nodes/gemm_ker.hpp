#ifndef _TBLIS_NODES_GEMM_MKR_HPP_
#define _TBLIS_NODES_GEMM_MKR_HPP_

#include "util/basic_types.h"
#include "util/thread.h"

#include "configs/configs.hpp"

#include "gemm_ukr.hpp"

namespace tblis
{

struct gemm_macro_kernel
{
    int distribute_m = 1;
    int distribute_n = 1;
    int chunk_m = 1;
    int chunk_n = 1;

    template <typename T>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha, normal_matrix<T>& A,
                             normal_matrix<T>& B,
                    T  beta, normal_matrix<T>& C) const
    {
        const len_type MR = cfg.gemm_mr.def<T>();
        const len_type NR = cfg.gemm_nr.def<T>();
        const len_type ME = cfg.gemm_mr.extent<T>();
        const len_type NE = cfg.gemm_nr.extent<T>();
        const bool row_major = cfg.gemm_row_major.value<T>();
        const bool flip_ukr = cfg.gemm_flip_ukr.value<T>();
        const len_type rs_ab = (row_major ? NR : 1);
        const len_type cs_ab = (row_major ? 1 : MR);

        len_type m = C.length(0);
        len_type n = C.length(1);
        len_type k = A.length(1);
        stride_type rs_c = C.stride(0);
        stride_type cs_c = C.stride(1);

        comm.distribute_over_threads({(m+MR-1)/MR}, {(n+NR-1)/NR},
        [&](len_type m_first, len_type m_last, len_type n_first, len_type n_last)
        {
            T p_ab[512] __attribute__((aligned(64)));
            static constexpr T zero = T(0);

            for (len_type n_off = n_first;n_off < n_last;n_off++)
            {
                len_type n_loc = std::min(NR, n-n_off*NR);

                for (len_type m_off = m_first;m_off < m_last;m_off++)
                {
                    len_type m_loc = std::min(MR, m-m_off*MR);

                    const T* p_a = A.data() + m_off*ME*k;
                    const T* p_b = B.data() + n_off*NE*k;
                          T* p_c = C.data() + m_off*MR*rs_c + n_off*NR*cs_c;

                    if (m_loc == MR && n_loc == NR)
                    {
                        if (flip_ukr)
                        {
                            cfg.gemm_ukr.call<T>(k, &alpha, p_b, p_a,
                                                 &beta, p_c, cs_c, rs_c);
                        }
                        else
                        {
                            cfg.gemm_ukr.call<T>(k, &alpha, p_a, p_b,
                                                 &beta, p_c, rs_c, cs_c);
                        }
                    }
                    else
                    {
                        if (flip_ukr)
                        {
                            cfg.gemm_ukr.call<T>(k, &alpha, p_b, p_a,
                                                 &zero, &p_ab[0], cs_ab, rs_ab);
                        }
                        else
                        {
                            cfg.gemm_ukr.call<T>(k, &alpha, p_a, p_b,
                                                 &zero, &p_ab[0], rs_ab, cs_ab);
                        }

                        accum_utile(m_loc, n_loc, p_ab, rs_ab, cs_ab,
                                    beta, p_c, rs_c, cs_c);
                    }
                }
            }
        });
    }
};

}

#endif
