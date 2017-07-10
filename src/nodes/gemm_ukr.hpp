#ifndef _TBLIS_NODES_GEMM_UKR_HPP_
#define _TBLIS_NODES_GEMM_UKR_HPP_

#include "util/basic_types.h"
#include "util/thread.h"

#include "configs/configs.hpp"

namespace tblis
{

template <typename T>
void accum_utile(len_type m, len_type n,
                 const T* TBLIS_RESTRICT p_ab, stride_type rs_ab, stride_type cs_ab,
                 T beta, T* TBLIS_RESTRICT p_c, stride_type rs_c, stride_type cs_c)
{
    if (beta == T(0))
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + j*cs_c] = p_ab[i*rs_ab + j*cs_ab];
            }
        }
    }
    else
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + j*cs_c] = p_ab[i*rs_ab + j*cs_ab] + beta*p_c[i*rs_c + j*cs_c];
            }
        }
    }
}

template <typename T>
void accum_utile(len_type m, len_type n,
                 const T* TBLIS_RESTRICT p_ab, stride_type rs_ab, stride_type cs_ab,
                 T beta, T* TBLIS_RESTRICT p_c,
                 const stride_type* TBLIS_RESTRICT rs_c, stride_type cs_c)
{
    if (beta == T(0))
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + j*cs_c] = p_ab[i*rs_ab + j*cs_ab];
            }
        }
    }
    else
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + j*cs_c] = p_ab[i*rs_ab + j*cs_ab] + beta*p_c[rs_c[i] + j*cs_c];
            }
        }
    }
}

template <typename T>
void accum_utile(len_type m, len_type n,
                 const T* TBLIS_RESTRICT p_ab, stride_type rs_ab, stride_type cs_ab,
                 T beta, T* TBLIS_RESTRICT p_c,
                 stride_type rs_c, const stride_type* TBLIS_RESTRICT cs_c)
{
    if (beta == T(0))
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + cs_c[j]] = p_ab[i*rs_ab + j*cs_ab];
            }
        }
    }
    else
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[i*rs_c + cs_c[j]] = p_ab[i*rs_ab + j*cs_ab] + beta*p_c[i*rs_c + cs_c[j]];
            }
        }
    }
}

template <typename T>
void accum_utile(len_type m, len_type n,
                 const T* TBLIS_RESTRICT p_ab, stride_type rs_ab, stride_type cs_ab,
                 T beta, T* TBLIS_RESTRICT p_c,
                 const stride_type* TBLIS_RESTRICT rs_c,
                 const stride_type* TBLIS_RESTRICT cs_c)
{
    if (beta == T(0))
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + cs_c[j]] = p_ab[i*rs_ab + j*cs_ab];
            }
        }
    }
    else
    {
        for (len_type j = 0;j < n;j++)
        {
            for (len_type i = 0;i < m;i++)
            {
                p_c[rs_c[i] + cs_c[j]] = p_ab[i*rs_ab + j*cs_ab] + beta*p_c[rs_c[i] + cs_c[j]];
            }
        }
    }
}

struct gemm_micro_kernel
{
    template <typename T>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha, matrix_view<T>& A,
                             matrix_view<T>& B,
                    T  beta, matrix_view<T>& C) const
    {
        (void)comm;

        const len_type MR = cfg.gemm_mr.def<T>();
        const len_type NR = cfg.gemm_nr.def<T>();
        const bool row_major = cfg.gemm_row_major.value<T>();
        const len_type rs_ab = (row_major ? NR : 1);
        const len_type cs_ab = (row_major ? 1 : MR);

        const T* p_a = A.data();
        const T* p_b = B.data();
              T* p_c = C.data();

        len_type m = C.length(0);
        len_type n = C.length(1);
        len_type k = A.length(1);
        stride_type rs_c = C.stride(0);
        stride_type cs_c = C.stride(1);

        if (m == MR && n == NR)
        {
            cfg.gemm_ukr.call<T>(k, &alpha, p_a, p_b,
                                 &beta, p_c, rs_c, cs_c);
        }
        else
        {
            T p_ab[512] __attribute__((aligned(64)));
            static constexpr T zero = T(0);

            cfg.gemm_ukr.call<T>(k, &alpha, p_a, p_b,
                                 &zero, &p_ab[0], rs_ab, cs_ab);

            accum_utile(m, n, p_ab, rs_ab, cs_ab,
                        beta, p_c, rs_c, cs_c);
        }
    }

    template <typename T>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha,         matrix_view<T>& A,
                                     matrix_view<T>& B,
                    T  beta, scatter_matrix_view<T>& C) const
    {
        (void)comm;

        const len_type MR = cfg.gemm_mr.def<T>();
        const len_type NR = cfg.gemm_nr.def<T>();
        const bool row_major = cfg.gemm_row_major.value<T>();
        const len_type rs_ab = (row_major ? NR : 1);
        const len_type cs_ab = (row_major ? 1 : MR);

        const T* p_a = A.data();
        const T* p_b = B.data();
              T* p_c = C.data();

        len_type m = C.length(0);
        len_type n = C.length(1);
        len_type k = A.length(1);
        stride_type rs_c = C.stride(0);
        stride_type cs_c = C.stride(1);
        const stride_type* rscat_c = C.scatter(0);
        const stride_type* cscat_c = C.scatter(1);

        if (m == MR && n == NR && rs_c != 0 && cs_c != 0)
        {
            cfg.gemm_ukr.call<T>(k, &alpha, p_a, p_b,
                                 &beta, p_c, rs_c, cs_c);
        }
        else
        {
            T p_ab[512] __attribute__((aligned(64)));
            static constexpr T zero = T(0);

            cfg.gemm_ukr.call<T>(k, &alpha, p_a, p_b,
                                 &zero, &p_ab[0], rs_ab, cs_ab);

            if (rs_c == 0 && cs_c == 0)
            {
                accum_utile(m, n, p_ab, rs_ab, cs_ab,
                            beta, p_c, rscat_c, cscat_c);
            }
            else if (rs_c == 0)
            {
                accum_utile(m, n, p_ab, rs_ab, cs_ab,
                            beta, p_c, rscat_c, cs_c);
            }
            else if (cs_c == 0)
            {
                accum_utile(m, n, p_ab, rs_ab, cs_ab,
                            beta, p_c, rs_c, cscat_c);
            }
            else
            {
                accum_utile(m, n, p_ab, rs_ab, cs_ab,
                            beta, p_c, rs_c, cs_c);
            }
        }
    }

    template <typename T>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha,          matrix_view<T>& A,
                                      matrix_view<T>& B,
                    T  beta, block_scatter_matrix<T>& C) const
    {
        (void)comm;

        const len_type MR = cfg.gemm_mr.def<T>();
        const len_type NR = cfg.gemm_nr.def<T>();
        const bool row_major = cfg.gemm_row_major.value<T>();
        const len_type rs_ab = (row_major ? NR : 1);
        const len_type cs_ab = (row_major ? 1 : MR);

        const T* p_a = A.data();
        const T* p_b = B.data();
              T* p_c = C.data();

        TBLIS_ASSERT(C.block_size(0) == MR &&
                     C.block_size(1) == NR);

        len_type m = C.length(0);
        len_type n = C.length(1);
        len_type k = A.length(1);
        stride_type rs_c = C.stride(0);
        stride_type cs_c = C.stride(1);
        const stride_type* rscat_c = C.scatter(0);
        const stride_type* cscat_c = C.scatter(1);

        if (m == MR && n == NR && rs_c != 0 && cs_c != 0)
        {
            cfg.gemm_ukr.call<T>(k, &alpha, p_a, p_b,
                                 &beta, p_c, rs_c, cs_c);
        }
        else
        {
            T p_ab[512] __attribute__((aligned(64)));
            static constexpr T zero = T(0);

            cfg.gemm_ukr.call<T>(k, &alpha, p_a, p_b,
                                 &zero, &p_ab[0], rs_ab, cs_ab);

            if (rs_c == 0 && cs_c == 0)
            {
                accum_utile(m, n, p_ab, rs_ab, cs_ab,
                            beta, p_c, rscat_c, cscat_c);
            }
            else if (rs_c == 0)
            {
                accum_utile(m, n, p_ab, rs_ab, cs_ab,
                            beta, p_c, rscat_c, cs_c);
            }
            else if (cs_c == 0)
            {
                accum_utile(m, n, p_ab, rs_ab, cs_ab,
                            beta, p_c, rs_c, cscat_c);
            }
            else
            {
                accum_utile(m, n, p_ab, rs_ab, cs_ab,
                            beta, p_c, rs_c, cs_c);
            }
        }
    }
};

}

#endif
