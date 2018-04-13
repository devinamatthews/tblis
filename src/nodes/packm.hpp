#ifndef _TBLIS_NODES_PACKM_HPP_
#define _TBLIS_NODES_PACKM_HPP_

#include "util/thread.h"
#include "util/basic_types.h"

#include "memory/alignment.hpp"
#include "memory/memory_pool.hpp"

#include "matrix/diag_scaled_matrix.hpp"
#include "matrix/scatter_matrix.hpp"
#include "matrix/block_scatter_matrix.hpp"

#include "configs/configs.hpp"

#include "iface/1m/reduce.h"

#define TBLIS_MAX_UNROLL 8

namespace tblis
{

template <typename T, int Mat>
struct pack_row_panel
{
    static constexpr bool Trans = Mat == matrix_constants::MAT_B;

    void operator()(const communicator& comm, const config& cfg,
                    matrix_view<T>& A, matrix_view<T>& Ap) const
    {
        const len_type MR = (!Trans ? cfg.gemm_mr.def<T>()
                                    : cfg.gemm_nr.def<T>());
        const len_type ME = (!Trans ? cfg.gemm_mr.extent<T>()
                                    : cfg.gemm_nr.extent<T>());
        const len_type KR = cfg.gemm_kr.def<T>();

        const len_type m_a = A.length( Trans);
        const len_type k_a = A.length(!Trans);
        const stride_type rs_a = A.stride( Trans);
        const stride_type cs_a = A.stride(!Trans);

        comm.distribute_over_threads({m_a, MR}, {k_a, KR},
        [&](len_type m_first, len_type m_last, len_type k_first, len_type k_last)
        {
            const T*  p_a =  A.data() +  m_first       *rs_a + k_first*cs_a;
                  T* p_ap = Ap.data() + (m_first/MR)*ME* k_a + k_first*  ME;

            for (len_type off_m = m_first;off_m < m_last;off_m += MR)
            {
                len_type m = std::min(MR, m_last-off_m);
                len_type k = k_last-k_first;

                if (!Trans)
                    cfg.pack_nn_mr_ukr.call<T>(m, k, p_a, rs_a, cs_a, p_ap);
                else
                    cfg.pack_nn_nr_ukr.call<T>(m, k, p_a, rs_a, cs_a, p_ap);

                p_a += m*rs_a;
                p_ap += ME*k_a;
            }
        });
    }

    void operator()(const communicator& comm, const config& cfg,
                    diag_scaled_matrix_view<T, !Trans>& A, matrix_view<T>& Ap) const
    {
        const len_type MR = (!Trans ? cfg.gemm_mr.def<T>()
                                    : cfg.gemm_nr.def<T>());
        const len_type ME = (!Trans ? cfg.gemm_mr.extent<T>()
                                    : cfg.gemm_nr.extent<T>());
        const len_type KR = cfg.gemm_kr.def<T>();

        const len_type m_a = A.length( Trans);
        const len_type k_a = A.length(!Trans);
        const stride_type rs_a = A.stride( Trans);
        const stride_type cs_a = A.stride(!Trans);
        const stride_type inc_d = A.diag_stride();

        comm.distribute_over_threads({m_a, MR}, {k_a, KR},
        [&](len_type m_first, len_type m_last, len_type k_first, len_type k_last)
        {
            const T*  p_a =  A.data() +  m_first       *rs_a + k_first* cs_a;
            const T*  p_d =  A.diag() +                        k_first*inc_d;
                  T* p_ap = Ap.data() + (m_first/MR)*ME* k_a + k_first*   ME;

            for (len_type off_m = m_first;off_m < m_last;off_m += MR)
            {
                len_type m = std::min(MR, m_last-off_m);
                len_type k = k_last-k_first;

                if (!Trans)
                    cfg.pack_nnd_mr_ukr.call<T>(m, k, p_a, rs_a, cs_a, p_d, inc_d, p_ap);
                else
                    cfg.pack_nnd_nr_ukr.call<T>(m, k, p_a, rs_a, cs_a, p_d, inc_d, p_ap);

                p_a += m*rs_a;
                p_ap += ME*k_a;
            }
        });
    }

    void operator()(const communicator& comm, const config& cfg,
                    diag_scaled_matrix_view<T, Trans>& A, matrix_view<T>& Ap) const
    {
        const len_type MR = (!Trans ? cfg.gemm_mr.def<T>()
                                    : cfg.gemm_nr.def<T>());
        const len_type ME = (!Trans ? cfg.gemm_mr.extent<T>()
                                    : cfg.gemm_nr.extent<T>());
        const len_type KR = cfg.gemm_kr.def<T>();

        const len_type m_a = A.length( Trans);
        const len_type k_a = A.length(!Trans);
        const stride_type rs_a = A.stride( Trans);
        const stride_type cs_a = A.stride(!Trans);
        const stride_type inc_d = A.diag_stride();

        comm.distribute_over_threads({m_a, MR}, {k_a, KR},
        [&](len_type m_first, len_type m_last, len_type k_first, len_type k_last)
        {
            const T*  p_a =  A.data() +  m_first       * rs_a + k_first*cs_a;
            const T*  p_d =  A.diag() +  m_first       *inc_d;
                  T* p_ap = Ap.data() + (m_first/MR)*ME*  k_a + k_first*  ME;

            for (len_type off_m = m_first;off_m < m_last;off_m += MR)
            {
                len_type m = std::min(MR, m_last-off_m);
                len_type k = k_last-k_first;

                if (!Trans)
                    cfg.pack_nne_mr_ukr.call<T>(m, k, p_a, rs_a, cs_a, p_d, inc_d, p_ap);
                else
                    cfg.pack_nne_nr_ukr.call<T>(m, k, p_a, rs_a, cs_a, p_d, inc_d, p_ap);

                p_d += m*inc_d;
                p_a += m*rs_a;
                p_ap += ME*k_a;
            }
        });
    }

    void operator()(const communicator& comm, const config& cfg,
                    const_scatter_matrix_view<T>& A, matrix_view<T>& Ap) const
    {
        const len_type MR = (!Trans ? cfg.gemm_mr.def<T>()
                                    : cfg.gemm_nr.def<T>());
        const len_type ME = (!Trans ? cfg.gemm_mr.extent<T>()
                                    : cfg.gemm_nr.extent<T>());
        const len_type KR = cfg.gemm_kr.def<T>();

        const len_type m_a = A.length( Trans);
        const len_type k_a = A.length(!Trans);
        const stride_type rs_a = A.stride( Trans);
        const stride_type cs_a = A.stride(!Trans);

        comm.distribute_over_threads({m_a, MR}, {k_a, KR},
        [&](len_type m_first, len_type m_last, len_type k_first, len_type k_last)
        {
            const stride_type* rscat_a = A.scatter( Trans) + m_first;
            const stride_type* cscat_a = A.scatter(!Trans) + k_first;
            const T*  p_a =  A.data() +  m_first       *rs_a + k_first*cs_a;
                  T* p_ap = Ap.data() + (m_first/MR)*ME* k_a + k_first*  ME;

            for (len_type off_m = m_first;off_m < m_last;off_m += MR)
            {
                len_type m = std::min(MR, m_last-off_m);
                len_type k = k_last-k_first;

                if (rs_a == 0 && cs_a == 0)
                {
                    if (!Trans)
                        cfg.pack_ss_mr_ukr.call<T>(m, k, p_a, rscat_a, cscat_a, p_ap);
                    else
                        cfg.pack_ss_nr_ukr.call<T>(m, k, p_a, rscat_a, cscat_a, p_ap);

                    rscat_a += m;
                }
                else if (rs_a == 0)
                {
                    if (!Trans)
                        cfg.pack_sn_mr_ukr.call<T>(m, k, p_a, rscat_a, cs_a, p_ap);
                    else
                        cfg.pack_sn_nr_ukr.call<T>(m, k, p_a, rscat_a, cs_a, p_ap);

                    rscat_a += m;
                }
                else if (cs_a == 0)
                {
                    if (!Trans)
                        cfg.pack_ns_mr_ukr.call<T>(m, k, p_a, rs_a, cscat_a, p_ap);
                    else
                        cfg.pack_ns_nr_ukr.call<T>(m, k, p_a, rs_a, cscat_a, p_ap);

                    p_a += m*rs_a;
                }
                else
                {
                    if (!Trans)
                        cfg.pack_nn_mr_ukr.call<T>(m, k, p_a, rs_a, cs_a, p_ap);
                    else
                        cfg.pack_nn_nr_ukr.call<T>(m, k, p_a, rs_a, cs_a, p_ap);

                    p_a += m*rs_a;
                }

                p_ap += ME*k_a;
            }
        });
    }

    void operator()(const communicator& comm, const config& cfg,
                    block_scatter_matrix<T> A, matrix_view<T>& Ap) const
    {
        const len_type MR = (!Trans ? cfg.gemm_mr.def<T>()
                                    : cfg.gemm_nr.def<T>());
        const len_type ME = (!Trans ? cfg.gemm_mr.extent<T>()
                                    : cfg.gemm_nr.extent<T>());
        const len_type KR = cfg.gemm_kr.def<T>();

        TBLIS_ASSERT(A.block_size(0) == (!Trans ? MR : KR));
        TBLIS_ASSERT(A.block_size(1) == (!Trans ? KR : MR));

        const len_type m_a = A.length( Trans);
        const len_type k_a = A.length(!Trans);

        comm.distribute_over_threads({m_a, MR}, {k_a, KR},
        [&](len_type m_first, len_type m_last, len_type k_first, len_type k_last)
        {
            const T* p_a = A.raw_data();
            T* p_ap = Ap.data() + (m_first/MR)*ME*k_a + k_first*ME;
            const stride_type* cscat_a = A.scatter(!Trans) + k_first;
            const stride_type* cbs_a = A.block_scatter(!Trans) + k_first/KR;
            const stride_type* rscat_a = A.scatter(Trans) + m_first;
            const stride_type* rbs_a = A.block_scatter(Trans) + m_first/MR;

            len_type off_m = m_first;
            while (off_m < m_last)
            {
                len_type m = std::min(MR, m_last-off_m);
                len_type k = k_last-k_first;
                stride_type rs_a = *rbs_a;

                if (rs_a == 0)
                {
                    if (!Trans)
                        cfg.pack_sb_mr_ukr.call<T>(m, k, p_a, rscat_a, cscat_a, cbs_a, p_ap);
                    else
                        cfg.pack_sb_nr_ukr.call<T>(m, k, p_a, rscat_a, cscat_a, cbs_a, p_ap);
                }
                else
                {
                    if (!Trans)
                        cfg.pack_nb_mr_ukr.call<T>(m, k, p_a+rscat_a[0], rs_a, cscat_a, cbs_a, p_ap);
                    else
                        cfg.pack_nb_nr_ukr.call<T>(m, k, p_a+rscat_a[0], rs_a, cscat_a, cbs_a, p_ap);
                }

                p_ap += ME*k_a;
                off_m += MR;
                rscat_a += MR;
                rbs_a++;
            }
        });
    }
};

template <typename Pack, int Mat> struct pack_and_run;

template <typename Pack>
struct pack_and_run<Pack, matrix_constants::MAT_A>
{
    template <typename Run, typename T, typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixP>
    pack_and_run(Run& run, const communicator& comm, const config& cfg,
                 T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C, MatrixP& P)
    {
        Pack()(comm, cfg, A, P);
        comm.barrier();
        run(comm, cfg, alpha, P, B, beta, C);
        comm.barrier();
    }
};

template <typename Pack>
struct pack_and_run<Pack, matrix_constants::MAT_B>
{
    template <typename Run, typename T, typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixP>
    pack_and_run(Run& run, const communicator& comm, const config& cfg,
                 T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C, MatrixP& P)
    {
        Pack()(comm, cfg, B, P);
        comm.barrier();
        run(comm, cfg, alpha, A, P, beta, C);
        comm.barrier();
    }
};

template <int Mat, blocksize config::*BS, MemoryPool& Pool, typename Child>
struct pack
{
    Child child;
    MemoryPool::Block pack_buffer;
    void* pack_ptr = nullptr;

    pack() {}

    pack(const pack& other)
    : child(other.child) {}

    template <typename T, typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        using namespace matrix_constants;

        constexpr bool Trans = (Mat == MAT_B);
        const blocksize& M = cfg.*BS;
        const len_type MR = M.def<T>();
        const len_type ME = M.extent<T>();

        len_type m_p = ceil_div(!Trans ? A.length(0) : B.length(1), MR)*ME;
        len_type k_p =         (!Trans ? A.length(1) : B.length(0));

        if (!pack_ptr)
        {
            if (comm.master())
            {
                pack_buffer = Pool.allocate<T>(m_p*k_p+std::max(m_p,k_p)*TBLIS_MAX_UNROLL);
                pack_ptr = pack_buffer.get();
            }

            comm.broadcast_value(pack_ptr);
        }

        matrix_view<T> P({!Trans ? m_p : k_p,
                          !Trans ? k_p : m_p},
                         static_cast<T*>(pack_ptr),
                         {!Trans? k_p :   1,
                          !Trans?   1 : k_p});

        typedef pack_row_panel<T, Mat> Pack;
        pack_and_run<Pack, Mat>(child, comm, cfg, alpha, A, B, beta, C, P);
    }
};

template <MemoryPool& Pool, typename Child>
using pack_a = pack<matrix_constants::MAT_A, &config::gemm_mr, Pool, Child>;

template <MemoryPool& Pool, typename Child>
using pack_b = pack<matrix_constants::MAT_B, &config::gemm_nr, Pool, Child>;

}

#endif
