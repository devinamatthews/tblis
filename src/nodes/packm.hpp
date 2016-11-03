#ifndef _TBLIS_NODES_PACKM_HPP_
#define _TBLIS_NODES_PACKM_HPP_

#include "util/thread.h"
#include "util/basic_types.h"

#include "memory/alignment.hpp"
#include "memory/memory_pool.hpp"

#include "matrix/scatter_matrix.hpp"
#include "matrix/block_scatter_matrix.hpp"

#include "configs/configs.hpp"

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

        len_type m_a = A.length( Trans);
        len_type k_a = A.length(!Trans);
        stride_type rs_a = A.stride( Trans);
        stride_type cs_a = A.stride(!Trans);
        const T* p_a = A.data();
        T* p_ap = Ap.data();

        //printf("packing %c\n", "AB"[Mat]);
        //printf("m_a/k_a/rs_a/cs_a: %ld/%ld/%ld/%ld\n", m_a, k_a, rs_a, cs_a);
        //printf("M: %p %ld %ld %ld %ld\n", A.data(), A.length(0), A.length(1), A.stride(0), A.stride(1));
        //printf("Mp: %p %ld %ld %ld %ld\n", Ap.data(), Ap.length(0), Ap.length(1), Ap.stride(0), Ap.stride(1));
        //printf("norm before: %.15f\n", reduce(REDUCE_NORM_2, A).first);

        len_type m_first, m_last, k_first, k_last;
        std::tie(m_first, m_last, std::ignore,
                 k_first, k_last, std::ignore) =
            comm.distribute_over_threads_2d(m_a, k_a, MR, KR);

        p_a += m_first*rs_a + k_first*cs_a;
        p_ap += m_first*k_a + k_first*ME;

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

        //printf("norm after: %.15f\n", reduce(REDUCE_NORM_2, Ap).first);
    }

    void operator()(const communicator& comm, const config& cfg,
                    const_scatter_matrix_view<T>& A, matrix_view<T>& Ap) const
    {
        const len_type MR = (!Trans ? cfg.gemm_mr.def<T>()
                                    : cfg.gemm_nr.def<T>());
        const len_type ME = (!Trans ? cfg.gemm_mr.extent<T>()
                                    : cfg.gemm_nr.extent<T>());
        const len_type KR = cfg.gemm_kr.def<T>();

        len_type m_a = A.length( Trans);
        len_type k_a = A.length(!Trans);
        stride_type rs_a = A.stride( Trans);
        stride_type cs_a = A.stride(!Trans);
        const stride_type* rscat_a = A.scatter( Trans);
        const stride_type* cscat_a = A.scatter(!Trans);
        const T* p_a = A.data();
        T* p_ap = Ap.data();

        len_type m_first, m_last, k_first, k_last;
        std::tie(m_first, m_last, std::ignore,
                 k_first, k_last, std::ignore) =
            comm.distribute_over_threads_2d(m_a, k_a, MR, KR);

        p_a += m_first*rs_a + k_first*cs_a;
        rscat_a += m_first;
        cscat_a += k_first;
        p_ap += m_first*k_a + k_first*ME;

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
    }

    template <len_type MB, len_type NB>
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

        len_type m_a = A.length( Trans);
        len_type k_a = A.length(!Trans);
        T* p_ap = Ap.data();

        len_type m_first, m_last, k_first, k_last;
        std::tie(m_first, m_last, std::ignore,
                 k_first, k_last, std::ignore) =
            comm.distribute_over_threads_2d(m_a, k_a, MR, KR);

        p_ap += m_first*k_a + k_first*ME;

        len_type off_m = m_first;

        A.length(Trans, MR);
        A.shift(Trans, off_m);

        while (off_m < m_last)
        {
            stride_type rs_a = A.stride(Trans);
            const stride_type* rscat_a = A.scatter(Trans);
            const stride_type* cscat_a = A.scatter(!Trans);
            const stride_type* cbs_a = A.block_scatter(!Trans);
            const T* p_a = A.data();

            len_type m = std::min(MR, m_last-off_m);
            len_type k = k_last-k_first;

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
                    cfg.pack_nb_mr_ukr.call<T>(m, k, p_a, rs_a, cscat_a, cbs_a, p_ap);
                else
                    cfg.pack_nb_nr_ukr.call<T>(m, k, p_a, rs_a, cscat_a, cbs_a, p_ap);
            }

            p_ap += ME*k_a;
            A.shift_block(Trans, 1);
            off_m += MR;
        }

        A.shift(Trans, -off_m);
        A.length(Trans, m_a);
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

template <int Mat, MemoryPool& Pool, typename Child>
struct pack
{
    Child child;
    MemoryPool::Block pack_buffer;
    void* pack_ptr = nullptr;

    template <typename T, typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        using namespace matrix_constants;

        constexpr bool Trans = (Mat == MAT_B);
        const len_type MR = (!Trans ? cfg.gemm_mr.def<T>()
                                    : cfg.gemm_nr.def<T>());
        const len_type ME = (!Trans ? cfg.gemm_mr.extent<T>()
                                    : cfg.gemm_nr.extent<T>());

        len_type m_p = ceil_div(!Trans ? A.length(0) : B.length(1), MR)*ME;
        len_type k_p =         (!Trans ? A.length(1) : B.length(0));

        if (!pack_ptr)
        {
            if (comm.master())
            {
                pack_buffer = Pool.allocate<T>(m_p*k_p+std::max(m_p,k_p)*TBLIS_MAX_UNROLL);
                pack_ptr = pack_buffer.get();
            }

            comm.broadcast(pack_ptr);
        }

        matrix_view<T> P({!Trans ? m_p : k_p,
                          !Trans ? k_p : m_p},
                         (T*)pack_ptr,
                         {!Trans? k_p :   1,
                          !Trans?   1 : k_p});

        typedef pack_row_panel<T, Mat> Pack;
        pack_and_run<Pack, Mat>(child, comm, cfg, alpha, A, B, beta, C, P);
    }
};

template <MemoryPool& Pool, typename Child>
using pack_a = pack<matrix_constants::MAT_A, Pool, Child>;

template <MemoryPool& Pool, typename Child>
using pack_b = pack<matrix_constants::MAT_B, Pool, Child>;

}

#endif
