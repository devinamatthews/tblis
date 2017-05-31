#ifndef _TBLIS_NODES_PACKM_HPP_
#define _TBLIS_NODES_PACKM_HPP_

#include "util/thread.h"
#include "util/basic_types.h"

#include "memory/alignment.hpp"
#include "memory/memory_pool.hpp"

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

    template <typename Communicator>
    void operator()(Communicator& comm, const config& cfg,
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

#if TBLIS_ENABLE_TBB

        tbb::task_group tg;

        // Decompose into coarse-grained blocks
        // TODO: May need tuning
        const len_type MR_BS = 2*MR;
        const len_type KR_BS = 8*KR;
        unsigned nthread = ((m_a+MR_BS-1)/MR_BS) * ((k_a+KR_BS-1)/KR_BS);

        for (unsigned tid = 0;tid < nthread;tid++)
        {

        tg.run([&,tid]
        {

#else //TBLIS_ENABLE_TBB

        unsigned nthread = comm.num_threads();
        unsigned tid = comm.thread_num();

#endif //TBLIS_ENABLE_TBB

        len_type m_first, m_last, k_first, k_last;
        std::tie(m_first, m_last, std::ignore,
                 k_first, k_last, std::ignore) =
            communicator::distribute_2d(nthread, tid, m_a, k_a, MR, KR);

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

#if TBLIS_ENABLE_TBB

        });

        }

        tg.wait();

#endif //TBLIS_ENABLE_TBB

    }

    template <typename Communicator>
    void operator()(Communicator& comm, const config& cfg,
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

#if TBLIS_ENABLE_TBB

        tbb::task_group tg;

        // Decompose into coarse-grained blocks
        // TODO: May need tuning
        const len_type MR_BS = 2*MR;
        const len_type KR_BS = 8*KR;
        unsigned nthread = ((m_a+MR_BS-1)/MR_BS) * ((k_a+KR_BS-1)/KR_BS);

        for (unsigned tid = 0;tid < nthread;tid++)
        {

        tg.run([&,tid]
        {

#else //TBLIS_ENABLE_TBB

        unsigned nthread = comm.num_threads();
        unsigned tid = comm.thread_num();

#endif //TBLIS_ENABLE_TBB

        len_type m_first, m_last, k_first, k_last;
        std::tie(m_first, m_last, std::ignore,
                 k_first, k_last, std::ignore) =
            communicator::distribute_2d(nthread, tid, m_a, k_a, MR, KR);

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

#if TBLIS_ENABLE_TBB

        });

        }

        tg.wait();

#endif //TBLIS_ENABLE_TBB
    }

    template <typename Communicator>
    void operator()(Communicator& comm, const config& cfg,
                    block_scatter_matrix<T> A_, matrix_view<T>& Ap) const
    {
        const len_type MR = (!Trans ? cfg.gemm_mr.def<T>()
                                    : cfg.gemm_nr.def<T>());
        const len_type ME = (!Trans ? cfg.gemm_mr.extent<T>()
                                    : cfg.gemm_nr.extent<T>());
        const len_type KR = cfg.gemm_kr.def<T>();

        TBLIS_ASSERT(A.block_size(0) == (!Trans ? MR : KR));
        TBLIS_ASSERT(A.block_size(1) == (!Trans ? KR : MR));

        const len_type m_a = A_.length( Trans);
        const len_type k_a = A_.length(!Trans);

#if TBLIS_ENABLE_TBB

        tbb::task_group tg;

        // Decompose into coarse-grained blocks
        // TODO: May need tuning
        const len_type MR_BS = 2*MR;
        const len_type KR_BS = 8*KR;
        unsigned nthread = ((m_a+MR_BS-1)/MR_BS) * ((k_a+KR_BS-1)/KR_BS);

        for (unsigned tid = 0;tid < nthread;tid++)
        {

        tg.run([&,tid]
        {

        auto A = A_;

#else //TBLIS_ENABLE_TBB

        unsigned nthread = comm.num_threads();
        unsigned tid = comm.thread_num();

        auto& A = A_;

#endif //TBLIS_ENABLE_TBB

        len_type m_first, m_last, k_first, k_last;
        std::tie(m_first, m_last, std::ignore,
                 k_first, k_last, std::ignore) =
            communicator::distribute_2d(nthread, tid, m_a, k_a, MR, KR);

        len_type off_m = m_first;

        A.length(Trans, MR);
        A.shift(Trans, off_m);

        const T* p_a = A.raw_data();
        T* p_ap = Ap.data() + (m_first/MR)*ME*k_a + k_first*ME;
        const stride_type* cscat_a = A.scatter(!Trans) + k_first;
        const stride_type* cbs_a = A.block_scatter(!Trans) + k_first/KR;

        while (off_m < m_last)
        {
            stride_type rs_a = A.stride(Trans);
            const stride_type* rscat_a = A.scatter(Trans);

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
                    cfg.pack_nb_mr_ukr.call<T>(m, k, p_a+rscat_a[0], rs_a, cscat_a, cbs_a, p_ap);
                else
                    cfg.pack_nb_nr_ukr.call<T>(m, k, p_a+rscat_a[0], rs_a, cscat_a, cbs_a, p_ap);
            }

            p_ap += ME*k_a;
            A.shift_block(Trans, 1);
            off_m += MR;
        }

        A.shift(Trans, -off_m);
        A.length(Trans, m_a);

#if TBLIS_ENABLE_TBB

        });

        }

        tg.wait();

#endif //TBLIS_ENABLE_TBB
    }
};

template <typename Pack, int Mat> struct pack_and_run;

template <typename Pack>
struct pack_and_run<Pack, matrix_constants::MAT_A>
{
    template <typename Run, typename T, typename Communicator, typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixP>
    pack_and_run(Run& run, Communicator& comm, const config& cfg,
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
    template <typename Run, typename T, typename Communicator, typename MatrixA, typename MatrixB, typename MatrixC, typename MatrixP>
    pack_and_run(Run& run, Communicator& comm, const config& cfg,
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

    template <typename T, typename Communicator, typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(Communicator& comm, const config& cfg,
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

            comm.broadcast(pack_ptr);
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
