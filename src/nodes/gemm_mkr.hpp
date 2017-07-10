#ifndef _TBLIS_NODES_GEMM_MKR_HPP_
#define _TBLIS_NODES_GEMM_MKR_HPP_

#include "util/basic_types.h"
#include "util/thread.h"

#include "configs/configs.hpp"

#include "gemm_ukr.hpp"

namespace tblis
{

/*
#if TBLIS_ENABLE_TBB

struct gemm_task_graph
{
    struct lazy_task
    {
        typedef tbb::flow::continue_node<tbb::flow::continue_msg> task_type;

        alignas(task_type) char data[sizeof(task_type)];
        bool exists = false;

        ~lazy_task()
        {
            if (exists) task().~task_type();
        }

        template <typename Body>
        void create(tbb::flow::graph& g, const Body& body)
        {
            TBLIS_ASSERT(!exists);
            new (&task()) task_type(g, [body](tbb::flow::continue_msg&)
            {
                body();
            });
            exists = true;
        }

        task_type& task()
        {
            return reinterpret_cast<task_type&>(data);
        }
    };

    tbb::flow::graph g;
    unsigned phase = 0;
    unsigned num_phases;
    unsigned pack_per_phase;
    std::vector<lazy_task> pack;
    unsigned ukr_per_phase;
    std::vector<lazy_task> ukr;

    void barrier() const {};

    template <typename T>
    void broadcast(const T&) const {};
};

#endif
*/

struct gemm_macro_kernel
{
    int distribute_m = 1;
    int distribute_n = 1;

    template <typename T>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha, matrix_view<T>& A,
                             matrix_view<T>& B,
                    T  beta, matrix_view<T>& C) const
    {
        const len_type MR = cfg.gemm_mr.def<T>();
        const len_type NR = cfg.gemm_nr.def<T>();
        const len_type ME = cfg.gemm_mr.extent<T>();
        const len_type NE = cfg.gemm_nr.extent<T>();
        const bool row_major = cfg.gemm_row_major.value<T>();
        const len_type rs_ab = (row_major ? NR : 1);
        const len_type cs_ab = (row_major ? 1 : MR);

        len_type m = C.length(0);
        len_type n = C.length(1);
        len_type k = A.length(1);
        stride_type rs_c = C.stride(0);
        stride_type cs_c = C.stride(1);

#if TBLIS_ENABLE_TBB

        tbb::task_group tg;

        for (unsigned tid_n = 0;tid_n < distribute_n;tid_n++)
        {
        for (unsigned tid_m = 0;tid_m < distribute_m;tid_m++)
        {

        tg.run([&,tid_m,tid_n]
        {

#else

        TBLIS_ASSERT(comm.num_threads() == distribute_m*distribute_n);

        unsigned tid = comm.thread_num();
        unsigned tid_m = tid%distribute_m;
        unsigned tid_n = tid/distribute_m;

#endif

        len_type m_first, m_last;
        std::tie(m_first, m_last, std::ignore) =
            communicator::distribute(distribute_m, tid_m, (m+MR-1)/MR);

        len_type n_first, n_last;
        std::tie(n_first, n_last, std::ignore) =
            communicator::distribute(distribute_n, tid_n, (n+NR-1)/NR);

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
                    cfg.gemm_ukr.call<T>(k, &alpha, p_a, p_b,
                                         &beta, p_c, rs_c, cs_c);
                }
                else
                {
                    cfg.gemm_ukr.call<T>(k, &alpha, p_a, p_b,
                                         &zero, &p_ab[0], rs_ab, cs_ab);

                    accum_utile(m_loc, n_loc, p_ab, rs_ab, cs_ab,
                                beta, p_c, rs_c, cs_c);
                }
            }
        }

#if TBLIS_ENABLE_TBB

        });

        }
        }

        tg.wait();

#endif
    }
};

}

#endif
