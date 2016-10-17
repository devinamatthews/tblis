#ifndef _TENSOR_TBLIS_PARTITION_HPP_
#define _TENSOR_TBLIS_PARTITION_HPP_

#include "../../external/tci/src/tblis_thread.hpp"
#include "../configs/configs.hpp.in"
#include "../util/assert.h"
#include "../util/basic_types.h"

namespace tblis
{

template <typename Config, int Dim>
struct Partition
{
    template <typename T, template <typename> class Child, template <typename> class... Children>
    struct run
    {
        run() {}

        run(const run& other) : child(other.child) {}

        typename Child<Config>::template run<T, Children...> child;
        thread_communicator subcomm;
        bool ganged = false;

        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(const gemm_thread_config& cfg, thread_communicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
        {
            using namespace matrix_constants;
            using namespace std::placeholders;

            using MB = typename Config::template BS<T,Dim>;
            constexpr len_type M_def  = MB::def;
            constexpr len_type M_max  = MB::max; // Equal to M_def for register block sizes
            constexpr len_type M_iota = MB::iota; // Equal to the corresponding register block size
            constexpr len_type M_ext  = MB::extent; // Equal to M_def for cache block sizes

            auto a_ptr = A.data();
            auto b_ptr = B.data();
            auto c_ptr = C.data();

            auto length = [&](len_type m_u, len_type m_v)
            {
                (Dim & DIM_M ? A.length(0, m_u) : Dim & DIM_N ? B.length(1, m_u) : A.length(1, m_u));
                (Dim & DIM_M ? C.length(0, m_v) : Dim & DIM_N ? C.length(1, m_v) : B.length(0, m_v));
            };

            auto shift = [&](len_type m_u, len_type m_v)
            {
                (Dim & DIM_M ? A.shift(0, m_u) : Dim & DIM_N ? B.shift(1, m_u) : A.shift(1, m_u));
                (Dim & DIM_M ? C.shift(0, m_v) : Dim & DIM_N ? C.shift(1, m_v) : B.shift(0, m_v));
            };

            len_type m_u = (Dim & DIM_M ? A.length(0) : Dim & DIM_N ? B.length(1) : A.length(1));
            len_type m_v = (Dim & DIM_M ? C.length(0) : Dim & DIM_N ? C.length(1) : B.length(0));

            len_type m_first = 0;
            len_type m_last = std::min(m_u, m_v);
            len_type m_max = m_last;

            int distribute = std::min(comm.num_threads(),
                (Dim == DIM_NC ? cfg.jc_nt :
                 Dim == DIM_MC ? cfg.ic_nt :
                 Dim == DIM_NR ? cfg.jr_nt :
                 Dim == DIM_MR ? cfg.ir_nt : 1));

            if (distribute > 1)
            {
                if (!ganged)
                {
                    subcomm = comm.gang_evenly(distribute);
                    ganged = true;
                }

                std::tie(m_first, m_last, m_max) =
                    subcomm.distribute_over_gangs(distribute,
                                                  std::min(m_u, m_v),
                                                  M_iota);
            }

            thread_communicator& child_comm = (distribute > 1 ? subcomm : comm);

            //
            // Shift to the region for our thread team. Shift in units of M_ext
            // for A and B for register block sizes since these are packed.
            //
            length(m_last-m_first, m_last-m_first);
            shift((m_first*M_ext)/M_def, m_first);

            len_type m_off = m_first;

            //
            // Do an extra-large block first if the remainder fits in the
            // max size (only happens for cache block sizes).
            //
            if ((m_last-m_first)%M_def <= (M_max-M_def))
            {
                len_type m_loc = std::min(m_last-m_off, M_max);
                length(m_loc, m_loc);
                child(cfg, child_comm, alpha, A, B, beta, C);
                shift(m_loc, m_loc);
                m_off += m_loc;
                if (Dim & DIM_K) beta = 1.0;
            }

            //
            // Loop over full blocks. Shift A/B by M_ext instead of M_def
            // for register block sizes since these are packed matrices.
            //
            while (m_off < m_last-M_def)
            {
                length(M_def, M_def);
                child(cfg, child_comm, alpha, A, B, beta, C);
                shift(M_ext, M_def);
                m_off += M_def;
                if (Dim & DIM_K) beta = 1.0;
            }

            //
            // Last (partial) block -- no shifting.
            //
            if (m_off < m_last)
            {
                len_type m_loc = m_last-m_off;
                length(m_loc, m_loc);
                child(cfg, child_comm, alpha, A, B, beta, C);
            }

            //
            // Shift back up, accounting for M_ext/M_def difference for A/B.
            //
            shift(-(m_off*M_ext)/M_def, -m_off);
            length(m_u, m_v);
        }
    };
};

template <typename Config>
using PartitionMC = Partition<Config, matrix_constants::DIM_MC>;

template <typename Config>
using PartitionMR = Partition<Config, matrix_constants::DIM_MR>;

template <typename Config>
using PartitionNC = Partition<Config, matrix_constants::DIM_NC>;

template <typename Config>
using PartitionNR = Partition<Config, matrix_constants::DIM_NR>;

template <typename Config>
using PartitionKC = Partition<Config, matrix_constants::DIM_KC>;

}

#endif
