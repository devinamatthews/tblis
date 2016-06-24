#ifndef _TENSOR_TBLIS_PARTITION_HPP_
#define _TENSOR_TBLIS_PARTITION_HPP_

#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <template <typename> class MT, int Dim, int Threading=matrix_constants::NT_NONE>
struct Partition
{
    template <typename T, typename Child, typename... Children>
    struct run
    {
        run() {}

        run(const run& other)
        : child(other.child), distribute(other.distribute) {}

        typename Child::template run<T, Children...> child;
        ThreadCommunicator subcomm;
        int distribute = 1;
        bool ganged = false;

        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
        {
            using namespace matrix_constants;
            using namespace std::placeholders;

            constexpr dim_t M_def  = MT<T>::def;
            constexpr dim_t M_max  = MT<T>::max;
            constexpr dim_t M_iota = MT<T>::iota;

            auto length = [&](dim_t m_u, dim_t m_v)
            {
                (Dim == DIM_M ? A.length(0, m_u) : Dim == DIM_N ? B.length(1, m_u) : A.length(1, m_u));
                (Dim == DIM_M ? C.length(0, m_v) : Dim == DIM_N ? C.length(1, m_v) : B.length(0, m_v));
            };

            auto shift = [&](dim_t m)
            {
                (Dim == DIM_M ? A.shift(0, m) : Dim == DIM_N ? B.shift(1, m) : A.shift(1, m));
                (Dim == DIM_M ? C.shift(0, m) : Dim == DIM_N ? C.shift(1, m) : B.shift(0, m));
            };

            dim_t m_u = (Dim == DIM_M ? A.length(0) : Dim == DIM_N ? B.length(1) : A.length(1));
            dim_t m_v = (Dim == DIM_M ? C.length(0) : Dim == DIM_N ? C.length(1) : B.length(0));

            ASSERT(distribute <= comm.num_threads());

            dim_t m_first = 0;
            dim_t m_last = std::min(m_u, m_v);
            dim_t m_max = m_last;

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

                //printf_locked("%d: gang (%d,%d) %ld %ld %ld %ld\n",
                //              Dim, subcomm.gang_num(), subcomm.thread_num(),
                //              m_u, m_v, m_first, m_last);
            }

            ThreadCommunicator& child_comm = (distribute > 1 ? subcomm : comm);

            //printf_locked("%d: bef (%d,%d) %p %p %p %p %p %p\n",
            //              Dim, child_comm.gang_num(), child_comm.thread_num(),
            //              &A, A.data(), &B, B.data(), &C, C.data());

            length(m_last-m_first);
            shift(m_first);

            int count = 0;
            dim_t m_loc = 0;
            dim_t m_off = m_first;

            if ((m_last-m_first)%M_def <= (M_max-M_def))
            {
                m_loc = std::min(m_last-m_off, M_max);
                length(m_loc, m_loc);
                child(child_comm, alpha, A, B, beta, C);
                shift(m_loc);
                if (Dim == DIM_K) beta = 1.0;
                m_off += m_loc;
                count++;
            }

            while (m_off < m_last)
            {
                m_loc = std::min(m_last-m_off, M_def);
                length(m_loc, m_loc);
                child(child_comm, alpha, A, B, beta, C);
                shift(m_loc);
                if (Dim == DIM_K) beta = 1.0;
                m_off += m_loc;
                count++;
            }

            dim_t n_this = (m_last-m_first+M_def-1)/M_def - (((m_last-m_first)%M_def) <= M_max-M_def);
            dim_t n_max = (m_max-m_first+M_def-1)/M_def - (((m_max-m_first)%M_def) <= M_max-M_def);
            assert(count == n_this);

            length(0, 0);
            for (;n_this < n_max;n_this++)
            {
                child(child_comm, alpha, A, B, beta, C);
            }

            shift(-m_last);
            length(m_u, m_v);

            //printf_locked("%d: aft (%d,%d) %p %p %p %p %p %p\n",
            //              Dim, child_comm.gang_num(), child_comm.thread_num(),
            //              &A, A.data(), &B, B.data(), &C, C.data());
        }
    };
};

template <template <typename> class MT, int Threading=matrix_constants::NT_NONE>
using PartitionM = Partition<MT, matrix_constants::DIM_M, Threading>;

template <template <typename> class NT, int Threading=matrix_constants::NT_NONE>
using PartitionN = Partition<NT, matrix_constants::DIM_N, Threading>;

template <template <typename> class KT, int Threading=matrix_constants::NT_NONE>
using PartitionK = Partition<KT, matrix_constants::DIM_K, Threading>;

}
}

#endif
