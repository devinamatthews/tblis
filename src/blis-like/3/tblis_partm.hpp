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

            constexpr dim_t M_def  = MT<T>::def;
            constexpr dim_t M_max  = MT<T>::max;
            constexpr dim_t M_iota = MT<T>::iota;

            dim_t m_u = (Dim == DIM_M ? A.length() : Dim == DIM_N ? B.width() : A.width());
            dim_t m_v = (Dim == DIM_M ? C.length() : Dim == DIM_N ? C.width() : B.length());

            ASSERT(distribute <= comm.num_threads());

            dim_t m_first = 0;
            dim_t m_last = std::max(m_u, m_v);

            if (distribute > 1)
            {
                if (!ganged)
                {
                    subcomm = comm.gang_evenly(distribute);
                    ganged = true;
                }

                std::tie(m_first, m_last) =
                    subcomm.distribute_over_gangs(distribute,
                                                  std::max(m_u, m_v),
                                                  M_iota);

                //printf_locked("%d: gang (%d,%d) %ld %ld %ld %ld\n",
                //              Dim, subcomm.gang_num(), subcomm.thread_num(),
                //              m_u, m_v, m_first, m_last);
            }

            ThreadCommunicator& child_comm = (distribute > 1 ? subcomm : comm);

            dim_t m_last_u = std::min(m_last, m_u);
            dim_t m_last_v = std::min(m_last, m_v);

            //printf_locked("%d: bef (%d,%d) %p %p %p %p %p %p\n",
            //              Dim, child_comm.gang_num(), child_comm.thread_num(),
            //              &A, A.data(), &B, B.data(), &C, C.data());

            (Dim == DIM_M ? A.length(m_last_u-m_first) : Dim == DIM_N ? B.width(m_last_u-m_first) : A.width(m_last_u-m_first));
            (Dim == DIM_M ? C.length(m_last_v-m_first) : Dim == DIM_N ? C.width(m_last_v-m_first) : B.length(m_last_v-m_first));
            (Dim == DIM_M ? A.shift_down(m_first) : Dim == DIM_N ? B.shift_right(m_first) : A.shift_right(m_first));
            (Dim == DIM_M ? C.shift_down(m_first) : Dim == DIM_N ? C.shift_right(m_first) : B.shift_down(m_first));

            dim_t u = 0;
            dim_t v = 0;
            for (dim_t off_u  =  m_first,   off_v  =  m_first;
                       off_u  < m_last_u && off_v  < m_last_v;
                       off_u +=        u,   off_v +=        v)
            {
                if (m_last_u-off_u <= M_max && m_last_v-off_v <= M_max)
                {
                    u = m_last_u-off_u;
                    v = m_last_v-off_v;
                }
                else
                {
                    u = M_def;
                    v = M_def;
                }

                (Dim == DIM_M ? A.length(u) : Dim == DIM_N ? B.width(u) : A.width(u));
                (Dim == DIM_M ? C.length(v) : Dim == DIM_N ? C.width(v) : B.length(v));

                child(child_comm, alpha, A, B, beta, C);

                (Dim == DIM_M ? A.shift_down() : Dim == DIM_N ? B.shift_right() : A.shift_right());
                (Dim == DIM_M ? C.shift_down() : Dim == DIM_N ? C.shift_right() : B.shift_down());

                if (Dim == DIM_K) beta = 1.0;
            }

            (Dim == DIM_M ? A.shift_up(m_last_u) : Dim == DIM_N ? B.shift_left(m_last_u) : A.shift_left(m_last_u));
            (Dim == DIM_M ? C.shift_up(m_last_v) : Dim == DIM_N ? C.shift_left(m_last_v) : B.shift_up(m_last_v));
            (Dim == DIM_M ? A.length(m_u) : Dim == DIM_N ? B.width(m_u) : A.width(m_u));
            (Dim == DIM_M ? C.length(m_v) : Dim == DIM_N ? C.width(m_v) : B.length(m_v));

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
