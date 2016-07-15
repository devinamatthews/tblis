#ifndef _TENSOR_TBLIS_PARTITION_HPP_
#define _TENSOR_TBLIS_PARTITION_HPP_

#include "tblis.hpp"

namespace tblis
{

template <template <typename> class MT, int Dim>
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

        static stride_type stride(const const_matrix_view<T>& m, unsigned dim) { return m.stride(dim); }
        static stride_type stride(const matrix_view<T>& m, unsigned dim) { return m.stride(dim); }
        static stride_type stride(const matrix<T>& m, unsigned dim) { return m.stride(dim); }
        template <typename Matrix>
        static stride_type stride(const Matrix& m, unsigned dim) { return 0; }

        template <typename MatrixA, typename MatrixB, typename MatrixC>
        void operator()(ThreadCommunicator& comm, T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
        {
            using namespace matrix_constants;
            using namespace std::placeholders;

            typedef blocksize_traits<MT<T>> MB;
            constexpr idx_type M_def  = MB::def;
            constexpr idx_type M_max  = MB::max;
            constexpr idx_type M_iota = MB::iota;

            auto a_ptr = A.data();
            auto b_ptr = B.data();
            auto c_ptr = C.data();

            auto length = [&](idx_type m_u, idx_type m_v)
            {
                (Dim == DIM_M ? A.length(0, m_u) : Dim == DIM_N ? B.length(1, m_u) : A.length(1, m_u));
                (Dim == DIM_M ? C.length(0, m_v) : Dim == DIM_N ? C.length(1, m_v) : B.length(0, m_v));
            };

            auto shift_down = [&](idx_type m)
            {
                (Dim == DIM_M ? A.shift_down(0, m) : Dim == DIM_N ? B.shift_down(1, m) : A.shift_down(1, m));
                (Dim == DIM_M ? C.shift_down(0, m) : Dim == DIM_N ? C.shift_down(1, m) : B.shift_down(0, m));
            };

            auto shift_up = [&](idx_type m)
            {
                (Dim == DIM_M ? A.shift_up(0, m) : Dim == DIM_N ? B.shift_up(1, m) : A.shift_up(1, m));
                (Dim == DIM_M ? C.shift_up(0, m) : Dim == DIM_N ? C.shift_up(1, m) : B.shift_up(0, m));
            };

            idx_type m_u = (Dim == DIM_M ? A.length(0) : Dim == DIM_N ? B.length(1) : A.length(1));
            idx_type m_v = (Dim == DIM_M ? C.length(0) : Dim == DIM_N ? C.length(1) : B.length(0));

            //assert(distribute <= comm.num_threads());

            idx_type m_first = 0;
            idx_type m_last = std::min(m_u, m_v);
            idx_type m_max = m_last;

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

                //printf_locked("%d: gang (%d,%d) %d %d %d %d\n",
                //              Dim, subcomm.gang_num(), subcomm.thread_num(),
                //              m_u, m_v, m_first, m_last);
            }

            ThreadCommunicator& child_comm = (distribute > 1 ? subcomm : comm);

            //printf_locked("part %d [%d/%d:%d/%d] %d %d %d:%d\n", Dim,
            //              child_comm.thread_num(), child_comm.num_threads(),
            //              child_comm.gang_num(), distribute,
            //              m_first, m_last, std::min(m_u, m_v), M_def);

            //printf_locked("%d: bef (%d,%d) %p %p %p %p %p %p\n",
            //              Dim, child_comm.gang_num(), child_comm.thread_num(),
            //              &A, A.data(), &B, B.data(), &C, C.data());

            length(m_last-m_first, m_last-m_first);
            shift_down(m_first);

            int count = 0;
            idx_type m_loc = 0;
            idx_type m_off = m_first;

            /*
            if ((m_last-m_first)%M_def <= (M_max-M_def))
            {
                m_loc = std::min(m_last-m_off, M_max);
                length(m_loc, m_loc);
                child(child_comm, alpha, A, B, beta, C);
                shift_down(m_loc);
                if (Dim == DIM_K) beta = 1.0;
                m_off += m_loc;
                count++;
            }
            */

            //printf("range: %d %d : %d\n", m_first, m_last, M_def);

            while (m_off < m_last)
            {
                m_loc = std::min(m_last-m_off, M_def);
                //printf("%d %d\n", m_off, m_loc);
                length(m_loc, m_loc);
                child(child_comm, alpha, A, B, beta, C);
                shift_down(m_loc);
                if (Dim == DIM_K) beta = 1.0;
                m_off += m_loc;
                count++;

                if (Dim == DIM_M)
                {
                    if (stride(A, 0)) assert(A.data() == a_ptr+m_off*stride(A, 0));
                    if (stride(C, 0)) assert(C.data() == c_ptr+m_off*stride(C, 0));
                }
                else if (Dim == DIM_N)
                {
                    if (stride(B, 1)) assert(B.data() == b_ptr+m_off*stride(B, 1));
                    if (stride(C, 1)) assert(C.data() == c_ptr+m_off*stride(C, 1));
                }
                else
                {
                    if (stride(A, 1)) assert(A.data() == a_ptr+m_off*stride(A, 1));
                    if (stride(B, 0)) assert(B.data() == b_ptr+m_off*stride(B, 0));
                }
            }

            /*
            idx_type n_this = (m_last-m_first+M_def-1)/M_def - (((m_last-m_first)%M_def) <= M_max-M_def);
            idx_type n_max = (m_max-m_first+M_def-1)/M_def - (((m_max-m_first)%M_def) <= M_max-M_def);
            assert(count == n_this);

            length(0, 0);
            for (;n_this < n_max;n_this++)
            {
                child(child_comm, alpha, A, B, beta, C);
            }
            */

            shift_up(m_last);
            length(m_u, m_v);

            assert(a_ptr == A.data());
            assert(b_ptr == B.data());
            assert(c_ptr == C.data());

            //printf_locked("%d: aft (%d,%d) %p %p %p %p %p %p\n",
            //              Dim, child_comm.gang_num(), child_comm.thread_num(),
            //              &A, A.data(), &B, B.data(), &C, C.data());
        }
    };
};

template <template <typename> class MT>
using PartitionM = Partition<MT, matrix_constants::DIM_M>;

template <template <typename> class NT>
using PartitionN = Partition<NT, matrix_constants::DIM_N>;

template <template <typename> class KT>
using PartitionK = Partition<KT, matrix_constants::DIM_K>;

}

#endif
