#ifndef _TBLIS_NODES_PARTM_HPP_
#define _TBLIS_NODES_PARTM_HPP_

#include "util/basic_types.h"
#include "util/thread.h"

#include "configs/configs.hpp"

namespace tblis
{

template <int Dim, blocksize config::*BS, typename Child>
struct partition
{
    Child child;
    communicator subcomm;
    bool ganged = false;
    int distribute = 1;

    partition() {}

    partition(const partition& other)
    : child(other.child), distribute(other.distribute) {}

    template <typename T, typename Communicator, typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(Communicator& comm, const config& cfg,
                    T alpha, MatrixA& A_, MatrixB& B_, T beta_, MatrixC& C_)
    {
        using namespace matrix_constants;

        const blocksize& M = cfg.*BS;
        const len_type M_def  = M.def<T>();
        const len_type M_max  = M.max<T>(); // Equal to M_def for register block sizes
        const len_type M_iota = M.iota<T>(); // Equal to the corresponding register block size
        const len_type M_ext  = M.extent<T>(); // Equal to M_def for cache block sizes
        const len_type M_over = M_max-M_def;

        TBLIS_ASSERT(M_ext == M_def);

        len_type m_u = (Dim == DIM_M ? A_.length(0) : Dim == DIM_N ? B_.length(1) : A_.length(1));
        len_type m_v = (Dim == DIM_M ? C_.length(0) : Dim == DIM_N ? C_.length(1) : B_.length(0));
        len_type m = std::min(m_u, m_v);

#if TBLIS_ENABLE_TBB

#if TBLIS_SIMPLE_TBB
        tbb::task_group tg;
#endif

        for (int tid = 0;tid < distribute;tid++)
        {

#if TBLIS_SIMPLE_TBB
        tg.run([&,tid,this]
        {

        auto child = this->child;
        auto A = A_;
        auto B = B_;
        auto C = C_;
        auto beta = beta_;
#endif

#else

        if (!ganged)
        {
            subcomm = comm.gang(TCI_EVENLY, distribute);
            ganged = true;
        }

        int tid = subcomm.gang_num();

        auto& A = A_;
        auto& B = B_;
        auto& C = C_;
        auto& beta = beta_;

#endif

        len_type m_first, m_last;
        std::tie(m_first, m_last, std::ignore) =
            communicator::distribute(distribute, tid, m, M_iota);

        auto length = [&](len_type m_u, len_type m_v)
        {
            (Dim == DIM_M ? A.length(0, m_u) : Dim == DIM_N ? B.length(1, m_u) : A.length(1, m_u));
            (Dim == DIM_M ? C.length(0, m_v) : Dim == DIM_N ? C.length(1, m_v) : B.length(0, m_v));
        };

        auto shift = [&](len_type m_u, len_type m_v)
        {
            (Dim == DIM_M ? A.shift(0, m_u) : Dim == DIM_N ? B.shift(1, m_u) : A.shift(1, m_u));
            (Dim == DIM_M ? C.shift(0, m_v) : Dim == DIM_N ? C.shift(1, m_v) : B.shift(0, m_v));
        };

        len_type m_off = m_first;
        len_type m_len = m_last-m_first;

        shift(m_off, m_off);

        len_type M_cur = (m_len%M_def <= M_over ? M_max : M_def);

        while (m_off < m_last)
        {
            len_type m_loc = std::min(m_last-m_off, M_cur);
            length(m_loc, m_loc);

            child(subcomm, cfg, alpha, A, B, beta, C);
            if (Dim == DIM_K) beta = 1.0;

            shift(M_cur, M_cur);
            m_off += M_cur;
            M_cur = M_def;
        }

        shift(-m_off, -m_off);
        length(m_u, m_v);

#if TBLIS_ENABLE_TBB

#if TBLIS_SIMPLE_TBB
        });
#endif

        }

#if TBLIS_SIMPLE_TBB
        tg.wait();
#endif

#endif
    }
};

template <typename Child>
using partition_gemm_mc = partition<matrix_constants::DIM_M, &config::gemm_mc, Child>;

template <typename Child>
using partition_gemm_mr = partition<matrix_constants::DIM_M, &config::gemm_mr, Child>;

template <typename Child>
using partition_gemm_nc = partition<matrix_constants::DIM_N, &config::gemm_nc, Child>;

template <typename Child>
using partition_gemm_nr = partition<matrix_constants::DIM_N, &config::gemm_nr, Child>;

template <typename Child>
using partition_gemm_kc = partition<matrix_constants::DIM_K, &config::gemm_kc, Child>;

}

#endif
