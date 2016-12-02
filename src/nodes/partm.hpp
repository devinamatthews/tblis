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

    template <typename T, typename MatrixA, typename MatrixB, typename MatrixC>
    void operator()(const communicator& comm, const config& cfg,
                    T alpha, MatrixA& A, MatrixB& B, T beta, MatrixC& C)
    {
        using namespace matrix_constants;

        const blocksize& M = cfg.*BS;
        const len_type M_def  = M.def<T>();
        const len_type M_max  = M.max<T>(); // Equal to M_def for register block sizes
        const len_type M_iota = M.iota<T>(); // Equal to the corresponding register block size
        const len_type M_ext  = M.extent<T>(); // Equal to M_def for cache block sizes
        const len_type M_over = M_max-M_def;

        TBLIS_ASSERT(M_ext == M_def);

        //printf("partition along: %c\n", "MNK"[Dim]);
        //printf("A before: %p %ld %ld %ld %ld\n", A.data(), A.length(0), A.length(1), A.stride(0), A.stride(1));
        //printf("B before: %p %ld %ld %ld %ld\n", B.data(), B.length(0), B.length(1), B.stride(0), B.stride(1));
        //printf("C before: %p %ld %ld %ld %ld\n", C.data(), C.length(0), C.length(1), C.stride(0), C.stride(1));

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

        len_type m_u = (Dim == DIM_M ? A.length(0) : Dim == DIM_N ? B.length(1) : A.length(1));
        len_type m_v = (Dim == DIM_M ? C.length(0) : Dim == DIM_N ? C.length(1) : B.length(0));

        if (!ganged)
        {
            //printf("distributing %d ways\n", distribute);
            subcomm = comm.gang(TCI_EVENLY, distribute);
            ganged = true;
        }

        len_type m_first, m_last;
        std::tie(m_first, m_last, std::ignore) =
            subcomm.distribute_over_gangs(std::min(m_u, m_v), M_iota);

        //printf("thread %d/%d in gang %d/%d got %ld-%ld/%ld\n",
        //       subcomm.thread_num(), subcomm.num_threads(),
        //       subcomm.gang_num(), subcomm.num_gangs(),
        //       m_first, m_last, m_u);

        len_type m_off = m_first;
        len_type m_len = m_last-m_first;
        shift(m_off, m_off);

        len_type M_cur = (m_len%M_def <= M_over ? M_max : M_def);

        while (m_off < m_last)
        {
            len_type m_loc = std::min(m_last-m_off, M_cur);
            length(m_loc, m_loc);

            //printf("[%ld:%ld)\n", m_off, m_off+m_loc);

            child(subcomm, cfg, alpha, A, B, beta, C);
            if (Dim == DIM_K) beta = 1.0;

            shift(M_cur, M_cur);
            m_off += M_cur;
            M_cur = M_def;
        }

        shift(-m_off, -m_off);
        length(m_u, m_v);

        //printf("A after: %p %ld %ld %ld %ld\n", A.data(), A.length(0), A.length(1), A.stride(0), A.stride(1));
        //printf("B after: %p %ld %ld %ld %ld\n", B.data(), B.length(0), B.length(1), B.stride(0), B.stride(1));
        //printf("C after: %p %ld %ld %ld %ld\n", C.data(), C.length(0), C.length(1), C.stride(0), C.stride(1));
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
