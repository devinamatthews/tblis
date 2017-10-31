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

    partition() {}

    partition(const partition& other) : child(other.child) {}

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

        const len_type m_u = (Dim == DIM_M ? A.length(0) :
                              Dim == DIM_N ? B.length(1) :
                                   /*DIM_K*/ A.length(1));
        const len_type m_v = (Dim == DIM_M ? C.length(0) :
                              Dim == DIM_N ? C.length(1) :
                                   /*DIM_K*/ B.length(0));

        subcomm.distribute_over_gangs({std::min(m_u, m_v), M_iota},
        [&,A,B,C,beta](len_type m_first, len_type m_last)
        {
            auto child = this->child;
            auto local_A = const_cast<MatrixA&>(A);
            auto local_B = const_cast<MatrixB&>(B);
            auto local_C = const_cast<MatrixC&>(C);
            auto local_beta = const_cast<T&>(beta);

            auto length = [&](len_type m_u, len_type m_v)
            {
                (Dim == DIM_M ? local_A.length(0, m_u) :
                 Dim == DIM_N ? local_B.length(1, m_u) :
                      /*DIM_K*/ local_A.length(1, m_u));
                (Dim == DIM_M ? local_C.length(0, m_v) :
                 Dim == DIM_N ? local_C.length(1, m_v) :
                      /*DIM_K*/ local_B.length(0, m_v));
            };

            auto shift = [&](len_type m_u, len_type m_v)
            {
                (Dim == DIM_M ? local_A.shift(0, m_u) :
                 Dim == DIM_N ? local_B.shift(1, m_u) :
                      /*DIM_K*/ local_A.shift(1, m_u));
                (Dim == DIM_M ? local_C.shift(0, m_v) :
                 Dim == DIM_N ? local_C.shift(1, m_v) :
                      /*DIM_K*/ local_B.shift(0, m_v));
            };

            len_type m_off = m_first;
            len_type m_len = m_last-m_first;

            shift(m_off, m_off);

            len_type M_cur = (m_len%M_def <= M_over ? M_max : M_def);

            while (m_off < m_last)
            {
                len_type m_loc = std::min(m_last-m_off, M_cur);
                length(m_loc, m_loc);

                child(subcomm, cfg, alpha, local_A, local_B, local_beta, local_C);
                if (Dim == DIM_K) local_beta = 1.0;

                shift(M_cur, M_cur);
                m_off += M_cur;
                M_cur = M_def;
            }
        });
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
