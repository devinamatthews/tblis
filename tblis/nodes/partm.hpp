#ifndef _TBLIS_NODES_PARTM_HPP_
#define _TBLIS_NODES_PARTM_HPP_

#include "util/basic_types.h"
#include "util/thread.h"

#include "configs/configs.hpp"

#include "matrix/abstract_matrix.hpp"

namespace tblis
{

template <int Dim, blocksize config::*BS, typename Child>
struct partition
{
    Child child;
    communicator* subcomm = nullptr;

    void operator()(const communicator&, const config& cfg,
                    abstract_matrix& A, abstract_matrix& B, abstract_matrix& C)
    {
        using namespace matrix_constants;

        const blocksize& M = cfg.*BS;
        const len_type M_def  = M.def(A.type());
        const len_type M_max  = M.max(A.type()); // Equal to M_def for register block sizes
        const len_type M_iota = M.iota(A.type()); // Equal to the corresponding register block size
        const len_type M_over = M_max-M_def;

        const len_type m_u = (Dim == DIM_M ? A.length(0) :
                              Dim == DIM_N ? B.length(1) :
                                   /*DIM_K*/ A.length(1));
        const len_type m_v = (Dim == DIM_M ? C.length(0) :
                              Dim == DIM_N ? C.length(1) :
                                   /*DIM_K*/ B.length(0));

        subcomm->distribute_over_gangs({std::min(m_u, m_v), M_iota},
        [&](len_type m_first, len_type m_last)
        {
            auto local_A = A;
            auto local_B = B;
            auto local_C = C;

            auto length = [&](len_type m)
            {
                (Dim == DIM_M ? local_A.length(0, m) :
                 Dim == DIM_N ? local_B.length(1, m) :
                      /*DIM_K*/ local_A.length(1, m));
                (Dim == DIM_M ? local_C.length(0, m) :
                 Dim == DIM_N ? local_C.length(1, m) :
                      /*DIM_K*/ local_B.length(0, m));
            };

            auto shift = [&](len_type m, len_type n=0)
            {
                (Dim == DIM_M ? local_A.shift_and_resize(0, m, n) :
                 Dim == DIM_N ? local_B.shift_and_resize(1, m, n) :
                      /*DIM_K*/ local_A.shift_and_resize(1, m, n));
                (Dim == DIM_M ? local_C.shift_and_resize(0, m, n) :
                 Dim == DIM_N ? local_C.shift_and_resize(1, m, n) :
                      /*DIM_K*/ local_B.shift_and_resize(0, m, n));
            };

            len_type m_off = m_first;
            len_type m_len = m_last-m_first;

            shift(m_off, m_len);

            len_type M_cur = (m_len%M_def <= M_over ? M_max : M_def);

            while (m_off < m_last)
            {
                len_type m_loc = std::min(m_last-m_off, M_cur);
                length(m_loc);

                child(*subcomm, cfg, local_A, local_B, local_C);
                if (Dim == DIM_K) local_C.set_scaled();

                shift(m_loc);
                m_off += m_loc;
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
