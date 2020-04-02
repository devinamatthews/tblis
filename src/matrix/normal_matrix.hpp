#ifndef _TBLIS_NORMAL_MATRIX_HPP_
#define _TBLIS_NORMAL_MATRIX_HPP_

#include "abstract_matrix.hpp"
#include "packed_matrix.hpp"

namespace tblis
{

struct normal_matrix_impl
{
    char* data_ = nullptr;
    std::array<stride_type,2> stride_ = {};

    template <typename T>
    normal_matrix_impl(T* ptr, stride_type rs, stride_type cs)
    : data_(ptr), stride_{rs, cs} {}
};

class normal_matrix : public abstract_matrix_adapter<normal_matrix,normal_matrix_impl>
{
    protected:
        static abstract_matrix do_pack(abstract_matrix& A_,
                                       const communicator& comm, const config& cfg,
                                       int mat, MemoryPool& pool)
        {
            auto& A = static_cast<normal_matrix&>(A_);

            const type_t type = A.type();
            const bool trans = mat == matrix_constants::MAT_B;
            const len_type MR = (!trans ? cfg.gemm_mr.def(type)
                                        : cfg.gemm_nr.def(type));
            const len_type ME = (!trans ? cfg.gemm_mr.extent(type)
                                        : cfg.gemm_nr.extent(type));
            const len_type KE = cfg.gemm_kr.extent(type);

            const stride_type ts = type_size[type];
            const len_type m = A.length( trans);
            const len_type k = A.length(!trans);
            const stride_type rs_a = A.stride( trans);
            const stride_type cs_a = A.stride(!trans);

            const len_type m_p = ceil_div(m, MR)*ME;
            const len_type k_p = round_up(k, KE);
            const len_type MC = std::max(!trans ? cfg.gemm_mc.max(type)
                                                : cfg.gemm_nc.max(type), m_p);
            const len_type KC = std::max(cfg.gemm_kc.max(type), k_p);
            const stride_type nelem = MC*KC + std::max(MC,KC)*TBLIS_MAX_UNROLL;

            packed_matrix P(type, !trans ? m : k_p, !trans ? k_p : m,
                            A.get_buffer(comm, nelem, pool), k_p*ME);

            pack(comm, cfg, A.scale(), A.conj(), trans, m, k,
                 A.data(), rs_a, cs_a, P.data(), P.panel_stride());

            return P;
        }

        static void do_gemm(abstract_matrix& C_, const communicator& comm, const config& cfg,
                            MemoryPool&, const abstract_matrix& A_, const abstract_matrix& B_)
        {
            auto& A = static_cast<const packed_matrix&>(A_);
            auto& B = static_cast<const packed_matrix&>(B_);
            auto& C = static_cast<normal_matrix&>(C_);

            const len_type m = C.length(0);
            const len_type n = C.length(1);
            const len_type k = A.length(1);
            const stride_type rs_c = C.stride(0);
            const stride_type cs_c = C.stride(1);

            gemm(comm, cfg, C.scale(), C.conj(), m, n, k,
                 A.data(), A.panel_stride(), B.data(), B.panel_stride(),
                 C.data(), rs_c, cs_c);
        }

    public:
        normal_matrix(const scalar& alpha, bool conj,
                      len_type m, len_type n, char* ptr,
                      stride_type rs, stride_type cs)
        : abstract_matrix_adapter(alpha, conj, cs == 1 && n > 1,
                                  m, n, ptr, rs, cs)
        {
            pack_ = &do_pack;
            gemm_ = &do_gemm;
        }

        static void pack(const communicator& comm, const config& cfg,
                         const scalar& alpha, bool conj,
                         bool trans, len_type m, len_type k,
                         char* p_a, stride_type rs, stride_type cs,
                         char* p_ap, stride_type ps_ap)
        {
            const type_t type = alpha.type;
            const len_type MR = (!trans ? cfg.gemm_mr.def(type)
                                        : cfg.gemm_nr.def(type));
            const len_type ME = (!trans ? cfg.gemm_mr.extent(type)
                                        : cfg.gemm_nr.extent(type));
            const len_type KE = cfg.gemm_kr.extent(type);
            const stride_type ts = type_size[type];

            comm.distribute_over_threads({m, MR}, {k, KE},
            [&](len_type m_first, len_type m_last, len_type k_first, len_type k_last)
            {
                const char*  p_a1 = p_a + m_first*rs*ts  + k_first*cs*ts;
                      char* p_ap1 = p_ap + (m_first/MR)*ps_ap*ts + k_first*ME*ts;

                for (len_type m_off = m_first;m_off < m_last;m_off += MR)
                {
                    len_type m_loc = std::min(MR, m-m_off);
                    len_type k_loc = k_last-k_first;

                    if (!trans)
                        cfg.pack_nn_mr_ukr.call(type, m_loc, k_loc,
                                                &alpha, conj, p_a1, rs, cs,
                                                nullptr, 0, nullptr, 0, p_ap1);
                    else
                        cfg.pack_nn_nr_ukr.call(type, m_loc, k_loc,
                                                &alpha, conj, p_a1, rs, cs,
                                                nullptr, 0, nullptr, 0, p_ap1);

                    p_a1 += rs*MR*ts;
                    p_ap1 += ps_ap*ts;
                }
            });
        }

        using abstract_matrix::pack;

        static void gemm(const communicator& comm, const config& cfg,
                         const scalar& beta, bool conj,
                         len_type m, len_type n, len_type k,
                         char* p_a, stride_type ps_a,
                         char* p_b, stride_type ps_b,
                         char* p_c, stride_type rs, stride_type cs)
        {
            const type_t type = beta.type;
            const len_type MR = cfg.gemm_mr.def(type);
            const len_type ME = cfg.gemm_mr.extent(type);
            const len_type NR = cfg.gemm_nr.def(type);
            const len_type NE = cfg.gemm_nr.extent(type);
            const len_type KE = cfg.gemm_kr.extent(type);
            const bool row_major = cfg.gemm_row_major.value(type);
            const stride_type ts = type_size[type];
            const len_type m_first = 0;
            const len_type m_last = ceil_div(m, MR);
            const stride_type rs_ab = row_major ? NR : 1;
            const stride_type cs_ab = row_major ? 1 : MR;
            const len_type k_p = round_up(k, KE);

            comm.distribute_over_threads(ceil_div(n, NR),
            [&](len_type n_first, len_type n_last)
            {
                for (len_type n_off = n_first;n_off < n_last;n_off++)
                for (len_type m_off = m_first;m_off < m_last;m_off++)
                {
                    len_type m_loc = std::min(MR, m-m_off*MR);
                    len_type n_loc = std::min(NR, n-n_off*NR);

                    char* p_a1 = p_a + m_off*ps_a*ts;
                    char* p_b1 = p_b + n_off*ps_b*ts;
                    char* p_c1 = p_c + m_off*MR*rs*ts
                                     + n_off*NR*cs*ts;

                    auxinfo_t aux{p_a1, p_b1, p_c1};
                    cfg.gemm_ukr.call(type, m_loc, n_loc, k, p_a1, p_b1,
                                      &beta, p_c1, rs, cs, &aux);
                }
            });
        }

        using abstract_matrix::gemm;

        stride_type stride(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return impl().stride_[dim^transposed()];
        }

        char* data() const
        {
            return impl().data_ + (stride(0)*offset(0) + stride(1)*offset(1))*type_size[type()];
        }
};

}

#endif
