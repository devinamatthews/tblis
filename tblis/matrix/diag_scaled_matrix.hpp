#ifndef TBLIS_DIAG_SCALED_MATRIX_HPP
#define TBLIS_DIAG_SCALED_MATRIX_HPP

#include "abstract_matrix.hpp"
#include "packed_matrix.hpp"

namespace tblis
{

struct diag_scaled_matrix_impl
{
    char* data_ = nullptr;
    std::array<stride_type,2> stride_ = {};
    std::array<char*,2> diag_ = {};
    std::array<stride_type,2> diag_stride_ = {};

    template <typename T>
    diag_scaled_matrix_impl(T* ptr, stride_type rs, stride_type cs,
                            T* cdiag, stride_type cds,
                            T* rdiag, stride_type rds)
    : data_(ptr), stride_{rs, cs}, diag_{rdiag, cdiag}, diag_stride_{rds, cds} {}
};

class diag_scaled_matrix : public abstract_matrix_adapter<diag_scaled_matrix,diag_scaled_matrix_impl>
{
    protected:
        static abstract_matrix do_pack(abstract_matrix& A_,
                                       const communicator& comm, const config& cfg,
                                       int mat, MemoryPool& pool)
        {
            auto A = static_cast<diag_scaled_matrix&>(A_);
            const type_t type = A.type();
            const bool trans = mat == matrix_constants::MAT_B;
            const len_type MR = (!trans ? cfg.gemm_mr.def(type)
                                        : cfg.gemm_nr.def(type));
            const len_type ME = (!trans ? cfg.gemm_mr.extent(type)
                                        : cfg.gemm_nr.extent(type));
            const len_type KR = cfg.gemm_kr.def(type);

            const stride_type ts = type_size[type];
            const len_type m = A.length( trans);
            const len_type k = A.length(!trans);
            const stride_type rs_a = A.stride( trans)*MR*ts;
            const stride_type cs_a = A.stride(!trans)*KR*ts;
            const stride_type inc_d = A.diag_stride( trans)*MR*ts;
            const stride_type inc_e = A.diag_stride(!trans)*KR*ts;

            const len_type m_p = ceil_div(m, MR)*ME;
            const len_type k_p = round_up(k, KR);
            const stride_type rs_ap = k_p*ME*ts;
            const stride_type cs_ap = KR*ME*ts;
            const len_type MC = std::max(!trans ? cfg.gemm_mc.max(type)
                                                : cfg.gemm_nc.max(type), m_p);
            const len_type KC = std::max(cfg.gemm_kc.max(type), k_p);
            const stride_type nelem = MC*KC + std::max(MC,KC)*TBLIS_MAX_UNROLL;

            packed_matrix P(type, !trans ? m : k, !trans ? k : m,
                            A.get_buffer(comm, nelem, pool), k_p*ME);

            comm.distribute_over_threads(ceil_div(m, MR), ceil_div(k, KR),
            [&](len_type m_first, len_type m_last, len_type k_first, len_type k_last)
            {
                for (len_type m_off = m_first;m_off < m_last;m_off++)
                {
                    const char*  p_a = A.data(      ) + m_off*rs_a  + k_first*cs_a;
                    const char*  p_d = A.diag( trans); if (p_d) p_d += m_off*inc_d;
                    const char*  p_e = A.diag(!trans); if (p_e) p_e += k_first*inc_e;
                          char* p_ap = P.data(      ) + m_off*rs_ap + k_first*cs_ap;

                    len_type m_loc = std::min(MR, m-m_off*MR);
                    len_type k_loc = std::min(k,k_last*KR) - k_first*KR;

                    if (!trans)
                        cfg.pack_nn_mr_ukr.call(type, m_loc, k_loc,
                                                &A.scale(), A.conj(), p_a, rs_a, cs_a,
                                                p_d, inc_d, p_e, inc_e, p_ap);
                    else
                        cfg.pack_nn_nr_ukr.call(type, m_loc, k_loc,
                                                &A.scale(), A.conj(), p_a, rs_a, cs_a,
                                                p_d, inc_d, p_e, inc_e, p_ap);
                }
            });

            return P;
        }

        static void do_gemm(abstract_matrix& C_, const communicator& comm, const config& cfg,
                            MemoryPool&, const abstract_matrix& A_, const abstract_matrix& B_)
        {
            auto A = static_cast<const packed_matrix&>(A_);
            auto B = static_cast<const packed_matrix&>(B_);
            auto C = static_cast<diag_scaled_matrix&>(C_);
            const type_t type = C.type();
            const len_type MR = cfg.gemm_mr.def(type);
            const len_type NR = cfg.gemm_nr.def(type);
            const bool row_major = cfg.gemm_row_major.value(type);

            const stride_type ts = type_size[type];
            const len_type m = C.length(0);
            const len_type n = C.length(1);
            const len_type k = A.length(1);
            const stride_type rs_c = C.stride(0)*MR*ts;
            const stride_type cs_c = C.stride(1)*NR*ts;
            const stride_type rs_ab = row_major ? NR : 1;
            const stride_type cs_ab = row_major ? 1 : MR;
            const stride_type ps_a = A.panel_stride()*ts;
            const stride_type ps_b = B.panel_stride()*ts;
            const stride_type inc_d = C.diag_stride(0)*MR*ts;
            const stride_type inc_e = C.diag_stride(1)*NR*ts;

            const len_type m_first = 0;
            const len_type m_last = ceil_div(m, MR);

            comm.distribute_over_threads(ceil_div(n, NR),
            [&](len_type n_first, len_type n_last)
            {
                constexpr static dcomplex zero{};
                char p_ab[8192] __attribute__((aligned(64)));

                for (len_type n_off = n_first;n_off < n_last;n_off++)
                for (len_type m_off = m_first;m_off < m_last;m_off++)
                {
                    len_type m_loc = std::min(MR, m-m_off*MR);
                    len_type n_loc = std::min(NR, n-n_off*NR);

                    const char* p_a = A.data() + m_off*ps_a;
                    const char* p_b = B.data() + n_off*ps_b;
                    const char* p_d = C.diag(0); if (p_d) p_d += m_off*inc_d;
                    const char* p_e = C.diag(1); if (p_e) p_e += n_off*inc_e;
                          char* p_c = C.data() + m_off*rs_c +
                                                 n_off*cs_c;

                    cfg.gemm_ukr.call(type, MR, NR, k, p_a, p_b,
                                      &zero, p_ab, rs_ab, cs_ab);

                    cfg.update_nn_ukr.call(type, m_loc, n_loc,
                                           p_ab, p_d, inc_d, p_e, inc_e,
                                           &C.scale(), p_c, rs_c, cs_c);
                }
            });
        }

    public:
        diag_scaled_matrix(const tblis_scalar& alpha, bool conj, len_type m, len_type n, char* ptr,
                           stride_type rs, stride_type cs,
                           char* cdiag, stride_type cds,
                           char* rdiag, stride_type rds)
        : abstract_matrix_adapter(alpha, conj, cs == 1 && n > 1,
                                  m, n, ptr, rs, cs, cdiag, cds, rdiag, rds)
        {
            pack_ = &do_pack;
            gemm_ = &do_gemm;
        }

        diag_scaled_matrix(type_t type, bool conj, len_type m, len_type n, char* ptr,
                           stride_type rs, stride_type cs,
                           char* cdiag, stride_type cds,
                           char* rdiag, stride_type rds)
        : diag_scaled_matrix({1.0, type}, conj, m, n, ptr, rs, cs,
                             cdiag, cds, rdiag, rds) {}

        stride_type stride(int dim) const
        {
            TBLIS_ASSERT(dim >= 0 && dim < 2);
            return impl().stride_[dim^transposed()];
        }

        char* data() const
        {
            return impl().data_ + (stride(0)*offset(0) + stride(1)*offset(1))*type_size[type()];
        }

        stride_type diag_stride(int dim) const
        {
            TBLIS_ASSERT(dim >= 0 && dim < 2);
            return impl().diag_stride_[dim^transposed()];
        }

        char* diag(int dim) const
        {
            TBLIS_ASSERT(dim >= 0 && dim < 2);
            auto diag = impl().diag_[dim^transposed()];
            return diag ? diag + diag_stride(dim)*offset(dim)*type_size[type()] : nullptr;
        }
};

}

#endif
