#ifndef _TBLIS_SCATTER_TENSOR_MATRIX_HPP_
#define _TBLIS_SCATTER_TENSOR_MATRIX_HPP_

#include "tensor_matrix.hpp"

namespace tblis
{

struct scatter_tensor_matrix_impl
{
    typedef MArray::detail::array_1d<len_type> len_array;
    typedef MArray::detail::array_1d<stride_type> stride_array;

    char* data_ = nullptr;
    std::array<len_vector, 2> lens_ = {};
    std::array<stride_vector, 2> strides_ = {};
    std::array<len_type, 2> inner_len_ = {};
    std::array<len_type, 2> outer_len_ = {};
    std::array<const stride_type*, 2> scatter_ = {};
    std::array<bool, 2> pack_3d_ = {};

    scatter_tensor_matrix_impl(char* ptr,
                               const len_array& len_m,
                               const len_array& len_n,
                               const stride_array& stride_m,
                               const stride_array& stride_n,
                               const row_view<const stride_type>& rscat,
                               const row_view<const stride_type>& cscat,
                               bool pack_m_3d, bool pack_n_3d)
    : data_(ptr), outer_len_{rscat.length(), cscat.length()},
      scatter_{rscat.data(), cscat.data()}, pack_3d_{pack_m_3d, pack_n_3d}
    {
        if (!outer_len_[0]) outer_len_[0] = 1;
        if (!outer_len_[1]) outer_len_[1] = 1;
        len_m.slurp(lens_[0]);
        len_n.slurp(lens_[1]);
        stride_m.slurp(strides_[0]);
        stride_n.slurp(strides_[1]);
        inner_len_ = {stl_ext::prod(lens_[0]), stl_ext::prod(lens_[1])};
    }
};

class scatter_tensor_matrix : public abstract_matrix_adapter<scatter_tensor_matrix,scatter_tensor_matrix_impl>
{
    protected:
        typedef MArray::detail::array_1d<unsigned> idx_array;
        typedef MArray::detail::array_1d<len_type> len_array;
        typedef MArray::detail::array_1d<stride_type> stride_array;

        static abstract_matrix do_pack(abstract_matrix& A_,
                                       const communicator& comm, const config& cfg,
                                       int mat, MemoryPool& pool)
        {
            auto& A = static_cast<scatter_tensor_matrix&>(A_);
            const type_t type = A.type();
            const bool trans = mat == matrix_constants::MAT_B;
            const len_type MR = (!trans ? cfg.gemm_mr.def(type)
                                        : cfg.gemm_nr.def(type));
            const len_type ME = (!trans ? cfg.gemm_mr.extent(type)
                                        : cfg.gemm_nr.extent(type));
            const len_type KR = cfg.gemm_kr.def(type);
            const len_type KE = cfg.gemm_kr.extent(type);

            const stride_type ts = type_size[type];
            const len_type m = A.length( trans);
            const len_type k = A.length(!trans);

            const len_type m_p = ceil_div(m, MR)*ME;
            const len_type k_p = round_up(k, KE);
            const len_type MC = std::max(!trans ? cfg.gemm_mc.max(type)
                                                : cfg.gemm_nc.max(type), m_p);
            const len_type KC = std::max(cfg.gemm_kc.max(type), k_p);
            const stride_type nelem = MC*KC + std::max(MC,KC)*TBLIS_MAX_UNROLL +
                                      2*size_as_type<stride_type>(MC+KC, type);

            packed_matrix P(type, !trans ? m : k_p, !trans ? k_p : m,
                            A.get_buffer(comm, nelem, pool), k_p*ME);

            stride_type* rscat = convert_and_align<stride_type>(P.data() + m_p*k_p*ts);
            stride_type* cscat = rscat + m_p;
            stride_type* rbs = cscat + k_p;
            stride_type* cbs = rbs + m_p;

            if (!trans)
                A.fill_block_scatter(comm, MR, KR, rscat, cscat, rbs, cbs);
            else
                A.fill_block_scatter(comm, KR, MR, cscat, rscat, cbs, rbs);

            tensor_matrix::pack(comm, cfg, A.scale(), A.conj(), trans, m, k,
                                A.data(), rscat, cscat, rbs, cbs,
                                P.data(), P.panel_stride());

            return P;
        }

        static void do_gemm(abstract_matrix& C_, const communicator& comm, const config& cfg,
                            MemoryPool& pool, const abstract_matrix& A_, const abstract_matrix& B_)
        {
            auto& A = static_cast<  const packed_matrix&>(A_);
            auto& B = static_cast<  const packed_matrix&>(B_);
            auto& C = static_cast<scatter_tensor_matrix&>(C_);

            const type_t type = C.type();
            const len_type MR = cfg.gemm_mr.def(type);
            const len_type NR = cfg.gemm_nr.def(type);
            const len_type m = C.length(0);
            const len_type n = C.length(1);
            const len_type k = A.length(1);
            const len_type MC = std::max(cfg.gemm_mc.max(type), m);
            const len_type NC = std::max(cfg.gemm_nc.max(type), n);
            const stride_type nelem = 2*size_as_type<stride_type>(MC+NC, type);

            stride_type* rscat = convert_and_align<stride_type>(C.get_buffer(comm, nelem, pool));
            stride_type* cscat = rscat + m;
            stride_type* rbs = cscat + n;
            stride_type* cbs = rbs + m;

            C.fill_block_scatter(comm, MR, NR, rscat, cscat, rbs, cbs);

            tensor_matrix::gemm(comm, cfg, C.scale(), C.conj(), m, n, k,
                                A.data(), A.panel_stride(),
                                B.data(), B.panel_stride(),
                                C.data(), rscat, cscat, rbs,  cbs);
        }

    public:
        scatter_tensor_matrix(const scalar& alpha, bool conj,
                              len_array len_m,
                              len_array len_n,
                              char* ptr,
                              stride_array stride_m,
                              stride_array stride_n,
                              const row_view<const stride_type>& rscat,
                              const row_view<const stride_type>& cscat,
                              bool pack_m_3d = false,
                              bool pack_n_3d = false)
        : abstract_matrix_adapter(alpha, conj, 0, 0, false,
                                  ptr, len_m, len_n, stride_m, stride_n,
                                  rscat, cscat, pack_m_3d, pack_n_3d)
        {
            TBLIS_ASSERT(lengths(0).size() == strides(0).size());
            TBLIS_ASSERT(lengths(1).size() == strides(1).size());

            len_type m = inner_length(0)*outer_length(0);
            len_type n = inner_length(1)*outer_length(1);
            bool row_major = !strides(1).empty() &&
                             strides(1)[0] == 1 &&
                             inner_length(1) > 1;

            reset(row_major, m, n);

            pack_ = &do_pack;
            gemm_ = &do_gemm;
        }

        void fill_block_scatter(const communicator& comm,
                                len_type MR, len_type NR,
                                stride_type* rscat, stride_type* cscat,
                                stride_type* rbs, stride_type* cbs)
        {
            if (comm.master())
            {
                for (unsigned dim : {0,1})
                {
                    len_type inner_len = inner_length(dim);
                    len_type outer_len = outer_length(dim);
                    len_type idx, off = offset(dim);
                    divide(off, inner_len, idx, off);
                    len_type len = length(dim);
                    const stride_type* outer_scat = scatter(dim);
                    auto& lens = lengths(dim);
                    auto& strides = this->strides(dim);
                    bool pack_3d = this->pack_3d(dim);

                    stride_type BS = dim == 0 ? MR : NR;
                    stride_type* scat = dim == 0 ? rscat : cscat;

                    while (len > 0)
                    {
                        len_type cur_len = std::min(len, inner_len - off);

                        tensor_matrix::fill_scatter(type(), lens, strides, BS,
                                                    off, cur_len, scat, pack_3d);

                        if (outer_scat)
                        {
                            stride_type idx_scat = outer_scat[idx];
                            for (len_type i = 0;i < cur_len;i++)
                                scat[i] += idx_scat;
                        }

                        off = 0;
                        scat += cur_len;
                        len -= cur_len;
                        idx++;
                    }

                    tensor_matrix::fill_block_stride(BS, length(dim),
                                                     dim == 0 ? rscat : cscat,
                                                     dim == 0 ? rbs : cbs);
                }
            }

            comm.barrier();
        }

        char* data() const
        {
            return impl().data_;
        }

        const len_vector& lengths(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return impl().lens_[dim^transposed()];
        }

        const stride_vector& strides(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return impl().strides_[dim^transposed()];
        }

        len_type inner_length(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return impl().inner_len_[dim^transposed()];
        }

        len_type outer_length(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return impl().outer_len_[dim^transposed()];
        }

        const stride_type* scatter(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return impl().scatter_[dim^transposed()];
        }

        bool pack_3d(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return impl().pack_3d_[dim^transposed()];
        }
};

}

#endif
