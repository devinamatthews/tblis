#ifndef _TBLIS_BLOCK_SCATTER_MATRIX_HPP_
#define _TBLIS_BLOCK_SCATTER_MATRIX_HPP_

#include "util/basic_types.h"
#include "util/macros.h"
#include "util/thread.h"

#include "memory/alignment.hpp"

#include "normal_matrix.hpp"
#include "tensor_matrix.hpp"
#include "dpd_tensor_matrix.hpp"

namespace tblis
{

template <typename T>
class block_scatter_matrix : public abstract_matrix<T>
{
    template <typename> friend class patch_block_scatter_matrix;

    public:
        typedef const stride_type* scatter_type;

    protected:
        using abstract_matrix<T>::data_;
        using abstract_matrix<T>::tot_len_;
        using abstract_matrix<T>::cur_len_;
        using abstract_matrix<T>::off_;
        std::array<stride_type*, 2> scatter_ = {};
        std::array<stride_type*, 2> block_stride_ = {};
        std::array<len_type, 2> block_size_ = {};

        static void fill_block_scatter(len_vector len, stride_vector stride, len_type BS,
                                       stride_type off, stride_type size,
                                       stride_type* scat, stride_type* bs)
        {
            if (size == 0) return;

            if (len.empty())
            {
                *scat = 0;
                *bs = 1;
                return;
            }

            stride_type tot_len = 1;
            for (auto& l : len) tot_len *= l;
            TBLIS_ASSERT(off >= 0);
            TBLIS_ASSERT(size >= 0);
            TBLIS_ASSERT(off+size <= tot_len);

            len_type m0 = (len.empty() ? 1 : len[0]);
            stride_type s0 = (stride.empty() ? 0 : stride[0]);

            len.erase(len.begin());
            stride.erase(stride.begin());
            viterator<> it(len, stride);

            len_type p0 = off%m0;
            stride_type pos = 0;
            it.position(off/m0, pos);

            for (len_type idx = 0;idx < size && it.next(pos);)
            {
                auto pos2 = pos + p0*s0;
                auto imax = std::min(m0-p0,size-idx);
                for (len_type i = 0;i < imax;i++)
                {
                    scat[idx++] = pos2;
                    pos2 += s0;
                }
                p0 = 0;
            }

            for (len_type i = 0, b = 0;i < size;i += BS, b++)
            {
                len_type bl = std::min(size-i, BS);
                stride_type s = bl > 1 ? scat[i+1]-scat[i] : 1;
                for (len_type j = i+1;j+1 < i+bl;j++)
                {
                    if (scat[j+1]-scat[j] != s) s = 0;
                }
                bs[b] = s;
            }
        }

    public:
        block_scatter_matrix() {}

        block_scatter_matrix(const communicator& comm, const tensor_matrix<T>& A,
                             len_type MB, stride_type* rscat, stride_type* rbs,
                             len_type NB, stride_type* cscat, stride_type* cbs)
        {
            data_ = A.data_;
            tot_len_ = cur_len_ = A.cur_len_;
            scatter_ = {rscat, cscat};
            block_stride_ = {rbs, cbs};
            block_size_ = {MB, NB};

            if (comm.master())
            {
                fill_block_scatter(A.lens_[0], A.strides_[0], block_size_[0], A.off_[0],
                                   tot_len_[0], scatter_[0], block_stride_[0]);
                fill_block_scatter(A.lens_[1], A.strides_[1], block_size_[1], A.off_[1],
                                   tot_len_[1], scatter_[1], block_stride_[1]);
            }

            comm.barrier();
        }

        block_scatter_matrix(const communicator& comm, const dpd_tensor_matrix<T>& A,
                             len_type MB, stride_type* rscat, stride_type* rbs,
                             len_type NB, stride_type* cscat, stride_type* cbs)
        {
            assert(0);
        }

        len_type block_size(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return block_size_[dim];
        }

        void transpose()
        {
            using std::swap;
            abstract_matrix<T>::transpose();
            swap(scatter_[0], scatter_[1]);
            swap(block_stride_[0], block_stride_[1]);
            swap(block_size_[0], block_size_[1]);
        }

        void block(T*& data,
                   scatter_type& rscat, stride_type& rbs, len_type& rbl,
                   scatter_type& cscat, stride_type& cbs, len_type& cbl) const
        {
            rscat = scatter_[0] + off_[0];
            cscat = scatter_[1] + off_[1];

            TBLIS_ASSERT(off_[0]%block_size_[0] == 0);
            TBLIS_ASSERT(off_[1]%block_size_[1] == 0);

            rbs = block_stride_[0][off_[0]/block_size_[0]];
            cbs = block_stride_[1][off_[1]/block_size_[1]];

            rbl = std::min(block_size_[0], cur_len_[0]);
            cbl = std::min(block_size_[1], cur_len_[1]);

            data = data_ + (rbs ? *rscat : 0) + (cbs ? *cscat : 0);
        }

        void pack(const communicator& comm, const config& cfg,
                  bool trans, normal_matrix<T>& Ap) const
        {
            const len_type MR = (!trans ? cfg.gemm_mr.def<T>()
                                        : cfg.gemm_nr.def<T>());
            const len_type ME = (!trans ? cfg.gemm_mr.extent<T>()
                                        : cfg.gemm_nr.extent<T>());
            const len_type KR = cfg.gemm_kr.def<T>();

            TBLIS_ASSERT(block_size_[ trans] == MR);
            TBLIS_ASSERT(block_size_[!trans] == KR);

            const len_type m_a = cur_len_[ trans];
            const len_type k_a = cur_len_[!trans];

            comm.distribute_over_threads({m_a, MR}, {k_a, KR},
            [&](len_type m_first, len_type m_last, len_type k_first, len_type k_last)
            {
                T* p_ap = Ap.data() + (m_first/MR)*ME*Ap.stride(trans) + k_first*ME;
                scatter_type rscat_a = scatter_[ trans] + off_[ trans] + m_first;
                scatter_type cscat_a = scatter_[!trans] + off_[!trans] + k_first;
                scatter_type rbs_a = block_stride_[ trans] + (off_[ trans] + m_first)/block_size_[ trans];
                scatter_type cbs_a = block_stride_[!trans] + (off_[!trans] + k_first)/block_size_[!trans];

                for (len_type m_off = m_first;m_off < m_last;)
                {
                    TBLIS_ASSERT(rscat_a - scatter_[ trans] < tot_len_[ trans]);
                    TBLIS_ASSERT(cscat_a - scatter_[!trans] < tot_len_[!trans]);
                    TBLIS_ASSERT(rbs_a - block_stride_[ trans] < ceil_div(tot_len_[ trans], block_size_[ trans]));
                    TBLIS_ASSERT(cbs_a - block_stride_[!trans] < ceil_div(tot_len_[!trans], block_size_[!trans]));

                    len_type m = std::min(MR, m_last - m_off);
                    len_type k = k_last - k_first;
                    stride_type rs_a = *rbs_a;
                    const T* p_a = data_ + (rs_a ? *rscat_a : 0);

                    TBLIS_ASSERT(p_ap + k*ME <= Ap.data() + ceil_div(Ap.length(trans), MR)*ME*Ap.length(!trans));

                    if (rs_a)
                    {
                        if (!trans)
                            cfg.pack_nb_mr_ukr.call<T>(m, k, p_a, rs_a, cscat_a, cbs_a, p_ap);
                        else
                            cfg.pack_nb_nr_ukr.call<T>(m, k, p_a, rs_a, cscat_a, cbs_a, p_ap);
                    }
                    else
                    {
                        if (!trans)
                            cfg.pack_sb_mr_ukr.call<T>(m, k, p_a, rscat_a, cscat_a, cbs_a, p_ap);
                        else
                            cfg.pack_sb_nr_ukr.call<T>(m, k, p_a, rscat_a, cscat_a, cbs_a, p_ap);
                    }

                    p_ap += ME*Ap.stride(trans);
                    m_off += MR;
                    rscat_a += MR;
                    rbs_a++;
                }
            });
        }
};

}

#endif
