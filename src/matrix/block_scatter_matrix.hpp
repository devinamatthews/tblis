#ifndef _TBLIS_block_scatter_matrix_HPP_
#define _TBLIS_block_scatter_matrix_HPP_

#include "util/basic_types.h"
#include "util/macros.h"
#include "util/thread.h"

#include "memory/alignment.hpp"

#include "tensor_matrix.hpp"
#include "dpd_tensor_matrix.hpp"

namespace tblis
{

template <typename T>
class block_scatter_matrix
{
    public:
        typedef size_t size_type;
        typedef const stride_type* scatter_type;
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;

    protected:
        pointer data_ = nullptr;
        std::array<len_type, 2> len_ = {};
        std::array<len_vector, 2> len_patch_ = {};
        std::array<unsigned, 2> patch_ = {};
        std::array<len_type, 2> off_ = {};
        std::array<len_type, 2> off_patch_ = {};
        std::array<matrix<stride_type*>, 2> scatter_ = {};
        std::array<matrix<stride_type*>, 2> block_stride_ = {};
        std::array<len_type, 2> block_size_ = {};

        void fill_block_scatter(len_vector len, stride_vector stride, len_type BS,
                                stride_type off, stride_type size,
                                stride_type* scat, stride_type* bs)
        {
            len_type m0 = (len.empty() ? 1 : len[0]);
            stride_type s0 = (stride.empty() ? 0 : stride[0]);

            len.erase(len.begin());
            stride.erase(stride.begin());
            viterator<> it(len, stride);

            len_type p0 = off%m0;
            stride_type pos = 0;
            it.position(off/m0, pos);

            for (len_type idx = 0;it.next(pos);)
            {
                for (len_type i0 = p0;i0 < m0;i0++)
                {
                    if (idx == size) return;
                    scat[idx++] = pos + i0*s0;
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
        block_scatter_matrix();

        block_scatter_matrix(const communicator& comm, const tensor_matrix<T>& A,
                             len_type MB, stride_type* rscat, stride_type* rbs,
                             len_type NB, stride_type* cscat, stride_type* cbs)
        {
            data_ = A.data_;
            len_ = A.tot_len_;
            len_patch_ = {{{len_[0]}, {len_[1]}}};
            scatter_= {{{{rscat}}, {{cscat}}}};
            block_stride_ = {{{{rbs}}, {{cbs}}}};
            block_size_ = {{MB, NB}};

            if (comm.master())
            {
                fill_block_scatter(A.len_[0], A.stride_[0], block_size_[0], A.offset_[0],
                                   len_[0], scatter_[0][0][0], block_stride_[0][0][0]);
                fill_block_scatter(A.len_[1], A.stride_[1], block_size_[1], A.offset_[1],
                                   len_[1], scatter_[1][0][0], block_stride_[1][0][0]);
            }
        }

        block_scatter_matrix(const communicator& comm, const dpd_tensor_matrix<T>& A,
                             len_type MB, stride_type* rscat, stride_type* rbs,
                             len_type NB, stride_type* cscat, stride_type* cbs)
        {
            data_ = A.data_;
            len_ = A.len_;
            block_size_ = {{MB, NB}};

            for (unsigned dim : {0,1})
            {
                len_type off = 0;
                len_type block_off = A.block_offset_[dim];
                auto& it = A.iterator_[dim];
                auto old_it = it;
                len_type new_len = 0;

                while (off < len_[dim])
                {
                    len_type loc = std::min(len_[dim]-off,
                        A.block_size(dim)-block_off);
                    len_patch_[dim].push_back(loc);
                    new_len += round_up(loc, block_size_[dim]);
                    block_off = 0;
                    it.next();
                }

                it.swap(old_it);
                len_[dim] = new_len;
            }

            unsigned row_npatch = len_patch_[0].size();
            unsigned col_npatch = len_patch_[1].size();

            scatter_[0].reset({row_npatch, col_npatch});
            scatter_[1].reset({row_npatch, col_npatch});

            block_stride_[0].reset({row_npatch, col_npatch});
            block_stride_[1].reset({row_npatch, col_npatch});

            auto row_it = A.iterator_[0];
            auto row_block_off = A.block_offset_[0];
            len_type row_offset = 0;
            unsigned row_patch = 0;

            irrep_vector irreps(A.tensor_.dimension());

            len_vector row_len(A.dims_[0].size());
            len_vector col_len(A.dims_[1].size());

            stride_vector row_stride(A.dims_[0].size());
            stride_vector col_stride(A.dims_[1].size());

            while (row_offset < len_[0])
            {
                len_type row_block_size = len_patch_[0][row_patch];

                if (!A.dims_[0].empty())
                {
                    irreps[A.dims_[0][0]] = A.block_[0];
                    for (unsigned i = 1;i < A.dims_[0].size();i++)
                        irreps[A.dims_[0][0]] ^= irreps[A.dims_[0][i]] = row_it.position(i-1);
                }

                auto col_it = A.iterator_[1];
                auto col_block_off = A.block_offset_[1];
                len_type col_offset = 0;
                unsigned col_patch = 0;

                while (col_offset < len_[1])
                {
                    len_type col_block_size = len_patch_[1][col_patch];

                    if (!A.dims_[1].empty())
                    {
                        irreps[A.dims_[1][0]] = A.block_[1];
                        for (unsigned i = 1;i < A.dims_[1].size();i++)
                            irreps[A.dims_[1][0]] ^= irreps[A.dims_[1][i]] = col_it.position(i-1);
                    }

                    scatter_[0][row_patch][col_patch] = rscat;
                    scatter_[1][row_patch][col_patch] = cscat;

                    block_stride_[0][row_patch][col_patch] = rbs;
                    block_stride_[1][row_patch][col_patch] = cbs;

                    if (comm.master())
                    {
                        auto A2 = A.tensor_(irreps);

                        for (unsigned i = 0;i < A.dims_[0].size();i++)
                        {
                            row_len[i] = A2.length(A.dims_[0][i]);
                            row_stride[i] = A2.stride(A.dims_[0][i]);
                        }

                        for (unsigned i = 0;i < A.dims_[1].size();i++)
                        {
                            col_len[i] = A2.length(A.dims_[1][i]);
                            col_stride[i] = A2.stride(A.dims_[1][i]);
                        }

                        fill_block_scatter(row_len, row_stride, block_size_[0], row_block_off,
                                           row_block_size, rscat, rbs);
                        fill_block_scatter(col_len, col_stride, block_size_[1], col_block_off,
                                           col_block_size, cscat, cbs);

                        stride_type ptr_off = A2.data() - data_;

                        if (row_block_size < col_block_size)
                        {
                            for (len_type i = 0;i < row_block_size;i++)
                                rscat[i] += ptr_off;
                        }
                        else
                        {
                            for (len_type i = 0;i < col_block_size;i++)
                                cscat[i] += ptr_off;
                        }
                    }

                    rscat += row_block_size;
                    rbs += ceil_div(row_block_size, block_size_[0]);

                    cscat += col_block_size;
                    cbs += ceil_div(col_block_size, block_size_[1]);

                    col_offset += round_up(col_block_size, NB);
                    col_block_off = 0;
                    col_patch++;
                }

                row_offset += round_up(row_block_size, MB);
                row_block_off = 0;
                row_patch++;
            }
        }

        block_scatter_matrix& operator=(const block_scatter_matrix&) = delete;

        len_type block_size(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return block_size_[dim];
        }

        len_type length(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return len_[dim];
        }

        len_type length(unsigned dim, len_type m)
        {
            TBLIS_ASSERT(dim < 2);
            std::swap(m, len_[dim]);
            return m;
        }

        void block(pointer& data,
                   scatter_type& rscat, stride_type& rbs, len_type& rbl,
                   scatter_type& cscat, stride_type& cbs, len_type& cbl) const
        {
            rscat = scatter_[0][patch_[0]][patch_[1]] + off_patch_[0];
            cscat = scatter_[1][patch_[0]][patch_[1]] + off_patch_[1];

            rbs = block_stride_[0][patch_[0]][patch_[1]][off_patch_[0]/block_size_[0]];
            cbs = block_stride_[1][patch_[0]][patch_[1]][off_patch_[1]/block_size_[1]];

            rbl = std::min(block_size_[0], len_patch_[0][patch_[0]] - off_patch_[0]);
            cbl = std::min(block_size_[1], len_patch_[1][patch_[1]] - off_patch_[1]);

            data = data_ + (rbs ? *rscat : 0) + (cbs ? *cscat : 0);
        }

        void shift(unsigned dim, len_type n)
        {
            TBLIS_ASSERT(dim < 2);

            off_[dim] += n;

            n += off_patch_[dim];
            off_patch_[dim] = 0;

            while (n < 0)
                n += len_patch_[dim][--patch_[dim]];

            while (n > len_patch_[dim][patch_[dim]])
                n -= len_patch_[dim][patch_[dim]++];

            off_patch_[dim] = n;
        }

        void shift_down(unsigned dim)
        {
            shift(dim, length(dim));
        }

        void shift_up(unsigned dim)
        {
            shift(dim, -length(dim));
        }

        void pack(const communicator& comm, const config& cfg,
                  bool trans, normal_matrix<T>& Ap) const
        {
            const len_type MR = (!trans ? cfg.gemm_mr.def<T>()
                                        : cfg.gemm_nr.def<T>());
            const len_type ME = (!trans ? cfg.gemm_mr.extent<T>()
                                        : cfg.gemm_nr.extent<T>());
            const len_type KR = cfg.gemm_kr.def<T>();

            TBLIS_ASSERT(A.block_size(0) == (!trans ? MR : KR));
            TBLIS_ASSERT(A.block_size(1) == (!trans ? KR : MR));

            const len_type m_a = len_[ trans];
            const len_type k_a = len_[!trans];

            comm.distribute_over_threads({m_a, MR}, {k_a, KR},
            [&](len_type m_first, len_type m_last, len_type k_first, len_type k_last)
            {
                T* p_ap = Ap.data() + (m_first/MR)*ME*k_a + k_first*ME;

                len_type m_off_patch = m_first + off_patch_[ trans];
                len_type k_off_patch = k_first + off_patch_[!trans];
                unsigned m_patch = patch_[ trans];
                unsigned k_patch = patch_[!trans];

                while (m_first > len_patch_[trans][m_patch])
                    m_off_patch -= len_patch_[trans][m_patch++];

                while (k_first > len_patch_[!trans][k_patch])
                    k_off_patch -= len_patch_[!trans][k_patch++];

                len_type m_off = m_first;
                while (m_off < m_last)
                {
                    len_type m_len_patch = len_patch_[ trans][m_patch];

                    while (m_off_patch < m_len_patch && m_off < m_last)
                    {
                        len_type m_loc = std::min({MR, m_last-m_off,
                            m_len_patch-m_off_patch});

                        unsigned k_patch_old = k_patch;
                        len_type k_off_patch_old = k_off_patch;

                        len_type k_off = k_first;
                        while (k_off < k_last)
                        {
                            len_type k_len_patch = len_patch_[!trans][k_patch];
                            len_type k_loc = std::min(k_last-k_off,
                                k_len_patch-k_off_patch);

                            scatter_type rscat_a = scatter_[ trans][!trans ? m_patch : k_patch]
                                                                   [!trans ? k_patch : m_patch] + m_off_patch;
                            scatter_type cscat_a = scatter_[!trans][!trans ? m_patch : k_patch]
                                                                   [!trans ? k_patch : m_patch] + k_off_patch;
                            stride_type rs_a = block_stride_[ trans][!trans ? m_patch : k_patch]
                                                                    [!trans ? k_patch : m_patch][m_off_patch/MR];
                            scatter_type cbs_a = block_stride_[!trans][!trans ? m_patch : k_patch]
                                                                      [!trans ? k_patch : m_patch] + k_off_patch/KR;
                            const T* p_a = data_ + (rs_a ? *rscat_a : 0);

                            if (rs_a)
                            {
                                if (!trans)
                                    cfg.pack_nb_mr_ukr.call<T>(m_loc, k_loc, p_a, rs_a, cscat_a, cbs_a, p_ap);
                                else
                                    cfg.pack_nb_nr_ukr.call<T>(m_loc, k_loc, p_a, rs_a, cscat_a, cbs_a, p_ap);
                            }
                            else
                            {
                                if (!trans)
                                    cfg.pack_sb_mr_ukr.call<T>(m_loc, k_loc, p_a, rscat_a, cscat_a, cbs_a, p_ap);
                                else
                                    cfg.pack_sb_nr_ukr.call<T>(m_loc, k_loc, p_a, rscat_a, cscat_a, cbs_a, p_ap);
                            }

                            p_ap += ME*k_loc;
                            k_off += round_up(k_loc, KR);
                            k_patch++;
                            k_off_patch = 0;
                        }

                        k_patch = k_patch_old;
                        k_off_patch = k_off_patch_old;

                        m_off += MR;
                    }

                    m_patch++;
                    m_off_patch = 0;
                }
            });
        }
};

}

#endif
