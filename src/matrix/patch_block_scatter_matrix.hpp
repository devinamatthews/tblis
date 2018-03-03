#ifndef _TBLIS_PATCH_BLOCK_SCATTER_MATRIX_HPP_
#define _TBLIS_PATCH_BLOCK_SCATTER_MATRIX_HPP_

#include "util/basic_types.h"
#include "util/macros.h"
#include "util/thread.h"

#include "memory/alignment.hpp"

#include "normal_matrix.hpp"
#include "dpd_tensor_matrix.hpp"

namespace tblis
{

template <typename T>
class patch_block_scatter_matrix : public abstract_matrix<T>
{
    public:
        typedef const stride_type* scatter_type;

    protected:
        using abstract_matrix<T>::data_;
        using abstract_matrix<T>::tot_len_;
        using abstract_matrix<T>::cur_len_;
        using abstract_matrix<T>::off_;
        std::array<len_vector, 2> len_patch_ = {};
        std::array<unsigned, 2> patch_ = {};
        std::array<len_type, 2> off_patch_ = {};
        std::array<matrix<stride_type*>, 2> scatter_ = {};
        std::array<matrix<stride_type*>, 2> block_stride_ = {};
        std::array<len_type, 2> block_size_ = {};
        std::array<len_type, 2> block_round_ = {};

    public:
        patch_block_scatter_matrix();

        patch_block_scatter_matrix(const communicator& comm, const tensor_matrix<T>& A,
                                   len_type MB, len_type ME, stride_type* rscat, stride_type* rbs,
                                   len_type NB, len_type NE, stride_type* cscat, stride_type* cbs)
        {
            data_ = A.data_;
            tot_len_ = cur_len_ = {round_up(A.cur_len_[0], ME),
                                   round_up(A.cur_len_[1], NE)};
            len_patch_ = {{{A.cur_len_[0]}, {A.cur_len_[1]}}};
            block_size_ = {MB, NB};
            block_round_ = {ME, NE};
            scatter_[0].reset({1, 1}, rscat);
            scatter_[1].reset({1, 1}, cscat);
            block_stride_[0].reset({1, 1}, rbs);
            block_stride_[1].reset({1, 1}, cbs);

            if (comm.master())
            {
                block_scatter_matrix<T>::fill_block_scatter(
                    A.lens_[0], A.strides_[0], block_size_[0], A.off_[0],
                    tot_len_[0], scatter_[0][0][0], block_stride_[0][0][0]);
                block_scatter_matrix<T>::fill_block_scatter(
                    A.lens_[1], A.strides_[1], block_size_[1], A.off_[1],
                    tot_len_[1], scatter_[1][0][0], block_stride_[1][0][0]);
            }

            comm.barrier();
        }

        patch_block_scatter_matrix(const communicator& comm, const dpd_tensor_matrix<T>& A,
                                   len_type MB, len_type ME, stride_type* rscat, stride_type* rbs,
                                   len_type NB, len_type NE, stride_type* cscat, stride_type* cbs)
        {
            block_size_ = {MB, NB};
            block_round_ = {ME, NE};

            data_ = A.tensor_.data();
            const unsigned nirrep = A.tensor_.num_irreps();
            const auto rscat_max = cscat;
            const auto cscat_max = rbs;
            const auto rbs_max = cbs;
            const auto cbs_max = cbs + (rbs - cscat);

            for (unsigned dim : {0,1})
            {
                len_type off = 0;
                len_type block_off = A.block_offset_[dim];
                unsigned block = A.block_[dim];

                while (off < A.cur_len_[dim])
                {
                    len_type loc = std::min(A.cur_len_[dim]-off,
                        A.block_size_[dim][block]-block_off);
                    len_patch_[dim].push_back(loc);
                    tot_len_[dim] += round_up(loc, block_round_[dim]);
                    off += loc;
                    block_off = 0;
                    block++;
                }
            }

            cur_len_ = tot_len_;

            unsigned row_npatch = len_patch_[0].size();
            unsigned col_npatch = len_patch_[1].size();

            scatter_[0].reset({row_npatch, col_npatch});
            scatter_[1].reset({row_npatch, col_npatch});

            block_stride_[0].reset({row_npatch, col_npatch});
            block_stride_[1].reset({row_npatch, col_npatch});

            irrep_vector irreps(A.tensor_.dimension());

            len_vector row_len(A.dims_[0].size());
            len_vector col_len(A.dims_[1].size());

            stride_vector row_stride(A.dims_[0].size());
            stride_vector col_stride(A.dims_[1].size());

            unsigned nthread = comm.num_threads();
            unsigned tid = comm.thread_num();
            unsigned task = 0;

            irrep_iterator row_it(A.irrep_[0], nirrep, row_len.size());
            unsigned row_idx = 0;
            auto row_block_off = A.block_offset_[0];
            len_type row_offset = 0;
            unsigned row_patch = 0;
            row_it.next();

            while (row_offset < tot_len_[0])
            {
                unsigned new_idx = A.block_idx_[0][A.block_[0]+row_patch];
                while (row_idx < new_idx)
                {
                    row_idx++;
                    row_it.next();
                }
                len_type row_block_size = len_patch_[0][row_patch];

                for (unsigned i = 0;i < A.dims_[0].size();i++)
                    irreps[A.dims_[0][i]] = row_it.irrep(i);

                irrep_iterator col_it(A.irrep_[1], nirrep, col_len.size());
                unsigned col_idx = 0;
                auto col_block_off = A.block_offset_[1];
                len_type col_offset = 0;
                unsigned col_patch = 0;
                col_it.next();

                while (col_offset < tot_len_[1])
                {
                    unsigned new_idx = A.block_idx_[1][A.block_[1]+col_patch];
                    while (col_idx < new_idx)
                    {
                        col_idx++;
                        col_it.next();
                    }
                    len_type col_block_size = len_patch_[1][col_patch];

                    for (unsigned i = 0;i < A.dims_[1].size();i++)
                        irreps[A.dims_[1][i]] = col_it.irrep(i);

                    TBLIS_ASSERT(row_block_size != 0);
                    TBLIS_ASSERT(col_block_size != 0);
                    TBLIS_ASSERT(rscat+row_block_size <= rscat_max);
                    TBLIS_ASSERT(cscat+col_block_size <= cscat_max);
                    TBLIS_ASSERT(rbs+ceil_div(row_block_size,block_size_[0]) <= rbs_max);
                    TBLIS_ASSERT(cbs+ceil_div(col_block_size,block_size_[1]) <= cbs_max);

                    scatter_[0][row_patch][col_patch] = rscat;
                    scatter_[1][row_patch][col_patch] = cscat;

                    block_stride_[0][row_patch][col_patch] = rbs;
                    block_stride_[1][row_patch][col_patch] = cbs;

                    if (task++ % nthread == tid)
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

                        block_scatter_matrix<T>::fill_block_scatter(
                            row_len, row_stride, block_size_[0],
                            row_block_off, row_block_size, rscat, rbs);
                        block_scatter_matrix<T>::fill_block_scatter(
                            col_len, col_stride, block_size_[1],
                            col_block_off, col_block_size, cscat, cbs);

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

                    col_offset += round_up(col_block_size, NE);
                    col_block_off = 0;
                    col_patch++;
                }

                row_offset += round_up(row_block_size, ME);
                row_block_off = 0;
                row_patch++;
            }

            comm.barrier();
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
            swap(len_patch_[0], len_patch_[1]);
            swap(patch_[0], patch_[1]);
            swap(off_patch_[0], off_patch_[1]);
            swap(scatter_[0], scatter_[1]);
            swap(block_stride_[0], block_stride_[1]);
            swap(block_size_[0], block_size_[1]);
        }

        void block(T*& data,
                   scatter_type& rscat, stride_type& rbs, len_type& rbl,
                   scatter_type& cscat, stride_type& cbs, len_type& cbl) const
        {
            rscat = scatter_[0][patch_[0]][patch_[1]] + off_patch_[0];
            cscat = scatter_[1][patch_[0]][patch_[1]] + off_patch_[1];

            TBLIS_ASSERT(off_patch_[0]%block_size_[0] == 0);
            TBLIS_ASSERT(off_patch_[1]%block_size_[1] == 0);

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
            TBLIS_ASSERT(off_[dim] >= 0);
            TBLIS_ASSERT(off_[dim]+cur_len_[dim] <= tot_len_[dim]);

            n += off_patch_[dim];
            off_patch_[dim] = 0;

            len_type size;

            while (n < 0)
            {
                patch_[dim]--;
                n += round_up(len_patch_[dim][patch_[dim]], block_round_[dim]);
            }

            while (n > 0 && n >= (size = round_up(len_patch_[dim][patch_[dim]], block_round_[dim])))
            {
                n -= size;
                patch_[dim]++;
            }

            off_patch_[dim] = n;
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

            comm.distribute_over_threads({m_a, MR}, k_a,
            [&](len_type m_first, len_type m_last, len_type k_first, len_type k_last)
            {
                len_type m_off_patch = m_first + off_patch_[ trans];
                len_type k_off_patch = k_first + off_patch_[!trans];
                unsigned m_patch = patch_[ trans];
                unsigned k_patch = patch_[!trans];
                auto& patch0 = !trans ? m_patch : k_patch;
                auto& patch1 = !trans ? k_patch : m_patch;

                len_type size;

                while (m_off_patch >= (size = round_up(len_patch_[trans][m_patch], block_round_[trans])))
                {
                    m_off_patch -= size;
                    m_patch++;
                }

                while (k_off_patch >= (size = round_up(len_patch_[!trans][k_patch], block_round_[!trans])))
                {
                    k_off_patch -= size;
                    k_patch++;
                }

                if (k_first > 0)
                {
                    k_first -= k_off_patch % KR;
                    k_off_patch -= k_off_patch % KR;
                }

                if (k_last < k_a)
                {
                    unsigned k_last_patch = k_patch;
                    unsigned k_last_off = k_last - k_first + k_off_patch;

                    while (k_last_off >= (size = round_up(len_patch_[!trans][k_last_patch], block_round_[!trans])))
                    {
                        k_last_off -= size;
                        k_last_patch++;
                    }

                    k_last -= k_last_off % KR;
                }

                T* p_ap = Ap.data() + (m_first/MR)*ME*k_a + ME*k_first;

                len_type m_block = 0;
                len_type m_off = m_first;
                while (m_off < m_last)
                {
                    len_type m_len_patch = len_patch_[trans][m_patch];

                    while (m_off_patch < m_len_patch && m_off < m_last)
                    {
                        len_type m = std::min({MR, m_last-m_off,
                            m_len_patch-m_off_patch});

                        unsigned k_patch_old = k_patch;
                        len_type k_off_patch_old = k_off_patch;

                        auto p_ap_old = p_ap;

                        len_type k_off = k_first;
                        while (k_off < k_last)
                        {
                            len_type k_len_patch = len_patch_[!trans][k_patch];
                            len_type k = std::min(k_len_patch - k_off_patch,
                                                  k_last - k_off);
                            len_type kp = round_up(k, block_round_[!trans]);

                            scatter_type rscat_a = scatter_[trans][patch0][patch1] + m_off_patch;
                            scatter_type rbs_a = block_stride_[trans][patch0][patch1] + m_off_patch/MR;
                            scatter_type cscat_a = scatter_[!trans][patch0][patch1] + k_off_patch;
                            scatter_type cbs_a = block_stride_[!trans][patch0][patch1] + k_off_patch/KR;

                            TBLIS_ASSERT(m_off_patch % MR == 0);
                            TBLIS_ASSERT(k_off_patch % KR == 0);
                            TBLIS_ASSERT(p_ap + ME*k <= Ap.data() + Ap.length(0)*Ap.length(1));
                            TBLIS_ASSERT(rscat_a - scatter_[trans][patch0][patch1] < m_len_patch);
                            TBLIS_ASSERT(cscat_a - scatter_[!trans][patch0][patch1] < k_len_patch);
                            TBLIS_ASSERT(rbs_a - block_stride_[trans][patch0][patch1] < ceil_div(m_len_patch, block_size_[trans]));
                            TBLIS_ASSERT(cbs_a - block_stride_[!trans][patch0][patch1] < ceil_div(k_len_patch, block_size_[!trans]));

                            stride_type rs_a = *rbs_a;
                            const T* p_a = data_ + (rs_a ? *rscat_a : 0);

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

                            for (len_type i = ME*k;i < ME*kp;i++)
                                p_ap[i] = T();

                            p_ap += ME*kp;
                            k_off += kp;
                            k_patch++;
                            k_off_patch = 0;
                        }

                        k_patch = k_patch_old;
                        k_off_patch = k_off_patch_old;

                        p_ap = p_ap_old + ME*k_a;
                        m_off += MR;
                        m_off_patch += MR;
                    }

                    m_patch++;
                    m_off_patch = 0;
                }
            });
        }
};

}

#endif
