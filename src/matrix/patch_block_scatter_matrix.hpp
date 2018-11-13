#ifndef _TBLIS_PATCH_BLOCK_SCATTER_MATRIX_HPP_
#define _TBLIS_PATCH_BLOCK_SCATTER_MATRIX_HPP_

#include "util/basic_types.h"
#include "util/macros.h"
#include "util/thread.h"
#include "util/tensor.hpp"

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
        typedef block_scatter_matrix<T>* patch_type;

    protected:
        using abstract_matrix<T>::data_;
        using abstract_matrix<T>::tot_len_;
        using abstract_matrix<T>::cur_len_;
        using abstract_matrix<T>::off_;
        matrix_view<block_scatter_matrix<T>> patches_;
        std::array<unsigned, 2> patch_ = {};
        std::array<len_type, 2> patch_off_ = {};
        std::array<len_type, 2> block_size_ = {};

    public:
        patch_block_scatter_matrix() {}

        patch_block_scatter_matrix(const communicator& comm, const tensor_matrix<T>& A,
                                   len_type MB, len_type, stride_type* rscat, stride_type* rbs,
                                   len_type NB, len_type, stride_type* cscat, stride_type* cbs,
                                   patch_type patches)
        {
            data_ = A.data_;
            tot_len_ = cur_len_ = {A.cur_len_[0], A.cur_len_[1]};
            block_size_ = {MB, NB};

            patches_.reset({1, 1}, patches);

            if (comm.master())
            {
                auto& patch = patches_[0][0];

                patch.data_ = data_;
                patch.off_ = {};
                patch.tot_len_ = patch.cur_len_ = cur_len_;
                patch.scatter_ = {rscat, cscat};
                patch.block_stride_ = {rbs, cbs};
                patch.block_size_ = block_size_;

                patch.fill_block_scatter(A.lens_[0], A.strides_[0], MB, A.off_[0],
                                         tot_len_[0], rscat, rbs, A.pack_3d_[0]);
                patch.fill_block_scatter(A.lens_[1], A.strides_[1], NB, A.off_[1],
                                         tot_len_[1], cscat, cbs, A.pack_3d_[1]);
            }

            comm.barrier();
        }

        patch_block_scatter_matrix(const communicator& comm, const dpd_tensor_matrix<T>& A,
                                   len_type MB, len_type ME, stride_type* rscat_, stride_type* rbs_,
                                   len_type NB, len_type NE, stride_type* cscat_, stride_type* cbs_,
                                   patch_type patches)
        {
            block_size_ = {MB, NB};
            std::array<len_type, 2> block_round = {ME, NE};

            data_ = A.tensor_.data();
            unsigned nirrep = A.tensor_.num_irreps();
            unsigned irrep_mask = nirrep - 1;
            unsigned irrep_bits = __builtin_popcount(irrep_mask);

            std::array<stride_type*, 2> scat = {rscat_, cscat_};
            std::array<stride_type*, 2> bs = {rbs_, cbs_};
            std::array<stride_type*, 2> scat_max = {cscat_, rbs_};
            std::array<stride_type*, 2> bs_max = {cbs_, cbs_ + (rbs_ - cscat_)};

            TBLIS_ASSERT(A.block_offset_[0] >= 0);
            TBLIS_ASSERT(A.block_offset_[1] >= 0);

            std::array<unsigned, 2> npatch = {};
            std::array<len_type, 2> first_size =
                {A.block_size_[0][A.block_[0]] - A.block_offset_[0],
                 A.block_size_[1][A.block_[1]] - A.block_offset_[1]};
            std::array<len_type, 2> last_size;

            for (unsigned dim : {0,1})
            {
                len_type off = 0;
                len_type block_off = A.block_offset_[dim];
                unsigned block = A.block_[dim];

                while (off < A.cur_len_[dim])
                {
                    last_size[dim] = A.cur_len_[dim] - off;
                    len_type loc = std::min(last_size[dim],
                        A.block_size_[dim][block] - block_off);
                    tot_len_[dim] += round_up(loc, block_round[dim]);
                    off += loc;
                    block_off = 0;
                    block++;
                }

                npatch[dim] = block - A.block_[dim];

                if (npatch[dim] == 1)
                {
                    first_size[dim] = last_size[dim] =
                        std::min(first_size[dim], last_size[dim]);
                }
            }

            TBLIS_ASSERT(tot_len_[0] % ME == 0);
            TBLIS_ASSERT(tot_len_[1] % NE == 0);

            cur_len_ = tot_len_;

            patches_.reset(npatch, patches);

            comm.do_tasks_deferred(npatch[0]*npatch[1],
            [&](auto& tasks)
            {
                std::array<unsigned,2> patch;

                for (patch[0] = 0;patch[0] < npatch[0];patch[0]++)
                {
                    for (patch[1] = 0;patch[1] < npatch[1];patch[1]++)
                    {
                        std::array<unsigned, 2> block =
                            {A.block_[0]+patch[0], A.block_[1]+patch[1]};
                        std::array<len_type, 2> loc =
                            {patch[0] == 0 ? first_size[0] :
                             patch[0] == npatch[0]-1 ? last_size[0] :
                             A.block_size_[0][block[0]],
                             patch[1] == 0 ? first_size[1] :
                             patch[1] == npatch[1]-1 ? last_size[1] :
                             A.block_size_[1][block[1]]};

                        tasks.visit(patch[0] + patch[1]*npatch[0],
                        [&,patch,block,loc,scat,bs]
                        (const communicator& comm)
                        {
                            if (!comm.master()) return;

                            irrep_vector irreps(A.tensor_.dimension());

                            for (unsigned i = 0;i < A.extra_dims_.size();i++)
                                irreps[A.extra_dims_[i]] = A.extra_irreps_[i];

                            for (unsigned dim : {0,1})
                            {
                                if (A.dims_[dim].empty()) continue;

                                unsigned idx = A.block_idx_[dim][block[dim]];
                                irreps[A.dims_[dim][0]] = A.irrep_[dim];
                                for (unsigned i = 1;i < A.dims_[dim].size();i++)
                                {
                                    irreps[A.dims_[dim][0]] ^=
                                        irreps[A.dims_[dim][i]] = idx & irrep_mask;
                                    idx >>= irrep_bits;
                                }
                            }

                            TBLIS_ASSERT(loc[0] != 0);
                            TBLIS_ASSERT(loc[1] != 0);
                            TBLIS_ASSERT(scat[0]+loc[0] <= scat_max[0]);
                            TBLIS_ASSERT(scat[1]+loc[1] <= scat_max[1]);
                            TBLIS_ASSERT(bs[0]+loc[0] <= bs_max[0]);
                            TBLIS_ASSERT(bs[1]+loc[1] <= bs_max[1]);

                            auto A2 = A.tensor_(irreps);

                            TBLIS_ASSERT(stl_ext::prod(A2.lengths()) ==
                                         A.block_size_[0][block[0]]*
                                         A.block_size_[1][block[1]]);

                            auto& this_patch = patches_[patch[0]][patch[1]];
                            this_patch.data_ = A2.data();
                            this_patch.off_ = {};
                            this_patch.cur_len_ = loc;
                            this_patch.tot_len_ = {round_up(loc[0], block_round[0]),
                                                   round_up(loc[1], block_round[1])};
                            this_patch.scatter_ = scat;
                            this_patch.block_stride_ = bs;
                            this_patch.block_size_ = block_size_;

                            //printf("loc for %d,%d of %d,%d: %ld (%ld,%ld,%ld), %ld (%ld,%ld,%ld), %ld\n",
                            //       row_patch, col_patch, npatch[0], npatch[1],
                            //       loc[0], first_size[0], A.block_size_[0][row_patch], last_size[0],
                            //       loc[1], first_size[1], A.block_size_[1][col_patch], last_size[1],
                            //       patch.data_ - data_);

                            for (unsigned dim : {0,1})
                            {
                                auto len = stl_ext::select_from(A2.lengths(), A.dims_[dim]);
                                auto stride = stl_ext::select_from(A2.strides(), A.dims_[dim]);

                                block_scatter_matrix<T>::fill_block_scatter(
                                    len, stride, block_size_[dim],
                                    patch[dim] == 0 ? A.block_offset_[dim] : 0,
                                    loc[dim], scat[dim], bs[dim], A.pack_3d_[dim]);
                            }
                        });

                        scat[0] += loc[0];
                        scat[1] += loc[1];
                        bs[0] += loc[0];
                        bs[1] += loc[1];
                    }
                }
            });
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
            swap(patches_[0], patches_[1]);
            swap(patch_[0], patch_[1]);
            swap(patch_off_[0], patch_off_[1]);
            swap(block_size_[0], block_size_[1]);
        }

        void block(T*& data,
                   scatter_type& rscat, stride_type& rbs, len_type& rbl,
                   scatter_type& cscat, stride_type& cbs, len_type& cbl) const
        {
            auto& patch = patches_[patch_[0]][patch_[1]];

            rscat = patch.scatter_[0] + patch_off_[0];
            cscat = patch.scatter_[1] + patch_off_[1];

            TBLIS_ASSERT(patch_off_[0] % block_size_[0] == 0);
            TBLIS_ASSERT(patch_off_[1] % block_size_[1] == 0);

            rbs = patch.block_stride_[0][patch_off_[0]];
            cbs = patch.block_stride_[1][patch_off_[1]];

            rbl = std::min({block_size_[0], patch.cur_len_[0] - patch_off_[0], cur_len_[0]});
            cbl = std::min({block_size_[1], patch.cur_len_[1] - patch_off_[1], cur_len_[1]});

            data = patch.data_ + (rbs ? *rscat : 0) + (cbs ? *cscat : 0);
        }

        void shift(unsigned dim, len_type n)
        {
            abstract_matrix<T>::shift(dim, n);

            auto patch = [&]{ return patches_[patch_[0]][patch_[1]]; };

            n += patch_off_[dim];
            patch_off_[dim] = 0;

            while (n < 0)
            {
                TBLIS_ASSERT(patch_[dim] > 0);
                patch_[dim]--;
                n += patch().tot_len_[dim];
            }

            len_type size;
            while (n > 0 && n >= (size = patch().tot_len_[dim]))
            {
                n -= size;
                patch_[dim]++;
                TBLIS_ASSERT(patch_[dim] <= patches_.length(dim));
                TBLIS_ASSERT(patch_[dim] < patches_.length(dim) || n == 0);
            }

            TBLIS_ASSERT(n >= 0);
            patch_off_[dim] = n;
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

            unsigned m_patch = patch_[ trans];
            unsigned k_patch = patch_[!trans];

            auto patch = [&]{ return patches_[!trans ? m_patch : k_patch]
                                             [!trans ? k_patch : m_patch]; };

            len_type m_off_patch = patch_off_[ trans];
            while (m_off_patch >= patch().tot_len_[trans])
            {
                m_off_patch -= patch().tot_len_[trans];
                m_patch++;
            }

            len_type k_off_patch = patch_off_[!trans];
            while (k_off_patch >= patch().tot_len_[!trans])
            {
                k_off_patch -= patch().tot_len_[!trans];
                k_patch++;
            }

            TBLIS_ASSERT(m_patch < patches_.length( trans) || m_a == 0);
            TBLIS_ASSERT(k_patch < patches_.length(!trans) || k_a == 0);

            unsigned k_patch_old = k_patch;
            len_type k_off_patch_old = k_off_patch;

            normal_matrix<T> Ap_sub = Ap;

            len_type m_off = 0;
            for (;m_off < m_a;)
            {
                len_type m = std::min(m_a - m_off,
                                      patch().cur_len_[trans] - m_off_patch);
                len_type mp = std::min(m_a - m_off,
                                       patch().tot_len_[trans] - m_off_patch);

                len_type k_off = 0;
                for (;k_off < k_a;)
                {
                    TBLIS_ASSERT(m_patch < patches_.length( trans));
                    TBLIS_ASSERT(k_patch < patches_.length(!trans));
                    TBLIS_ASSERT(m_off % MR == 0);
                    TBLIS_ASSERT(m_off_patch % MR == 0);
                    TBLIS_ASSERT(m_off_patch < patch().tot_len_[ trans]);
                    TBLIS_ASSERT(k_off_patch < patch().tot_len_[!trans]);

                    len_type k = std::min(k_a - k_off,
                                          patch().cur_len_[!trans] - k_off_patch);

                    len_type m_old = patch().length( trans, m);
                    len_type k_old = patch().length(!trans, k);
                    patch().shift( trans, m_off_patch);
                    patch().shift(!trans, k_off_patch);

                    Ap_sub.data_ = Ap.data() + ceil_div(m_off, MR)*ME*k_a + k_off*ME;
                    patch().pack(comm, cfg, trans, Ap_sub);

                    patch().shift( trans, -m_off_patch);
                    patch().shift(!trans, -k_off_patch);
                    patch().length( trans, m_old);
                    patch().length(!trans, k_old);

                    k_patch++;
                    k_off += k;
                    k_off_patch = 0;
                }

                k_patch = k_patch_old;
                k_off_patch = k_off_patch_old;

                m_patch++;
                m_off += mp;
                m_off_patch = 0;
            }
        }
};

}

#endif
