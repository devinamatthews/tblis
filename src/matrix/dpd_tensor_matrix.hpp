#ifndef _TBLIS_DPD_TENSOR_MATRIX_HPP_
#define _TBLIS_DPD_TENSOR_MATRIX_HPP_

#include "abstract_matrix.hpp"

#include "internal/1t/dpd/util.hpp"

namespace tblis
{

template <typename T>
class dpd_tensor_matrix : public abstract_matrix<T>
{
    template <typename> friend class patch_block_scatter_matrix;

    public:
        typedef const stride_type* scatter_type;

    protected:
        using abstract_matrix<T>::cur_len_;
        using abstract_matrix<T>::tot_len_;
        using abstract_matrix<T>::off_;
        dpd_varray_view<T>& tensor_;
        std::array<dim_vector, 2> dims_ = {};
        dim_vector extra_dims_ = {};
        irrep_vector extra_irreps_ = {};
        len_vector extra_idx_ = {};
        std::array<unsigned, 2> irrep_ = {};
        std::array<unsigned, 2> block_ = {};
        std::array<len_vector, 2> block_size_ = {};
        std::array<len_vector, 2> block_idx_ = {};
        std::array<len_type, 2> block_offset_ = {};
        std::array<stride_type, 2> leading_stride_ = {};
        std::array<bool, 2> pack_3d_ = {};

    public:
        dpd_tensor_matrix();

        template <typename U, typename V, typename W, typename X, typename Y>
        dpd_tensor_matrix(dpd_varray_view<T>& other,
                          const U& row_inds,
                          const V& col_inds,
                          unsigned col_irrep,
                          const W& extra_inds,
                          const X& extra_irreps,
                          const Y& extra_idx,
                          bool pack_row_3d = false,
                          bool pack_col_3d = false)
        : tensor_(other), pack_3d_{pack_row_3d, pack_col_3d}
        {
            TBLIS_ASSERT(row_inds.size()+col_inds.size()+extra_inds.size() ==
                         other.dimension());

            const unsigned nirrep = other.num_irreps();

            dims_[0].assign(row_inds.begin(), row_inds.end());
            dims_[1].assign(col_inds.begin(), col_inds.end());
            irrep_ = {col_irrep^other.irrep(), col_irrep};
            extra_dims_.assign(extra_inds.begin(), extra_inds.end());
            extra_irreps_.assign(extra_irreps.begin(), extra_irreps.end());
            extra_idx_.assign(extra_idx.begin(), extra_idx.end());

            for (unsigned irrep : extra_irreps)
                irrep_[0] ^= irrep;

            TBLIS_ASSERT(dims_[0].size() + dims_[1].size() + extra_dims_.size() == other.dimension());
            TBLIS_ASSERT(extra_dims_.size() == extra_irreps_.size());
            TBLIS_ASSERT(extra_dims_.size() == extra_idx_.size());

            for (unsigned dim : {0,1})
            {
                for (auto& i : dims_[dim])
                    for (auto& j : extra_dims_)
                        TBLIS_ASSERT(i != j);

                if (dims_[dim].empty())
                {
                    tot_len_[dim] = irrep_[dim] == 0 ? 1 : 0;
                    block_size_[dim].push_back(tot_len_[dim]);
                    block_idx_[dim].push_back(0);
                }
                else
                {
                    tot_len_[dim] = 0;
                    internal::irrep_iterator it(irrep_[dim], nirrep, dims_[dim].size());
                    for (unsigned idx = 0;it.next();idx++)
                    {
                        stride_type size = 1;
                        for (unsigned i = 0;i < dims_[dim].size();i++)
                            size *= other.length(dims_[dim][i], it.irrep(i));

                        if (size == 0) continue;

                        block_size_[dim].push_back(size);
                        block_idx_[dim].push_back(idx);
                        tot_len_[dim] += size;
                    }
                }
            }

            cur_len_ = tot_len_;

            std::array<len_vector,1> len;
            std::array<stride_vector,1> stride;
            internal::dense_total_lengths_and_strides(len, stride, other, row_inds);

            leading_stride_ = {row_inds.empty() ? 1 : stride[0][row_inds[0]],
                               col_inds.empty() ? 1 : stride[0][col_inds[0]]};
        }

        template <typename U, typename V, typename W, typename X, typename Y>
        dpd_tensor_matrix(dpd_varray_view<const T>& other,
                          const U& row_inds,
                          const V& col_inds,
                          unsigned col_irrep,
                          const W& extra_inds,
                          const X& extra_irreps,
                          const Y& extra_idx,
                          bool pack_row_3d = false,
                          bool pack_col_3d = false)
        : dpd_tensor_matrix(reinterpret_cast<dpd_varray_view<T>&>(other),
                            row_inds, col_inds, col_irrep,
                            extra_inds, extra_irreps, extra_idx,
                            pack_row_3d, pack_col_3d) {}

        template <typename U, typename V>
        dpd_tensor_matrix(dpd_varray_view<T>& other,
                          const U& row_inds,
                          const V& col_inds,
                          unsigned col_irrep,
                          bool pack_row_3d = false,
                          bool pack_col_3d = false)
        : dpd_tensor_matrix(other, row_inds, col_inds, col_irrep,
                            dim_vector{}, irrep_vector{}, len_vector{},
                            pack_row_3d, pack_col_3d) {}

        template <typename U, typename V>
        dpd_tensor_matrix(dpd_varray_view<const T>& other,
                          const U& row_inds,
                          const V& col_inds,
                          unsigned col_irrep,
                          bool pack_row_3d = false,
                          bool pack_col_3d = false)
        : dpd_tensor_matrix(reinterpret_cast<dpd_varray_view<T>&>(other),
                            row_inds, col_inds, col_irrep,
                            pack_row_3d, pack_col_3d) {}

        dpd_tensor_matrix& operator=(const dpd_tensor_matrix& other) = delete;

        void transpose()
        {
            using std::swap;
            abstract_matrix<T>::transpose();
            swap(dims_[0], dims_[1]);
            swap(irrep_[0], irrep_[1]);
            swap(block_[0], block_[1]);
            swap(block_size_[0], block_size_[1]);
            swap(block_idx_[0], block_idx_[1]);
            swap(block_offset_[0], block_offset_[1]);
            swap(leading_stride_[0], leading_stride_[1]);
            swap(pack_3d_[0], pack_3d_[1]);
        }

        stride_type stride(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return leading_stride_[dim];
        }

        void shift(unsigned dim, len_type n)
        {
            abstract_matrix<T>::shift(dim, n);

            n += block_offset_[dim];
            block_offset_[dim] = 0;

            while (n < 0)
                n += block_size_[dim][--block_[dim]];

            while (n > 0 && n >= block_size_[dim][block_[dim]])
                n -= block_size_[dim][block_[dim]++];

            TBLIS_ASSERT(n >= 0);
            block_offset_[dim] = n;
        }

        unsigned num_patches(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return block_size_[dim].size();
        }

        using abstract_matrix<T>::length;
        using abstract_matrix<T>::lengths;
};

}

#endif
