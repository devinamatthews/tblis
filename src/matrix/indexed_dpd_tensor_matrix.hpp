#ifndef _TBLIS_INDEXED_DPD_TENSOR_MATRIX_HPP_
#define _TBLIS_INDEXED_DPD_TENSOR_MATRIX_HPP_

#include "dpd_tensor_matrix.hpp"

#include "internal/1t/dpd/util.hpp"

namespace tblis
{

template <typename T>
class indexed_dpd_tensor_matrix : public dpd_tensor_matrix<T>
{
    template <typename> friend class indexed_patch_block_scatter_matrix;

    public:
        typedef const stride_type* scatter_type;

        static constexpr bool needs_matrify = true;

    protected:
        using dpd_tensor_matrix<T>::cur_len_;
        using dpd_tensor_matrix<T>::tot_len_;
        using dpd_tensor_matrix<T>::off_;
        using dpd_tensor_matrix<T>::data_;
        using dpd_tensor_matrix<T>::tensor_;
        using dpd_tensor_matrix<T>::dims_;
        using dpd_tensor_matrix<T>::irrep_;
        using dpd_tensor_matrix<T>::block_;
        using dpd_tensor_matrix<T>::block_size_;
        using dpd_tensor_matrix<T>::block_idx_;
        using dpd_tensor_matrix<T>::block_offset_;
        using dpd_tensor_matrix<T>::leading_stride_;
        using dpd_tensor_matrix<T>::pack_3d_;

    public:
        indexed_dpd_tensor_matrix();

        template <typename U, typename V, typename W, unsigned N>
        indexed_dpd_tensor_matrix(dpd_varray_view<T>& other,
                                  const U& row_inds,
                                  const V& col_inds,
                                  unsigned col_irrep,
                                  const group_indices<T, N>& group,
                                  unsigned row_idx,
                                  unsigned col_idx,
                                  const W& indices
                                  bool pack_row_3d = false,
                                  bool pack_col_3d = false)
        : dpd_tensor_matrix<T>(other, row_inds, col_inds, col_irrep, pack_row_3d, pack_col_3d)
        {

        }

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

        using abstract_matrix<T>::data;
        using abstract_matrix<T>::length;
        using abstract_matrix<T>::lengths;
        using abstract_matrix<T>::shift;
        using abstract_matrix<T>::transpose;
};

}

#endif
