#ifndef _TBLIS_SCATTER_TENSOR_MATRIX_HPP_
#define _TBLIS_SCATTER_TENSOR_MATRIX_HPP_

#include "util/basic_types.h"

#include "tensor_matrix.hpp"

namespace tblis
{

template <typename T>
class scatter_tensor_matrix : public tensor_matrix<T>
{
    template <typename> friend class block_scatter_matrix;
    template <typename> friend class patch_block_scatter_matrix;

    public:
        typedef const stride_type* scatter_type;

        static constexpr bool needs_matrify = true;

    protected:
        using tensor_matrix<T>::tot_len_;
        using tensor_matrix<T>::cur_len_;
        using tensor_matrix<T>::off_;
        using tensor_matrix<T>::data_;
        using tensor_matrix<T>::lens_;
        using tensor_matrix<T>::strides_;
        using tensor_matrix<T>::pack_3d_;
        std::array<len_type,2> sub_len_;
        std::array<row_view<const stride_type>,2> scat_;

    public:
        scatter_tensor_matrix();

        template <typename U, typename V>
        scatter_tensor_matrix(varray_view<const T> other,
                              const U& row_inds,
                              const V& col_inds,
                              row_view<const stride_type> row_scat,
                              row_view<const stride_type> col_scat,
                              bool pack_row_3d = false,
                              bool pack_col_3d = false)
        : tensor_matrix<T>{other, row_inds, col_inds, pack_row_3d, pack_col_3d},
          scat_{row_scat, col_scat}
        {
            sub_len_ = tot_len_;

            for (unsigned dim : {0,1})
                if (scat_[dim].length(0))
                    tot_len_[dim] = cur_len_[dim] = sub_len_[dim]*scat_[dim].length(0);
        }

        template <typename U, typename V, typename W, typename X>
        scatter_tensor_matrix(const U& len_m,
                              const V& len_n,
                              T* ptr,
                              const W& stride_m,
                              const X& stride_n,
                              row_view<const stride_type> row_scat,
                              row_view<const stride_type> col_scat,
                              bool pack_m_3d = false,
                              bool pack_n_3d = false)
        : tensor_matrix<T>{len_m, len_n, ptr, stride_m, stride_n, pack_m_3d, pack_n_3d},
          scat_{row_scat, col_scat}
        {
            sub_len_ = tot_len_;

            for (unsigned dim : {0,1})
                if (scat_[dim].length(0))
                    tot_len_[dim] = cur_len_[dim] = sub_len_[dim]*scat_[dim].length(0);
        }

        using tensor_matrix<T>::data;
        using tensor_matrix<T>::stride;
        using tensor_matrix<T>::num_patches;

        void transpose()
        {
            using std::swap;
            tensor_matrix<T>::transpose();
            swap(sub_len_[0], sub_len_[1]);
            swap(scat_[0], scat_[1]);
        }
};

}

#endif
