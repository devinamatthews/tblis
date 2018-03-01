#ifndef _TBLIS_TENSOR_MATRIX_HPP_
#define _TBLIS_TENSOR_MATRIX_HPP_

#include "util/basic_types.h"

#include "normal_matrix.hpp"

namespace tblis
{

template <typename T>
class tensor_matrix : public abstract_matrix<T>
{
    template <typename> friend class block_scatter_matrix;
    template <typename> friend class patch_block_scatter_matrix;

    public:
        typedef const stride_type* scatter_type;

    protected:
        using abstract_matrix<T>::data_;
        using abstract_matrix<T>::tot_len_;
        using abstract_matrix<T>::cur_len_;
        using abstract_matrix<T>::off_;
        std::array<len_vector, 2> lens_ = {};
        std::array<stride_vector, 2> strides_ = {};

    public:
        tensor_matrix();

        template <typename U, typename V>
        tensor_matrix(varray_view<const T> other,
                      const U& row_inds,
                      const V& col_inds)
        {
            TBLIS_ASSERT(row_inds.size()+col_inds.size() == other.dimension());

            data_ = const_cast<T*>(other.data());

            tot_len_ = {1, 1};

            for (unsigned i = 0;i < row_inds.size();i++)
            {
                lens_[0].push_back(other.length(row_inds[i]));
                strides_[0].push_back(other.stride(row_inds[i]));
                tot_len_[0] *= other.length(row_inds[i]);
            }

            for (unsigned i = 0;i < col_inds.size();i++)
            {
                lens_[1].push_back(other.length(col_inds[i]));
                strides_[1].push_back(other.stride(col_inds[i]));
                tot_len_[1] *= other.length(col_inds[i]);
            }

            cur_len_ = tot_len_;
        }

        template <typename U, typename V, typename W, typename X>
        tensor_matrix(const U& len_m,
                      const V& len_n,
                      T* ptr,
                      const W& stride_m,
                      const X& stride_n)
        {
            TBLIS_ASSERT(len_m.size() == stride_m.size());
            TBLIS_ASSERT(len_n.size() == stride_n.size());

            data_ = ptr;

            lens_[0].assign(len_m.begin(), len_m.end());
            lens_[1].assign(len_n.begin(), len_n.end());
            strides_[0].assign(stride_m.begin(), stride_m.end());
            strides_[1].assign(stride_n.begin(), stride_n.end());

            tot_len_ = {1, 1};

            for (len_type len : lens_[0]) tot_len_[0] *= len;
            for (len_type len : lens_[1]) tot_len_[1] *= len;

            cur_len_ = tot_len_;
        }

        T* data() const
        {
            return data_;
        }

        stride_type stride(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return strides_[dim].empty() ? 1 : strides_[dim][0];
        }

        std::array<stride_type, 2> strides() const
        {
            return {stride(0), stride(1)};
        }

        void transpose()
        {
            using std::swap;
            abstract_matrix<T>::transpose();
            swap(lens_[0], lens_[1]);
            swap(strides_[0], strides_[1]);
        }

        constexpr unsigned num_patches(unsigned dim) const
        {
            return 1;
        }
};

}

#endif
