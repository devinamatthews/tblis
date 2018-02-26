#ifndef _TBLIS_TENSOR_MATRIX_HPP_
#define _TBLIS_TENSOR_MATRIX_HPP_

#include "util/basic_types.h"

namespace tblis
{

template <typename T>
class tensor_matrix
{
    template <typename> friend class block_scatter_matrix;

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
        std::array<len_type, 2> tot_len_ = {};
        std::array<len_type, 2> offset_ = {};
        std::array<len_vector, 2> len_ = {};
        std::array<stride_vector, 2> stride_ = {};

    public:
        tensor_matrix();

        template <typename U, typename V>
        tensor_matrix(varray_view<const T> other,
                      const U& row_inds,
                      const V& col_inds)
        {
            TBLIS_ASSERT(row_inds.size()+col_inds.size() == other.dimension());

            data_ = const_cast<T*>(other.data());

            for (unsigned i = 0;i < row_inds.size();i++)
            {
                len_[0].push_back(other.length(row_inds[i]));
                stride_[0].push_back(other.stride(row_inds[i]));
                tot_len_[0] *= other.length(row_inds[i]);
            }

            for (unsigned i = 0;i < col_inds.size();i++)
            {
                len_[1].push_back(other.length(col_inds[i]));
                stride_[1].push_back(other.stride(col_inds[i]));
                tot_len_[1] *= other.length(col_inds[i]);
            }
        }

        template <typename U, typename V, typename W, typename X>
        tensor_matrix(const U& len_m,
                      const V& len_n,
                      pointer ptr,
                      const W& stride_m,
                      const X& stride_n)
        {
            TBLIS_ASSERT(len_m.size() == stride_m.size());
            TBLIS_ASSERT(len_n.size() == stride_n.size());

            data_ = ptr;

            len_[0].assign(len_m.begin(), len_m.end());
            len_[1].assign(len_n.begin(), len_n.end());
            stride_[0].assign(stride_m.begin(), stride_m.end());
            stride_[1].assign(stride_n.begin(), stride_n.end());

            for (len_type len : len_[0]) tot_len_[0] *= len;
            for (len_type len : len_[1]) tot_len_[1] *= len;
        }

        void transpose()
        {
            using std::swap;
            swap(tot_len_[0], tot_len_[1]);
            swap(offset_[0], offset_[1]);
            swap(len_[0], len_[1]);
            swap(stride_[0], stride_[1]);
        }

        void swap(tensor_matrix& other)
        {
            using std::swap;
            swap(data_, other.data_);
            swap(tot_len_, other.tot_len_);
            swap(offset_, other.offset_);
            swap(len_, other.len_);
            swap(stride_, other.stride_);
        }

        friend void swap(tensor_matrix& a, tensor_matrix& b)
        {
            a.swap(b);
        }

        len_type length(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return tot_len_[dim];
        }

        len_type length(unsigned dim, len_type m)
        {
            TBLIS_ASSERT(dim < 2);
            std::swap(m, tot_len_[dim]);
            return m;
        }

        stride_type stride(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return (stride_[dim].empty() ? 1 : stride_[dim][0]);
        }

        void shift(unsigned dim, len_type n)
        {
            TBLIS_ASSERT(dim < 2);
            offset_[dim] += n;
        }

        void shift_down(unsigned dim)
        {
            shift(dim, tot_len_[dim]);
        }

        void shift_up(unsigned dim)
        {
            shift(dim, -tot_len_[dim]);
        }

        pointer data()
        {
            return data_;
        }

        const_pointer data() const
        {
            return data_;
        }

        pointer data(pointer ptr)
        {
            using std::swap;
            swap(ptr, data_);
            return ptr;
        }

        constexpr unsigned num_patches(unsigned dim) const
        {
            return 1;
        }
};

}

#endif
