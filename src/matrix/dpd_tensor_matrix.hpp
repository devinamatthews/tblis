#ifndef _TBLIS_DPD_TENSOR_MATRIX_HPP_
#define _TBLIS_DPD_TENSOR_MATRIX_HPP_

#include "util/basic_types.h"

namespace tblis
{

template <typename T>
class dpd_tensor_matrix
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
        std::array<unsigned, 2> block_ = {};
        dpd_varray_view<T> tensor_;
        std::array<dim_vector, 2> dims_ = {};
        std::array<len_type, 2> len_ = {};
        std::array<len_type, 2> offset_ = {};
        std::array<viterator<0>, 2> iterator_ = {};
        std::array<len_type, 2> block_offset_ = {};
        std::array<stride_type, 2> leading_stride_ = {};

        stride_type block_size(unsigned dim) const
        {
            stride_type size = 1;
            unsigned irr0 = block_[dim];

            for (unsigned i = 1;i < dims_[dim].size();i++)
            {
                irr0 ^= iterator_[dim].position(i-1);
                size += tensor_.length(dims_[dim][i], iterator_[dim].position(i-1));
            }

            if (!dims_[dim].empty())
                size *= tensor_.length(dims_[dim][0], irr0);

            return size;
        }

    public:
        dpd_tensor_matrix();

        template <typename U, typename V>
        dpd_tensor_matrix(dpd_varray_view<T> other,
                          const U& row_inds,
                          const V& col_inds,
                          unsigned block)
        : tensor_(std::move(other))
        {
            TBLIS_ASSERT(row_inds.size()+col_inds.size() == other.dimension());

            const unsigned nirrep = other.num_irreps();

            dims_[0].assign(row_inds.begin(), row_inds.end());
            dims_[1].assign(col_inds.begin(), col_inds.end());
            block_ = {block^other.irrep(), block};
            iterator_ = {viterator<>(std::vector<unsigned>(std::max(size_t(1), row_inds.size())-1, nirrep)),
                         viterator<>(std::vector<unsigned>(std::max(size_t(1), col_inds.size())-1, nirrep))};

            if (row_inds.empty())
            {
                len_[0] = 1;
            }
            else
            {
                stride_vector size(row_inds.size());
                stride_vector newsize(row_inds.size());
                size[0] = 1;

                for (unsigned i = 0;i < row_inds.size();i++)
                {
                    newsize[i] = 0;
                    for (unsigned irr1 = 0;irr1 < nirrep;irr1++)
                        for (unsigned irr2 = 0;irr2 < nirrep;irr2++)
                            newsize[irr1] += size[irr2]*other.length(row_inds[i], irr1^irr2);
                    size.swap(newsize);
                }

                len_[0] = size[block_[0]];
            }

            if (col_inds.empty())
            {
                len_[1] = 1;
            }
            else
            {
                stride_vector size(col_inds.size());
                stride_vector newsize(col_inds.size());
                size[0] = 1;

                for (unsigned i = 0;i < col_inds.size();i++)
                {
                    newsize[i] = 0;
                    for (unsigned irr1 = 0;irr1 < nirrep;irr1++)
                        for (unsigned irr2 = 0;irr2 < nirrep;irr2++)
                            newsize[irr1] += size[irr2]*other.length(col_inds[i], irr1^irr2);
                    size.swap(newsize);
                }

                len_[1] = size[block_[1]];
            }

            len_vector stride(other.dimension());

            for (unsigned i = 0;i < other.dimension();i++)
            {
                for (unsigned irrep = 0;irrep < nirrep;irrep++)
                    stride[other.permutation()[i]] += other.length(i, irrep);
                if (i > 0) stride[i] *= stride[i-1];
            }

            leading_stride_ = {row_inds.empty() ? 1 : stride[other.permutation()[row_inds[0]]],
                               col_inds.empty() ? 1 : stride[other.permutation()[col_inds[0]]]};
        }

        dpd_tensor_matrix& operator=(const dpd_tensor_matrix& other) = delete;

        void transpose()
        {
            using std::swap;
            swap(block_[0], block_[1]);
            swap(len_[0], len_[1]);
            swap(offset_[0], offset_[1]);
            swap(iterator_[0], iterator_[1]);
            swap(block_offset_[0], block_offset_[1]);
            swap(leading_stride_[0], leading_stride_[1]);
        }

        void swap(dpd_tensor_matrix& other)
        {
            using std::swap;
            swap(block_, other.block_);
            swap(tensor_, other.tensor_);
            swap(len_, other.len_);
            swap(offset_, other.offset_);
            swap(iterator_, other.iterator_);
            swap(block_offset_, other.block_offset_);
            swap(leading_stride_, other.leading_stride_);
        }

        friend void swap(dpd_tensor_matrix& a, dpd_tensor_matrix& b)
        {
            a.swap(b);
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

        stride_type stride(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return leading_stride_[dim];
        }

        void shift(unsigned dim, len_type n)
        {
            TBLIS_ASSERT(dim < 2);

            offset_[dim] += n;

            n += block_offset_[dim];
            block_offset_[dim] = 0;

            while (n < 0)
            {
                iterator_[dim].prev();
                n += block_size(dim);
            }

            stride_type size;
            while (n > (size = block_size(dim)))
            {
                n -= size;
                iterator_[dim].next();
            }

            block_offset_[dim] = n;
        }

        void shift_down(unsigned dim)
        {
            shift(dim, len_[dim]);
        }

        void shift_up(unsigned dim)
        {
            shift(dim, -len_[dim]);
        }
};

}

#endif
