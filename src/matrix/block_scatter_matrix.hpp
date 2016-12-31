#ifndef _TBLIS_BLOCK_SCATTER_MATRIX_HPP_
#define _TBLIS_BLOCK_SCATTER_MATRIX_HPP_

#include "util/basic_types.h"

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
        pointer data_;
        std::array<len_type, 2> len_;
        std::array<scatter_type, 2> block_scatter_;
        std::array<scatter_type, 2> scatter_;
        std::array<len_type, 2> block_size_;

    public:
        block_scatter_matrix()
        {
            reset();
        }

        block_scatter_matrix(const block_scatter_matrix&) = default;

        block_scatter_matrix(len_type m, len_type n, pointer p,
                             scatter_type rscat, len_type MB, scatter_type rbs,
                             scatter_type cscat, len_type NB, scatter_type cbs)
        {
            reset(m, n, p, rscat, MB, rbs, cscat, NB, cbs);
        }

        block_scatter_matrix& operator=(const block_scatter_matrix&) = delete;

        void reset()
        {
            data_ = nullptr;
            len_[0] = 0;
            len_[1] = 0;
            block_scatter_[0] = nullptr;
            block_scatter_[1] = nullptr;
            scatter_[0] = nullptr;
            scatter_[1] = nullptr;
        }

        void reset(const block_scatter_matrix& other)
        {
            data_ = other.data_;
            len_[0] = other.len_[0];
            len_[1] = other.len_[1];
            block_scatter_[0] = other.block_scatter_[0];
            block_scatter_[1] = other.block_scatter_[1];
            scatter_[0] = other.scatter_[0];
            scatter_[1] = other.scatter_[1];
        }

        void reset(len_type m, len_type n, pointer p,
                   scatter_type rscat, len_type MB, scatter_type rbs,
                   scatter_type cscat, len_type NB, scatter_type cbs)
        {
            data_ = p;
            len_[0] = m;
            len_[1] = n;
            block_scatter_[0] = rbs;
            block_scatter_[1] = cbs;
            scatter_[0] = rscat;
            scatter_[1] = cscat;
            block_size_[0] = MB;
            block_size_[1] = NB;

            for (len_type i = 0;i < m;i += MB)
            {
                stride_type s = (m-i) > 1 ? rscat[i+1]-rscat[i] : 1;
                for (len_type j = i+1;j+1 < std::min(i+MB,m);j++)
                {
                    if (rscat[j+1]-rscat[j] != s) s = 0;
                }
                TBLIS_ASSERT(s == -1 || s == rbs[i/MB]);
            }

            for (len_type i = 0;i < n;i += NB)
            {
                stride_type s = (n-i) > 1 ? cscat[i+1]-cscat[i] : 1;
                for (len_type j = i+1;j+1 < std::min(i+NB,n);j++)
                {
                    if (cscat[j+1]-cscat[j] != s) s = 0;
                }
                TBLIS_ASSERT(s == -1 || s == cbs[i/NB]);
            }
        }

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

        stride_type stride(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return *block_scatter_[dim];
        }

        scatter_type scatter(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return scatter_[dim];
        }

        scatter_type block_scatter(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return block_scatter_[dim];
        }

        void shift(unsigned dim, len_type n)
        {
            TBLIS_ASSERT(dim < 2);
            scatter_[dim] += n;
            block_scatter_[dim] += ceil_div(n, block_size_[dim]);
        }

        void shift_down(unsigned dim)
        {
            shift(dim, length(dim));
        }

        void shift_up(unsigned dim)
        {
            shift(dim, -length(dim));
        }

        void shift_block(unsigned dim, len_type n)
        {
            TBLIS_ASSERT(dim < 2);
            scatter_[dim] += n*block_size_[dim];
            block_scatter_[dim] += n;
        }

        pointer data()
        {
            return data_ + (stride(0) == 0 ? 0 : *scatter_[0])
                         + (stride(1) == 0 ? 0 : *scatter_[1]);
        }

        const_pointer data() const
        {
            return const_cast<block_scatter_matrix&>(*this).data();
        }

        pointer raw_data() { return data_; }

        const_pointer raw_data() const { return data_; }
};

}

#endif
