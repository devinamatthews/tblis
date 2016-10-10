#ifndef _TBLIS_BLOCK_SCATTER_MATRIX_HPP_
#define _TBLIS_BLOCK_SCATTER_MATRIX_HPP_

#include "tblis.hpp"

namespace tblis
{

template <typename T, len_type MB, len_type NB>
class block_scatter_matrix
{
    static_assert(MB > 0, "MB must be positive");
    static_assert(NB > 0, "NB must be positive");

    public:
        typedef ssize_t idx_type;
        typedef size_t size_type;
        typedef ptrdiff_t stride_type;
        typedef const stride_type* scatter_type;
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;

    protected:
        pointer data_;
        std::array<idx_type, 2> len_;
        std::array<scatter_type, 2> block_scatter_;
        std::array<scatter_type, 2> scatter_;

        constexpr static bool M_BLOCKED = MB > 1;
        constexpr static bool N_BLOCKED = NB > 1;

    public:
        block_scatter_matrix()
        {
            reset();
        }

        block_scatter_matrix(const block_scatter_matrix&) = default;

        block_scatter_matrix(idx_type m, idx_type n, pointer p,
                             scatter_type rbs, scatter_type cbs,
                             scatter_type rscat, scatter_type cscat)
        {
            reset(m, n, p, rbs, cbs, rscat, cscat);
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

        void reset(idx_type m, idx_type n, pointer p, scatter_type rbs, scatter_type cbs,
                   scatter_type rscat, scatter_type cscat)
        {
            data_ = p;
            len_[0] = m;
            len_[1] = n;
            block_scatter_[0] = rbs;
            block_scatter_[1] = cbs;
            scatter_[0] = rscat;
            scatter_[1] = cscat;

            for (idx_type i = 0;i < m;i += MB)
            {
                stride_type s = (m-i) > 1 ? rscat[i+1]-rscat[i] : -1;
                for (int j = i+1;j+1 < std::min(i+MB,m);j++)
                {
                    if (rscat[j+1]-rscat[j] != s) s = 0;
                }
                if (M_BLOCKED) TBLIS_ASSERT(s == -1 || s == rbs[i/MB]);
            }

            for (idx_type i = 0;i < n;i += NB)
            {
                stride_type s = (n-i) > 1 ? cscat[i+1]-cscat[i] : -1;
                for (int j = i+1;j+1 < std::min(i+NB,n);j++)
                {
                    if (cscat[j+1]-cscat[j] != s) s = 0;
                }
                if (N_BLOCKED) TBLIS_ASSERT(s == -1 || s == cbs[i/NB]);
            }
        }

        idx_type length(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return len_[dim];
        }

        idx_type length(unsigned dim, idx_type m)
        {
            TBLIS_ASSERT(dim < 2);
            std::swap(m, len_[dim]);
            return m;
        }

        stride_type stride(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            if (dim == 0)
                return (M_BLOCKED ? *block_scatter_[0] : 0);
            else
                return (N_BLOCKED ? *block_scatter_[1] : 0);
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

        void shift(unsigned dim, idx_type n)
        {
            TBLIS_ASSERT(dim < 2);
            scatter_[dim] += n;
            block_scatter_[dim] += ceil_div(n, (dim ? NB : MB));
        }

        void shift_down(unsigned dim)
        {
            shift(dim, length(dim));
        }

        void shift_up(unsigned dim)
        {
            shift(dim, -length(dim));
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
};

}

#endif
