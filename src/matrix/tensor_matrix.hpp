#ifndef _TBLIS_TENSOR_MATRIX_HPP_
#define _TBLIS_TENSOR_MATRIX_HPP_

#include "util/basic_types.h"

namespace tblis
{

template <typename T>
class tensor_matrix
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
        std::array<len_type, 2> offset_;
        std::array<len_type, 2> leading_len_;
        std::array<stride_type, 2> leading_stride_;
        std::array<MArray::viterator<>, 2> iterator_;

    public:
        tensor_matrix()
        {
            reset();
        }

        tensor_matrix(const tensor_matrix& other)
        {
            reset(other);
        }

        tensor_matrix(tensor_matrix&& other)
        {
            reset(std::move(other));
        }

        template <typename U, typename V>
        tensor_matrix(tensor_view<T> other,
                      const std::vector<U>& row_inds,
                      const std::vector<V>& col_inds)
        {
            reset(std::move(other), row_inds, col_inds);
        }

        template <typename U, typename V, typename W, typename X>
        tensor_matrix(const std::vector<U>& len_m,
                      const std::vector<V>& len_n,
                      pointer ptr,
                      const std::vector<W>& stride_m,
                      const std::vector<X>& stride_n)
        {
            reset(len_m, len_n, ptr, stride_m, stride_n);
        }

        tensor_matrix& operator=(const tensor_matrix& other) = delete;

        void reset()
        {
            data_ = nullptr;
            len_[0] = 0;
            len_[1] = 0;
            offset_[0] = 0;
            offset_[1] = 0;
            leading_len_[0] = 0;
            leading_len_[1] = 0;
            leading_stride_[0] = 0;
            leading_stride_[1] = 0;
            iterator_[0] = MArray::viterator<>();
            iterator_[1] = MArray::viterator<>();
        }

        void reset(const tensor_matrix& other)
        {
            data_ = other.data_;
            len_[0] = other.len_[0];
            len_[1] = other.len_[1];
            offset_[0] = other.offset_[0];
            offset_[1] = other.offset_[1];
            leading_len_[0] = other.leading_len_[0];
            leading_len_[1] = other.leading_len_[1];
            leading_stride_[0] = other.leading_stride_[0];
            leading_stride_[1] = other.leading_stride_[1];
            iterator_[0] = other.iterator_[0];
            iterator_[1] = other.iterator_[1];
        }

        void reset(tensor_matrix&& other)
        {
            data_ = other.data_;
            len_[0] = other.len_[0];
            len_[1] = other.len_[1];
            offset_[0] = other.offset_[0];
            offset_[1] = other.offset_[1];
            leading_len_[0] = other.leading_len_[0];
            leading_len_[1] = other.leading_len_[1];
            leading_stride_[0] = other.leading_stride_[0];
            leading_stride_[1] = other.leading_stride_[1];
            iterator_[0] = std::move(other.iterator_[0]);
            iterator_[1] = std::move(other.iterator_[1]);
        }

        template <typename U, typename V>
        void reset(tensor_view<T> other,
                   const std::vector<U>& row_inds,
                   const std::vector<V>& col_inds)
        {
            std::vector<len_type> len_m(row_inds.size());
            std::vector<len_type> len_n(col_inds.size());
            std::vector<stride_type> stride_m(row_inds.size());
            std::vector<stride_type> stride_n(col_inds.size());

            for (size_t i = 0;i < row_inds.size();i++)
            {
                len_m[i-1] = other.length(row_inds[i]);
                stride_m[i-1] = other.stride(row_inds[i]);
            }

            for (size_t i = 0;i < col_inds.size();i++)
            {
                len_n[i-1] = other.length(col_inds[i]);
                stride_n[i-1] = other.stride(col_inds[i]);
            }

            reset(len_m, len_n, other.data(), stride_m, stride_n);
        }

        template <typename U, typename V, typename W, typename X>
        void reset(const std::vector<U>& len_m,
                   const std::vector<V>& len_n,
                   pointer ptr,
                   const std::vector<W>& stride_m,
                   const std::vector<X>& stride_n)
        {
            TBLIS_ASSERT(len_m.size() == stride_m.size());
            TBLIS_ASSERT(len_n.size() == stride_n.size());

            data_ = ptr;
            len_[0] = leading_len_[0] = (len_m.empty() ? 1 : len_m[0]);
            len_[1] = leading_len_[1] = (len_n.empty() ? 1 : len_n[0]);
            leading_stride_[0] = (stride_m.empty() ? 1 : stride_m[0]);
            leading_stride_[1] = (stride_n.empty() ? 1 : stride_n[0]);
            offset_[0] = 0;
            offset_[1] = 0;

            std::vector<len_type> len_m_, len_n_;
            std::vector<stride_type> stride_m_, stride_n_;
            if (!len_m.empty()) len_m_.assign(len_m.begin()+1, len_m.end());
            if (!len_n.empty()) len_n_.assign(len_n.begin()+1, len_n.end());
            if (!stride_m.empty()) stride_m_.assign(stride_m.begin()+1, stride_m.end());
            if (!stride_n.empty()) stride_n_.assign(stride_n.begin()+1, stride_n.end());

            for (len_type len : len_m_) len_[0] *= len;
            for (len_type len : len_n_) len_[1] *= len;

            iterator_[0] = MArray::viterator<>(len_m_, stride_m_);
            iterator_[1] = MArray::viterator<>(len_n_, stride_n_);
        }

        void transpose()
        {
            using std::swap;
            swap(len_[0], len_[1]);
            swap(offset_[0], offset_[1]);
            swap(leading_len_[0], leading_len_[1]);
            swap(leading_stride_[0], leading_stride_[1]);
            swap(iterator_[0], iterator_[1]);
        }

        void swap(tensor_matrix& other)
        {
            using std::swap;
            swap(data_, other.data_);
            swap(len_, other.len_);
            swap(offset_, other.offset_);
            swap(leading_len_, other.leading_len_);
            swap(leading_stride_, other.leading_stride_);
            swap(iterator_, other.iterator_);
        }

        friend void swap(tensor_matrix& a, tensor_matrix& b)
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
        }

        void shift_down(unsigned dim)
        {
            shift(dim, len_[dim]);
        }

        void shift_up(unsigned dim)
        {
            shift(dim, -len_[dim]);
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

        void fill_scatter(unsigned dim, stride_type* scatter)
        {
            TBLIS_ASSERT(dim < 2);

            len_type m = len_[dim];
            len_type off_m = offset_[dim];
            len_type m0 = leading_len_[dim];
            stride_type s0 = leading_stride_[dim];
            auto& it = iterator_[dim];

            len_type p0 = off_m%m0;
            stride_type off = 0;
            it.position(off_m/m0, off);

            for (len_type idx = 0;it.next(off);)
            {
                for (len_type i0 = p0;i0 < m0;i0++)
                {
                    if (idx == m) return;
                    scatter[idx++] = off + i0*s0;
                }
                p0 = 0;
            }
        }

        void fill_block_scatter(unsigned dim, stride_type* scatter, len_type MB, stride_type* block_scatter)
        {
            /*
            TBLIS_ASSERT(dim < 2);

            const auto& m = len_[dim];
            const auto& off_m = offset_[dim];
            const auto& m0 = leading_len_[dim];
            const auto& s0 = leading_stride_[dim];
            auto& it = iterator_[dim];

            len_type p0 = off_m%m0;
            stride_type off = 0;
            it.position(off_m/m0, off);

            len_type nleft = 0;
            for (len_type idx = 0, bidx = 0;it.next(off);)
            {
                for (len_type i0 = p0;i0 < m0;i0++)
                {
                    if (idx == m) return;

                    if (nleft == 0)
                    {
                        block_scatter[bidx++] = (m0-i0 >= MR || m0-i0+idx >= m ? s0 : 0);
                        //block_scatter[bidx++] = 0;
                        nleft = MR;
                    }

                    scatter[idx++] = off + i0*s0;
                    nleft--;
                }
                p0 = 0;
            }
            */

            fill_scatter(dim, scatter);

            len_type m = len_[dim];

            for (len_type i = 0, b = 0;i < m;i += MB, b++)
            {
                stride_type s = (m-i) > 1 ? scatter[i+1]-scatter[i] : 1;
                for (len_type j = i+1;j+1 < std::min(i+MB,m);j++)
                {
                    if (scatter[j+1]-scatter[j] != s) s = 0;
                }
                block_scatter[b] = s;
            }
        }
};

}

#endif
