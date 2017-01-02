#ifndef _TBLIS_SCATTER_TENSOR_MATRIX_HPP_
#define _TBLIS_SCATTER_TENSOR_MATRIX_HPP_

#include "util/basic_types.h"

namespace tblis
{

template <typename T>
class scatter_tensor_matrix
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
        std::array<len_type, 2> dense_len_;
        std::array<len_type, 2> offset_;
        std::array<len_type, 2> leading_len_;
        std::array<stride_type, 2> leading_stride_;
        std::array<MArray::viterator<>, 2> iterator_;
        std::array<len_type, 2> scatter_len_;
        std::array<scatter_type, 2> scatter_;

    public:
        scatter_tensor_matrix()
        {
            reset();
        }

        scatter_tensor_matrix(const scatter_tensor_matrix& other)
        {
            reset(other);
        }

        scatter_tensor_matrix(scatter_tensor_matrix&& other)
        {
            reset(std::move(other));
        }

        template <typename U, typename V, typename W, typename X>
        scatter_tensor_matrix(const std::vector<U>& dense_len_m,
                              len_type scatter_len_m,
                              const std::vector<V>& dense_len_n,
                              len_type scatter_len_n,
                              pointer ptr,
                              const std::vector<W>& stride_m,
                              scatter_type scatter_m,
                              const std::vector<X>& stride_n,
                              scatter_type scatter_n)
        {
            reset(dense_len_m, scatter_len_m, dense_len_n, scatter_len_n,
                  ptr, stride_m, scatter_m, stride_n, scatter_n);
        }

        scatter_tensor_matrix& operator=(const scatter_tensor_matrix& other) = delete;

        void reset()
        {
            data_ = nullptr;
            dense_len_[0] = 0;
            dense_len_[1] = 0;
            offset_[0] = 0;
            offset_[1] = 0;
            leading_len_[0] = 0;
            leading_len_[1] = 0;
            leading_stride_[0] = 0;
            leading_stride_[1] = 0;
            iterator_[0] = MArray::viterator<>();
            iterator_[1] = MArray::viterator<>();
            scatter_len_[0] = 0;
            scatter_len_[1] = 0;
            scatter_[0] = nullptr;
            scatter_[1] = nullptr;
        }

        void reset(const scatter_tensor_matrix& other)
        {
            data_ = other.data_;
            dense_len_[0] = other.dense_len_[0];
            dense_len_[1] = other.dense_len_[1];
            offset_[0] = other.offset_[0];
            offset_[1] = other.offset_[1];
            leading_len_[0] = other.leading_len_[0];
            leading_len_[1] = other.leading_len_[1];
            leading_stride_[0] = other.leading_stride_[0];
            leading_stride_[1] = other.leading_stride_[1];
            iterator_[0] = other.iterator_[0];
            iterator_[1] = other.iterator_[1];
            scatter_len_[0] = other.scatter_len_[0];
            scatter_len_[1] = other.scatter_len_[1];
            scatter_[0] = other.scatter_[0];
            scatter_[1] = other.scatter_[1];
            len_[0] = other.len_[0];
            len_[1] = other.len_[1];
        }

        void reset(scatter_tensor_matrix&& other)
        {
            data_ = other.data_;
            dense_len_[0] = other.dense_len_[0];
            dense_len_[1] = other.dense_len_[1];
            offset_[0] = other.offset_[0];
            offset_[1] = other.offset_[1];
            leading_len_[0] = other.leading_len_[0];
            leading_len_[1] = other.leading_len_[1];
            leading_stride_[0] = other.leading_stride_[0];
            leading_stride_[1] = other.leading_stride_[1];
            iterator_[0] = std::move(other.iterator_[0]);
            iterator_[1] = std::move(other.iterator_[1]);
            scatter_len_[0] = other.scatter_len_[0];
            scatter_len_[1] = other.scatter_len_[1];
            scatter_[0] = other.scatter_[0];
            scatter_[1] = other.scatter_[1];
            len_[0] = other.len_[0];
            len_[1] = other.len_[1];
        }

        template <typename U, typename V, typename W, typename X>
        void reset(const std::vector<U>& dense_len_m,
                   len_type scatter_len_m,
                   const std::vector<V>& dense_len_n,
                   len_type scatter_len_n,
                   pointer ptr,
                   const std::vector<W>& stride_m,
                   scatter_type scatter_m,
                   const std::vector<X>& stride_n,
                   scatter_type scatter_n)
        {
            TBLIS_ASSERT(dense_len_m.size() == stride_m.size());
            TBLIS_ASSERT(dense_len_n.size() == stride_n.size());

            data_ = ptr;
            dense_len_[0] = leading_len_[0] = (dense_len_m.empty() ? 1 : dense_len_m[0]);
            dense_len_[1] = leading_len_[1] = (dense_len_n.empty() ? 1 : dense_len_n[0]);
            leading_stride_[0] = (stride_m.empty() ? 1 : stride_m[0]);
            leading_stride_[1] = (stride_n.empty() ? 1 : stride_n[0]);
            offset_[0] = 0;
            offset_[1] = 0;

            std::vector<len_type> len_m_, len_n_;
            std::vector<stride_type> stride_m_, stride_n_;
            if (!dense_len_m.empty()) len_m_.assign(dense_len_m.begin()+1, dense_len_m.end());
            if (!dense_len_n.empty()) len_n_.assign(dense_len_n.begin()+1, dense_len_n.end());
            if (!stride_m.empty()) stride_m_.assign(stride_m.begin()+1, stride_m.end());
            if (!stride_n.empty()) stride_n_.assign(stride_n.begin()+1, stride_n.end());

            for (len_type len : len_m_) dense_len_[0] *= len;
            for (len_type len : len_n_) dense_len_[1] *= len;

            iterator_[0] = MArray::viterator<>(len_m_, stride_m_);
            iterator_[1] = MArray::viterator<>(len_n_, stride_n_);

            scatter_len_[0] = scatter_len_m;
            scatter_len_[1] = scatter_len_n;
            scatter_[0] = scatter_m;
            scatter_[1] = scatter_n;

            len_[0] = dense_len_[0]*scatter_len_[0];
            len_[1] = dense_len_[1]*scatter_len_[1];
        }

        void transpose()
        {
            using std::swap;
            swap(dense_len_[0], dense_len_[1]);
            swap(offset_[0], offset_[1]);
            swap(leading_len_[0], leading_len_[1]);
            swap(leading_stride_[0], leading_stride_[1]);
            swap(iterator_[0], iterator_[1]);
            swap(scatter_len_[0], scatter_len_[1]);
            swap(scatter_[0], scatter_[1]);
            swap(len_[0], len_[1]);
        }

        void swap(scatter_tensor_matrix& other)
        {
            using std::swap;
            swap(data_, other.data_);
            swap(dense_len_, other.dense_len_);
            swap(offset_, other.offset_);
            swap(leading_len_, other.leading_len_);
            swap(leading_stride_, other.leading_stride_);
            swap(iterator_, other.iterator_);
            swap(scatter_len_, other.scatter_len_);
            swap(scatter_, other.scatter_);
            swap(len_, other.len_);
        }

        friend void swap(scatter_tensor_matrix& a, scatter_tensor_matrix& b)
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

        len_type scatter_length(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return scatter_len_[dim];
        }

        len_type scatter_length(unsigned dim, len_type m)
        {
            TBLIS_ASSERT(dim < 2);
            std::swap(m, scatter_len_[dim]);
            len_[dim] = dense_len_[dim]*scatter_len_[dim];
            return m;
        }

        stride_type stride(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return leading_stride_[dim];
        }

        scatter_type scatter(unsigned dim) const
        {
            TBLIS_ASSERT(dim < 2);
            return scatter_[dim];
        }

        scatter_type scatter(unsigned dim, scatter_type scatter)
        {
            TBLIS_ASSERT(dim < 2);
            std::swap(scatter, scatter_[dim]);
            return scatter;
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

            len_type dense_m = dense_len_[dim];
            len_type m = len_[dim];
            len_type off_m = offset_[dim];
            len_type m0 = leading_len_[dim];
            stride_type s0 = leading_stride_[dim];
            auto& it = iterator_[dim];

            len_type scat_min = off_m/dense_m;
            len_type scat_max = (off_m+m-1)/dense_m + 1;
            off_m = off_m%dense_m;
            len_type m_max = (off_m+m-1)%dense_m + 1;

            TBLIS_ASSERT(scat_min >= 0 && scat_min <  scatter_len_[dim]);
            TBLIS_ASSERT(scat_max >  0 && scat_max <= scatter_len_[dim]);
            TBLIS_ASSERT(off_m >= 0 && off_m <  dense_len_[dim]);
            TBLIS_ASSERT(m_max >  0 && m_max <= dense_len_[dim]);

            if (scatter_[dim])
            {
                for (len_type scat_m = scat_min, scat_idx = 0;scat_m < scat_max;scat_m++)
                {
                    m = (scat_m == scat_max-1 ? m_max : dense_m) - off_m;
                    len_type p0 = off_m%m0;
                    stride_type off = scatter_[dim][scat_m];
                    it.position(off_m/m0, off);

                    for (len_type idx = 0;it.next(off);)
                    {
                        for (len_type i0 = p0;i0 < m0;i0++)
                        {
                            if (idx == m) return;
                            scatter[scat_idx + idx++] = off + i0*s0;
                        }
                        p0 = 0;
                    }

                    off_m = 0;
                    scat_idx += m;
                }
            }
            else
            {
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
        }

        void fill_block_scatter(unsigned dim, stride_type* scatter, len_type MB, stride_type* block_scatter)
        {
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
