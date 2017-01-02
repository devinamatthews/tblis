#ifndef _TBLIS_BATCHED_TENSOR_HPP_
#define _TBLIS_BATCHED_TENSOR_HPP_

#include "util/basic_types.h"
#include "memory/aligned_allocator.hpp"

namespace tblis
{

template <typename T>
class const_batched_tensor_view;

template <typename T>
class batched_tensor_view;

template <typename T>
class const_batched_tensor_view
{
    template <typename T_> friend class const_batched_tensor_view;
    template <typename T_> friend class batched_tensor_view;
    template <typename T_, typename Allocator_> friend class batched_tensor;

    public:
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;

    protected:
        const_row_view<const_pointer> data_;
        const_matrix_view<len_type> batch_idx_;
        std::vector<len_type> len_;
        std::vector<stride_type> stride_;

        const_batched_tensor_view& operator=(const const_batched_tensor_view& other) = delete;

    public:
        const_batched_tensor_view() {}

        const_batched_tensor_view(const const_batched_tensor_view<T>& other)
        {
            reset(other);
        }

        const_batched_tensor_view(const batched_tensor_view<T>& other)
        {
            reset(other);
        }

        const_batched_tensor_view(const_batched_tensor_view<T>&& other)
        {
            reset(std::move(other));
        }

        const_batched_tensor_view(batched_tensor_view<T>&& other)
        {
            reset(std::move(other));
        }

        const_batched_tensor_view(const std::vector<len_type>& len, const_row_view<const_pointer> ptr, const_matrix_view<len_type> batch_idx, Layout layout=DEFAULT)
        {
            reset(len, ptr, batch_idx, layout);
        }

        template <typename U, typename=
            MArray::detail::enable_if_integral_t<U>>
        const_batched_tensor_view(const std::vector<U>& len, const_row_view<const_pointer> ptr, const_matrix_view<len_type> batch_idx, Layout layout=DEFAULT)
        {
            reset(len, ptr, batch_idx, layout);
        }

        const_batched_tensor_view(const std::vector<len_type>& len, const_row_view<const_pointer> ptr, const_matrix_view<len_type> batch_idx, const std::vector<stride_type>& stride)
        {
            reset(len, ptr, batch_idx, stride);
        }

        template <typename U, typename V, typename=
            MArray::detail::enable_if_t<std::is_integral<U>::value &&
                                        std::is_integral<V>::value>>
        const_batched_tensor_view(const std::vector<U>& len, const_row_view<const_pointer> ptr, const_matrix_view<len_type> batch_idx, const std::vector<V>& stride)
        {
            reset(len, ptr, batch_idx, stride);
        }

        void reset()
        {
            data_.reset();
            batch_idx_.reset();
            len_.clear();
            stride_.clear();
        }

        void reset(const const_batched_tensor_view<T>& other)
        {
            data_.reset(other.data_);
            batch_idx_.reset(other.batch_idx_);
            len_ = other.len_;
            stride_ = other.stride_;
        }

        void reset(const batched_tensor_view<T>& other)
        {
            reset(static_cast<const const_batched_tensor_view<T>&>(other));
        }

        void reset(const_batched_tensor_view<T>&& other)
        {
            data_.reset(other.data_);
            batch_idx_.reset(other.batch_idx_);
            len_ = std::move(other.len_);
            stride_ = std::move(other.stride_);
        }

        void reset(batched_tensor_view<T>&& other)
        {
            reset(static_cast<const_batched_tensor_view<T>&&>(other));
        }

        void reset(const std::vector<len_type>& len, const_row_view<const_pointer> ptr, const_matrix_view<len_type> batch_idx, Layout layout = DEFAULT)
        {
            reset<len_type>(len, ptr, batch_idx, layout);
        }

        template <typename U>
        MArray::detail::enable_if_integral_t<U>
        reset(const std::vector<U>& len, const_row_view<const_pointer> ptr, const_matrix_view<len_type> batch_idx, Layout layout = DEFAULT)
        {
            reset(len, ptr, batch_idx, tensor<T>::default_strides(len, layout));
        }

        void reset(const std::vector<len_type>& len, const_row_view<const_pointer> ptr, const_matrix_view<len_type> batch_idx, const std::vector<stride_type>& stride)
        {
            reset<len_type, stride_type>(len, ptr, batch_idx, stride);
        }

        template <typename U, typename V>
        MArray::detail::enable_if_t<std::is_integral<U>::value &&
                                    std::is_integral<V>::value>
        reset(const std::vector<U>& len, const_row_view<const_pointer> ptr, const_matrix_view<len_type> batch_idx, const std::vector<V>& stride)
        {
            TBLIS_ASSERT(len.size() == stride.size()+batch_idx.length(1));
            TBLIS_ASSERT(ptr.length() == batch_idx.length(0));
            data_.reset(ptr);
            batch_idx_.reset(batch_idx);
            len_ = len;
            stride_ = stride;
        }

        const_tensor_view<T> operator[](len_type batch) const
        {
            TBLIS_ASSERT(0 <= batch && batch < num_batches());
            return {{len_.begin(), len_.begin()+dense_dimension()},
                    data_[batch], stride_};
        }

        const const_row_view<const_pointer>& batch_data() const
        {
            return data_;
        }

        const_pointer batch_data(len_type batch) const
        {
            TBLIS_ASSERT(0 <= batch && batch < num_batches());
            return data_[batch];
        }

        const const_matrix_view<len_type>& batch_indices() const
        {
            return batch_idx_;
        }

        const_row_view<len_type> batch_indices(len_type batch) const
        {
            TBLIS_ASSERT(0 <= batch && batch < num_batches());
            return batch_idx_[batch];
        }

        len_type length(unsigned dim) const
        {
            TBLIS_ASSERT(dim < dimension());
            return len_[dim];
        }

        const std::vector<len_type>& lengths() const
        {
            return len_;
        }

        len_type num_batches() const
        {
            return batch_idx_.length(0);
        }

        stride_type stride(unsigned dim) const
        {
            TBLIS_ASSERT(dim < dense_dimension());
            return stride_[dim];
        }

        const std::vector<stride_type>& strides() const
        {
            return stride_;
        }

        unsigned dimension() const
        {
            return len_.size();
        }

        unsigned dense_dimension() const
        {
            return stride_.size();
        }

        unsigned batched_dimension() const
        {
            return batch_idx_.length(1);
        }

        void swap(const_batched_tensor_view& other)
        {
            using std::swap;
            swap(data_,      other.data_);
            swap(batch_idx_, other.batch_idx_);
            swap(len_,       other.len_);
            swap(stride_,    other.stride_);
        }

        friend void swap(const_batched_tensor_view& a, const_batched_tensor_view& b)
        {
            a.swap(b);
        }
};

template <typename T>
class batched_tensor_view : public const_batched_tensor_view<T>
{
    template <typename T_> friend class const_batched_tensor_view;
    template <typename T_> friend class batched_tensor_view;
    template <typename T_, typename Allocator_> friend class batched_tensor;

    protected:
        typedef const_batched_tensor_view<T> base;

    public:
        typedef typename base::value_type value_type;
        typedef typename base::pointer pointer;
        typedef typename base::const_pointer const_pointer;
        typedef typename base::reference reference;
        typedef typename base::const_reference const_reference;

    protected:
        using base::data_;
        using base::batch_idx_;
        using base::len_;
        using base::stride_;

        batched_tensor_view& operator=(const batched_tensor_view& other) = delete;

    public:
        batched_tensor_view() {}

        batched_tensor_view(const batched_tensor_view<T>& other)
        {
            reset(other);
        }

        batched_tensor_view(batched_tensor_view<T>&& other)
        {
            reset(std::move(other));
        }

        batched_tensor_view(const std::vector<len_type>& len, const pointer* ptr, const_matrix_view<len_type> batch_idx, Layout layout=DEFAULT)
        {
            reset(len, ptr, batch_idx, layout);
        }

        template <typename U, typename=
            MArray::detail::enable_if_integral_t<U>>
        batched_tensor_view(const std::vector<U>& len, const pointer* ptr, const_matrix_view<len_type> batch_idx, Layout layout=DEFAULT)
        {
            reset(len, ptr, batch_idx, layout);
        }

        batched_tensor_view(const std::vector<len_type>& len, const pointer* ptr, const_matrix_view<len_type> batch_idx, const std::vector<stride_type>& stride)
        {
            reset(len, ptr, batch_idx, stride);
        }

        template <typename U, typename V, typename=
            MArray::detail::enable_if_t<std::is_integral<U>::value &&
                                        std::is_integral<V>::value>>
        batched_tensor_view(const std::vector<U>& len, const pointer* ptr, const_matrix_view<len_type> batch_idx, const std::vector<V>& stride)
        {
            reset(len, ptr, batch_idx, stride);
        }

        void reset()
        {
            base::reset();
        }

        void reset(const batched_tensor_view<T>& other)
        {
            base::reset(other);
        }

        void reset(batched_tensor_view<T>&& other)
        {
            base::reset(std::move(other));
        }

        void reset(const std::vector<len_type>& len, const pointer* ptr, const_matrix_view<len_type> batch_idx, Layout layout = DEFAULT)
        {
            base::reset(len, ptr, batch_idx, layout);
        }

        template <typename U>
        MArray::detail::enable_if_integral_t<U>
        reset(const std::vector<U>& len, const pointer& ptr, const_matrix_view<len_type> batch_idx, Layout layout = DEFAULT)
        {
            base::reset(len, ptr, batch_idx, layout);
        }

        void reset(const std::vector<len_type>& len, const pointer* ptr, const_matrix_view<len_type> batch_idx, const std::vector<stride_type>& stride)
        {
            base::reset(len, ptr, batch_idx, stride);
        }

        template <typename U, typename V>
        MArray::detail::enable_if_t<std::is_integral<U>::value &&
                                    std::is_integral<V>::value>
        reset(const std::vector<U>& len, const pointer* ptr, const_matrix_view<len_type> batch_idx, const std::vector<V>& stride)
        {
            base::reset(len, ptr, batch_idx, stride);
        }

        tensor_view<T> operator[](len_type batch) const
        {
            return {{len_.begin(), len_.begin()+dense_dimension()},
                    const_cast<pointer>(data_[batch]), stride_};
        }

        pointer batch_data(len_type batch) const
        {
            return const_cast<pointer>(base::batch_data(batch));
        }

        using base::batch_indices;
        using base::length;
        using base::lengths;
        using base::num_batches;
        using base::stride;
        using base::strides;
        using base::dimension;
        using base::dense_dimension;
        using base::batched_dimension;

        void swap(batched_tensor_view& other)
        {
            base::swap(other);
        }

        friend void swap(batched_tensor_view& a, batched_tensor_view& b)
        {
            a.swap(b);
        }
};

template <typename T, typename Allocator=aligned_allocator<T,64>>
class batched_tensor : public batched_tensor_view<T>, private Allocator
{
    template <typename T_> friend class const_batched_tensor_view;
    template <typename T_> friend class batched_tensor_view;
    template <typename T_, typename Allocator_> friend class batched_tensor;

    protected:
        typedef batched_tensor_view<T> base;

    public:
        typedef typename base::value_type value_type;
        typedef typename base::pointer pointer;
        typedef typename base::const_pointer const_pointer;
        typedef typename base::reference reference;
        typedef typename base::const_reference const_reference;

    protected:
        using base::data_;
        using base::batch_idx_;
        using base::len_;
        using base::stride_;
        row<T> data_ptr_;
        row<const_pointer> data_alloc_;
        matrix<len_type> batch_idx_alloc_;
        len_type size_ = 0;
        Layout layout_ = DEFAULT;

        batched_tensor& operator=(const batched_tensor& other) = delete;

        std::vector<stride_type> default_strides(const std::vector<len_type>& len, Layout layout=DEFAULT)
        {
            std::vector<stride_type> stride(len.size());

            if (stride.empty()) return stride;

            auto ndim = len.size();
            if (layout == ROW_MAJOR)
            {
                stride[ndim-1] = 1;
                for (auto i = ndim;i --> 1;)
                {
                    stride[i-1] = stride[i]*len[i];
                }
            }
            else
            {
                stride[0] = 1;
                for (unsigned i = 1;i < ndim;i++)
                {
                    stride[i] = stride[i-1]*len[i-1];
                }
            }

            return stride;
        }

    public:
        batched_tensor() {}

        batched_tensor(const const_batched_tensor_view<T>& other, Layout layout=DEFAULT)
        {
            reset(other, layout);
        }

        batched_tensor(const batched_tensor_view<T>& other, Layout layout=DEFAULT)
        {
            reset(other, layout);
        }

        template <typename OAlloc, typename=
            MArray::detail::enable_if_not_integral_t<OAlloc>>
        batched_tensor(const batched_tensor<T, OAlloc>& other, Layout layout=DEFAULT)
        {
            reset(other, layout);
        }

        batched_tensor(const batched_tensor& other)
        {
            reset(other);
        }

        batched_tensor(batched_tensor<T>&& other)
        {
            reset(std::move(other));
        }

        batched_tensor(const std::vector<len_type>& len, const_matrix_view<len_type> batch_idx, const T& value=T(), Layout layout=DEFAULT)
        {
            reset(len, batch_idx, value, layout);
        }

        template <typename U, typename=
            MArray::detail::enable_if_integral_t<U>>
        batched_tensor(const std::vector<U>& len, const_matrix_view<len_type> batch_idx, const T& value=T(), Layout layout=DEFAULT)
        {
            reset(len, batch_idx, value, layout);
        }

        batched_tensor(const std::vector<len_type>& len, const_matrix_view<len_type> batch_idx, uninitialized_t u, Layout layout=DEFAULT)
        {
            reset(len, batch_idx, uninitialized, layout);
        }

        template <typename U, typename=
            MArray::detail::enable_if_integral_t<U>>
        batched_tensor(const std::vector<U>& len, const_matrix_view<len_type> batch_idx, uninitialized_t u, Layout layout=DEFAULT)
        {
            reset(len, batch_idx, uninitialized, layout);
        }

        ~batched_tensor()
        {
            reset();
        }

        void reset()
        {
            data_alloc_.reset();
            batch_idx_alloc_.reset();
            size_ = 0;
            layout_ = DEFAULT;

            base::reset();
        }

        void reset(const const_batched_tensor_view<T>& other, Layout layout=DEFAULT)
        {
            if (std::is_scalar<T>::value)
            {
                reset(other.len_, other.batch_idx_, uninitialized, layout);
            }
            else
            {
                reset(other.len_, other.batch_idx_, T(), layout);
            }

            auto it = MArray::make_iterator(std::vector<len_type>{len_.begin(), len_.begin()+dense_dimension()},
                                            stride_, stride_);
            for (len_type batch = 0;batch < num_batches();batch++)
            {
                auto a = other.batch_data(batch);
                auto b =       batch_data(batch);
                while (it.next(a, b)) *b = *a;
            }
        }

        void reset(const batched_tensor_view<T>& other, Layout layout=DEFAULT)
        {
            reset(static_cast<const const_batched_tensor_view<T>&>(other), layout);
        }

        template <typename OAlloc>
        MArray::detail::enable_if_not_integral_t<OAlloc>
        reset(const batched_tensor<T, OAlloc>& other, Layout layout=DEFAULT)
        {
            reset(static_cast<const const_batched_tensor_view<T>&>(other), layout);
        }

        void reset(batched_tensor&& other)
        {
            swap(other);
        }

        void reset(const std::vector<len_type>& len, const_matrix_view<len_type> batch_idx, const T& val=T(), Layout layout=DEFAULT)
        {
            reset<len_type>(len, batch_idx, val, layout);
        }

        template <typename U>
        MArray::detail::enable_if_integral_t<U>
        reset(const std::vector<U>& len, const_matrix_view<len_type> batch_idx, const T& val=T(), Layout layout=DEFAULT)
        {
            reset(len, batch_idx, uninitialized, layout);
            std::uninitialized_fill_n(data_ptr_.data(), size_*num_batches(), val);
        }

        void reset(const std::vector<len_type>& len, const_matrix_view<len_type> batch_idx, uninitialized_t u, Layout layout=DEFAULT)
        {
            reset<len_type>(len, batch_idx, uninitialized, layout);
        }

        template <typename U>
        MArray::detail::enable_if_integral_t<U>
        reset(const std::vector<U>& len, const_matrix_view<len_type> batch_idx, uninitialized_t u, Layout layout=DEFAULT)
        {
            batch_idx_alloc_.reset(batch_idx, ROW_MAJOR);
            batch_idx_.reset(batch_idx_alloc_);

            len_ = len;
            stride_ = default_strides({len.begin(), len.end()-batched_dimension()}, layout);

            size_ = std::accumulate(len.begin(), len.end()-batched_dimension(), size_t(1), std::multiplies<size_t>());
            layout_ = layout;

            data_ptr_.reset({size_*num_batches()}, uninitialized);
            data_alloc_.reset({num_batches()}, uninitialized);
            data_.reset(data_alloc_);

            for (len_type b = 0;b < num_batches();b++)
            {
                data_alloc_[b] = data_ptr_.data() + b*size_;
            }
        }

        const_tensor_view<T> operator[](len_type batch) const
        {
            return base::operator[](batch);
        }

        tensor_view<T> operator[](len_type batch)
        {
            return base::operator[](batch);
        }

        const_pointer batch_data(len_type batch) const
        {
            return base::batch_data(batch);
        }

        pointer batch_data(len_type batch)
        {
            return base::batch_data(batch);
        }

        using base::batch_indices;
        using base::length;
        using base::lengths;
        using base::num_batches;
        using base::stride;
        using base::strides;
        using base::dimension;
        using base::dense_dimension;
        using base::batched_dimension;

        void swap(batched_tensor& other)
        {
            using std::swap;
            base::swap(other);
            swap(data_alloc_,      other.data_alloc_);
            swap(batch_idx_alloc_, other.batch_idx_alloc_);
            swap(size_,            other.size_);
            swap(layout_,          other.layout_);
        }

        friend void swap(batched_tensor& a, batched_tensor& b)
        {
            a.swap(b);
        }
};

}

#endif
