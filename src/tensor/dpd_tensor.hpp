#ifndef _TBLIS_DPD_TENSOR_HPP_
#define _TBLIS_DPD_TENSOR_HPP_

#include "util/basic_types.h"
#include "util/macros.h"
#include "memory/aligned_allocator.hpp"

namespace tblis
{

#define TBLIS_DPD_DEFAULT_LAYOUT(type) \
    TBLIS_PASTE(TBLIS_PASTE(type,_),MARRAY_DEFAULT_LAYOUT)

enum class dpd_layout
{
    BALANCED_ROW_MAJOR,
    BALANCED_COLUMN_MAJOR,
    BALANCED = TBLIS_DPD_DEFAULT_LAYOUT(BALANCED),
    BLOCKED_ROW_MAJOR,
    BLOCKED_COLUMN_MAJOR,
    BLOCKED = TBLIS_DPD_DEFAULT_LAYOUT(BLOCKED),
    RECURSIVE_ROW_MAJOR,
    RECURSIVE_COLUMN_MAJOR,
    RECURSIVE = TBLIS_DPD_DEFAULT_LAYOUT(RECURSIVE),
    DEFAULT_ROW_MAJOR = BALANCED_ROW_MAJOR,
    DEFAULT_COLUMN_MAJOR = BALANCED_COLUMN_MAJOR,
    DEFAULT = TBLIS_DPD_DEFAULT_LAYOUT(DEFAULT)
};

template <typename T>
class dpd_tensor_view
{
    public:
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;
        typedef uint32_t mask_type;

        class irrep
        {
            protected:
                unsigned value_ = 0;

            public:
                irrep() {}

                irrep(unsigned value)
                : value_(value) {}

                irrep& operator*=(const irrep& other)
                {
                    //                               n
                    // Valid for groups of the form Z , 0 <= n < 32
                    //                               2
                    // i.e. non-denegerate point groups
                    //
                    value_ ^= other.value_;
                    return *this;
                }

                irrep operator*(const irrep& other) const
                {
                    irrep ret(*this);
                    ret *= other;
                    return ret;
                }

                explicit operator bool() const
                {
                    return value_ == 0;
                }

                operator unsigned() const
                {
                    return value_;
                }
        };

    protected:
        // data_[i][j][k]... = pointer to block with irreps i, j, k...
        tensor<pointer> data_;
        // len_[a][i] = length of index a under irrep i
        matrix<len_type> len_;
        // stride_[a][i] = stride of index a when the direct product of
        // the irreps precedingindices (see mask_) is i
        matrix<stride_type> stride_;
        // mask_[a][b:b] (i.e. the bth bit) = 1 iff index b precedes index a
        std::vector<mask_type> mask_;

        dpd_tensor_view& operator=(const dpd_tensor_view& other) = delete;

    public:
        dpd_tensor_view() {}

        dpd_tensor_view(const dpd_tensor_view<T>& other)
        {
            reset(other);
        }

        template <typename U, typename=
            stl_ext::enable_if_t< std::is_same<const U,T>::value &&
                                 !std::is_same<      U,T>::value>>
        dpd_tensor_view(const dpd_tensor_view<U>& other)
        {
            reset(other);
        }

        dpd_tensor_view(dpd_tensor_view<T>&& other)
        {
            reset(std::move(other));
        }

        template <typename U, typename=
            MArray::detail::enable_if_t< std::is_same<const U,T>::value &&
                                        !std::is_same<      U,T>::value>>
        dpd_tensor_view(dpd_tensor_view<U>&& other)
        {
            reset(std::move(other));
        }

        template <typename U, typename=MArray::detail::enable_if_integral_t<U>>
        dpd_tensor_view(const matrix_view<U>& len, pointer ptr,
                        dpd_layout layout=dpd_layout::DEFAULT)
        {
            reset(len, ptr, layout);
        }

        template <typename U, typename V, typename W, typename X, typename=
            MArray::detail::enable_if_t<std::is_integral<U>::value &&
                                        std::is_convertible<V,pointer>::value &&
                                        std::is_integral<W>::value &&
                                        std::is_integral<X>::value>>
        dpd_tensor_view(const matrix_view<U>& len, const tensor_view<V>& ptr,
                        const matrix_view<V>& stride, const matrix_view<X>& mask)
        {
            reset(len, ptr, stride, mask);
        }

        void reset()
        {
            data_.reset();
            len_.reset();
            stride_.reset();
            mask_.clear();
        }

        template <typename U, typename=
            stl_ext::enable_if_t< std::is_same<const U,T>::value &&
                                 !std::is_same<      U,T>::value>>
        void reset(const dpd_tensor_view<U>& other)
        {
            data_.reset(other.data_);
            len_.reset(other.len_);
            stride_.reset(other.stride_);
            mask_ = other.mask_;
        }

        template <typename U, typename=
            stl_ext::enable_if_t< std::is_same<const U,T>::value &&
                                 !std::is_same<      U,T>::value>>
        void reset(dpd_tensor_view<U>&& other)
        {
            data_.reset(std::move(other.data_));
            len_.reset(std::move(other.len_));
            stride_.reset(std::move(other.stride_));
            mask_ = std::move(other.mask_);
        }

        template <typename U, typename=MArray::detail::enable_if_integral_t<U>>
        void reset(const matrix_view<U>& len, pointer ptr,
                   dpd_layout layout=dpd_layout::DEFAULT)
        {
            //TODO
        }

        template <typename U, typename V, typename W, typename X, typename=
            MArray::detail::enable_if_t<std::is_integral<U>::value &&
                                        std::is_convertible<V,pointer>::value &&
                                        std::is_integral<W>::value &&
                                        std::is_integral<X>::value>>
        void reset(const matrix_view<U>& len, const tensor_view<V>& ptr,
                   const matrix_view<V>& stride, const matrix_view<X>& mask)
        {
            const unsigned ndim = len.length(0);
            const unsigned nirrep = len.length(1);

            TBLIS_ASSERT(stride.length(0) == ndim);
            TBLIS_ASSERT(stride.length(1) == nirrep);
            TBLIS_ASSERT(mask.length(0) == ndim);
            TBLIS_ASSERT(mask.length(1) == ndim);

            TBLIS_ASSERT(ptr.dimension() == ndim);
            for (unsigned i = 0;i < ndim;i++)
                TBLIS_ASSERT(ptr.length(i) == nirrep);

            // i.e. nirrep == positive power of two
            TBLIS_ASSERT(nirrep&(nirrep-1) == 0);

            //TODO
        }

        const tensor<pointer>& data() const
        {
            return data_;
        }

        row_view<len_type> length(unsigned dim) const
        {
            TBLIS_ASSERT(dim < dimension());
            return len_[dim];
        }

        const matrix<len_type>& lengths() const
        {
            return len_;
        }

        row_view<stride_type> stride(unsigned dim) const
        {
            TBLIS_ASSERT(dim < dense_dimension());
            return stride_[dim];
        }

        const matrix<stride_type>& strides() const
        {
            return stride_;
        }

        unsigned dimension() const
        {
            return len_.length(0);
        }

        void swap(dpd_tensor_view& other)
        {
            using std::swap;
            swap(data_,   other.data_);
            swap(len_,    other.len_);
            swap(stride_, other.stride_);
            swap(mask_,   other.mask_);
        }

        friend void swap(dpd_tensor_view& a, dpd_tensor_view& b)
        {
            a.swap(b);
        }
};

template <typename T, typename Allocator=aligned_allocator<T,64>>
class dpd_tensor : private Allocator
{
    protected:
        typedef dpd_tensor_view<T> base;

    public:
        typedef T value_type;
        typedef T* pointer;
        typedef const T* const_pointer;
        typedef T& reference;
        typedef const T& const_reference;
        typedef uint32_t mask_type;

        typedef dpd_tensor_view<T>::irrep irrep;

    protected:
        irrep irrep_;
        pointer data_;
        matrix<len_type> len_;          // len_[a][i] = length of index a under irrep i
        matrix<stride_type> stride_;    // stride_[a][i] = stride of index a when the direct product of
                                        // the irreps precedingindices is i
        len_type size_ = 0;
        dpd_layout layout_ = dpd_layout::DEFAULT;

        dpd_tensor& operator=(const dpd_tensor& other) = delete;

        static void default_strides(const matrix<len_type>& len,
                                    dpd_layout layout=dpd_layout::DEFAULT,
                                    matrix<stride_type>& stride,
                                    std::vector<mask_type>& mask)
        {
            auto ndim = len.length(0);
            auto nirrep = len.length(1);

            switch (layout)
            {
                case dpd_layout::BALANCED_ROW_MAJOR:
                {

                }
                break;
                case dpd_layout::BALANCED_COLUMN_MAJOR:
                {

                }
                break;
                case dpd_layout::BLOCKED_ROW_MAJOR:
                {

                }
                break;
                case dpd_layout::BLOCKED_COLUMN_MAJOR:
                {

                }
                break;
                case dpd_layout::RECURSIVE_ROW_MAJOR:
                {

                }
                break;
                case dpd_layout::RECURSIVE_COLUMN_MAJOR:
                {

                }
                break;
            }
        }

    public:
        dpd_tensor() {}

        dpd_tensor(const dpd_tensor_view<T>& other, dpd_layout layout=dpd_layout::DEFAULT)
        {
            reset(other, dpd_layout);
        }

        dpd_tensor(const dpd_tensor_view<T>& other, dpd_layout layout=dpd_layout::DEFAULT)
        {
            reset(other, dpd_layout);
        }

        template <typename OAlloc, typename=
            MArray::detail::enable_if_not_integral_t<OAlloc>>
        dpd_tensor(const dpd_tensor<T, OAlloc>& other, dpd_layout layout=DEFAULT)
        {
            reset(other, dpd_layout);
        }

        dpd_tensor(const dpd_tensor& other)
        {
            reset(other);
        }

        dpd_tensor(dpd_tensor<T>&& other)
        {
            reset(std::move(other));
        }

        dpd_tensor(const std::vector<len_type>& len, const_matrix_view<len_type> batch_idx, const T& value=T(), dpd_layout layout=dpd_layout::DEFAULT)
        {
            reset(len, batch_idx, value, dpd_layout);
        }

        template <typename U, typename=
            MArray::detail::enable_if_integral_t<U>>
        dpd_tensor(const std::vector<U>& len, const_matrix_view<len_type> batch_idx, const T& value=T(), dpd_layout layout=dpd_layout::DEFAULT)
        {
            reset(len, batch_idx, value, dpd_layout);
        }

        dpd_tensor(const std::vector<len_type>& len, const_matrix_view<len_type> batch_idx, uninitialized_t u, dpd_layout layout=dpd_layout::DEFAULT)
        {
            reset(len, batch_idx, uninitialized, dpd_layout);
        }

        template <typename U, typename=
            MArray::detail::enable_if_integral_t<U>>
        dpd_tensor(const std::vector<U>& len, const_matrix_view<len_type> batch_idx, uninitialized_t u, dpd_layout layout=dpd_layout::DEFAULT)
        {
            reset(len, batch_idx, uninitialized, dpd_layout);
        }

        ~dpd_tensor()
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

        void reset(const dpd_tensor_view<T>& other, dpd_layout layout=dpd_layout::DEFAULT)
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

        void reset(const dpd_tensor_view<T>& other, dpd_layout layout=dpd_layout::DEFAULT)
        {
            reset(static_cast<const dpd_tensor_view<T>&>(other), layout);
        }

        template <typename OAlloc>
        MArray::detail::enable_if_not_integral_t<OAlloc>
        reset(const dpd_tensor<T, OAlloc>& other, dpd_layout layout=dpd_layout::DEFAULT)
        {
            reset(static_cast<const dpd_tensor_view<T>&>(other), layout);
        }

        void reset(dpd_tensor&& other)
        {
            swap(other);
        }

        void reset(const std::vector<len_type>& len, const_matrix_view<len_type> batch_idx, const T& val=T(), dpd_layout layout=dpd_layout::DEFAULT)
        {
            reset<len_type>(len, batch_idx, val, dpd_layout);
        }

        template <typename U>
        MArray::detail::enable_if_integral_t<U>
        reset(const std::vector<U>& len, const_matrix_view<len_type> batch_idx, const T& val=T(), dpd_layout layout=dpd_layout::DEFAULT)
        {
            reset(len, batch_idx, uninitialized, dpd_layout);
            std::uninitialized_fill_n(data_ptr_.data(), size_*num_batches(), val);
        }

        void reset(const std::vector<len_type>& len, const_matrix_view<len_type> batch_idx, uninitialized_t u, dpd_layout layout=dpd_layout::DEFAULT)
        {
            reset<len_type>(len, batch_idx, uninitialized, dpd_layout);
        }

        template <typename U>
        MArray::detail::enable_if_integral_t<U>
        reset(const std::vector<U>& len, const_matrix_view<len_type> batch_idx, uninitialized_t u, dpd_layout layout=dpd_layout::DEFAULT)
        {
            batch_idx_alloc_.reset(batch_idx, ROW_MAJOR);
            batch_idx_.reset(batch_idx_alloc_);

            len_ = len;
            stride_ = default_strides({len.begin(), len.end()-batched_dimension()}, dpd_layout);

            size_ = std::accumulate(len.begin(), len.end()-batched_dimension(), size_t(1), std::multiplies<size_t>());
            layout_ = dpd_layout;

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

        void swap(dpd_tensor& other)
        {
            using std::swap;
            base::swap(other);
            swap(data_alloc_,      other.data_alloc_);
            swap(batch_idx_alloc_, other.batch_idx_alloc_);
            swap(size_,            other.size_);
            swap(layout_,          other.layout_);
        }

        friend void swap(dpd_tensor& a, dpd_tensor& b)
        {
            a.swap(b);
        }
};

}

#endif
