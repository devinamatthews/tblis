#ifndef _MARRAY_INDEXED_VARRAY_HPP_
#define _MARRAY_INDEXED_VARRAY_HPP_

#include "indexed_varray_view.hpp"

namespace MArray
{

template <typename Type, typename Allocator>
class indexed_varray : public indexed_varray_base<Type, indexed_varray<Type, Allocator>, true>
{
    template <typename, typename, bool> friend class indexed_varray_base;
    template <typename> friend class indexed_varray_view;
    template <typename, typename> friend class indexed_varray;

    protected:
        typedef indexed_varray_base<Type, indexed_varray, true> base;
        typedef std::allocator_traits<Allocator> alloc_traits;

    public:
        using typename base::value_type;
        using typename base::pointer;
        using typename base::const_pointer;
        using typename base::reference;
        using typename base::const_reference;

    protected:
        using base::data_;
        using base::idx_;
        using base::dense_len_;
        using base::idx_len_;
        using base::dense_stride_;
        row<pointer> real_data_;
        matrix<len_type> real_idx_;
        layout layout_ = DEFAULT;
        struct : Allocator { stride_type size = 0; } storage_;

        template <typename U> using initializer_matrix =
            std::initializer_list<std::initializer_list<U>>;

    public:

        /***********************************************************************
         *
         * Constructors
         *
         **********************************************************************/

        indexed_varray()
        {
            reset();
        }

        indexed_varray(const indexed_varray& other)
        {
            reset(other);
        }

        indexed_varray(indexed_varray&& other)
        {
            reset(std::move(other));
        }

        template <typename U, typename A,
            typename=detail::enable_if_assignable_t<reference,U>>
        indexed_varray(const indexed_varray<U, A>& other)
        {
            reset(other);
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_assignable_t<reference,U>>
        indexed_varray(const indexed_varray_base<U, D, O>& other, layout layout = DEFAULT)
        {
            reset(other, layout);
        }

        indexed_varray(std::initializer_list<len_type> len,
                       initializer_matrix<len_type> idx,
                       const Type& value = Type(), layout layout = DEFAULT)
        {
            reset(len, idx, value, layout);
        }

        template <typename U,
            typename=detail::enable_if_container_of_t<U,len_type>>
        indexed_varray(std::initializer_list<len_type> len,
                       std::initializer_list<U> idx,
                       const Type& value = Type(), layout layout = DEFAULT)
        {
            reset(len, idx, value, layout);
        }

        template <typename U, typename V,
            typename=detail::enable_if_t<
                detail::is_container_of<U,len_type>::value &&
                (detail::is_container_of_containers_of<V,len_type>::value ||
                 detail::is_matrix_of<V,len_type>::value)>>
        indexed_varray(const U& len, const V& idx,
                       const Type& value = Type(), layout layout = DEFAULT)
        {
            reset(len, idx, value, layout);
        }

        indexed_varray(std::initializer_list<len_type> len,
                       initializer_matrix<len_type> idx,
                       layout layout)
        {
            reset(len, idx, Type(), layout);
        }

        template <typename U,
            typename=detail::enable_if_container_of_t<U,len_type>>
        indexed_varray(std::initializer_list<len_type> len,
                       std::initializer_list<U> idx,
                       layout layout)
        {
            reset(len, idx, Type(), layout);
        }

        template <typename U, typename V,
            typename=detail::enable_if_t<
                detail::is_container_of<U,len_type>::value &&
                (detail::is_container_of_containers_of<V,len_type>::value ||
                 detail::is_matrix_of<V,len_type>::value)>>
        indexed_varray(const U& len, const V& idx, layout layout)
        {
            reset(len, idx, Type(), layout);
        }

        indexed_varray(std::initializer_list<len_type> len,
                       initializer_matrix<len_type> idx,
                       uninitialized_t, layout layout = DEFAULT)
        {
            reset(len, idx, uninitialized, layout);
        }

        template <typename U,
            typename=detail::enable_if_container_of_t<U,len_type>>
        indexed_varray(std::initializer_list<len_type> len,
                       std::initializer_list<U> idx,
                       uninitialized_t, layout layout = DEFAULT)
        {
            reset(len, idx, uninitialized, layout);
        }

        template <typename U, typename V,
            typename=detail::enable_if_t<
                detail::is_container_of<U,len_type>::value &&
                (detail::is_container_of_containers_of<V,len_type>::value ||
                 detail::is_matrix_of<V,len_type>::value)>>
        indexed_varray(const U& len, const V& idx,
                       uninitialized_t, layout layout = DEFAULT)
        {
            reset(len, idx, uninitialized, layout);
        }

        /***********************************************************************
         *
         * Base operations
         *
         **********************************************************************/

        indexed_varray& operator=(const indexed_varray& other)
        {
            return base::operator=(other);
        }

        using base::operator=;
        using base::cview;
        using base::view;
        using base::operator[];
        using base::cdata;
        using base::data;
        using base::indices;
        using base::dense_length;
        using base::dense_lengths;
        using base::indexed_length;
        using base::indexed_lengths;
        using base::length;
        using base::lengths;
        using base::num_indices;
        using base::dense_stride;
        using base::dense_strides;
        using base::dimension;
        using base::dense_dimension;
        using base::indexed_dimension;

        /***********************************************************************
         *
         * Constructors
         *
         **********************************************************************/

        void reset()
        {
            if (storage_.size > 0)
            {
                pointer data_ = real_data_[0];
                for (stride_type i = storage_.size;i --> 0;)
                {
                    alloc_traits::destroy(storage_, data_+i);
                }
                alloc_traits::deallocate(storage_, data_, storage_.size);
                storage_.size = 0;
            }

            real_data_.reset();
            real_idx_.reset();
            layout_ = DEFAULT;
            base::reset();
        }

        void reset(indexed_varray&& other)
        {
            swap(other);
        }

        template <typename U, typename A,
            typename=detail::enable_if_assignable_t<reference,U>>
        void reset(const indexed_varray<U, A>& other)
        {
            reset(other, other.layout_);
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_assignable_t<reference,U>>
        void reset(const indexed_varray_base<U, D, O>& other, layout layout = DEFAULT)
        {
            if (std::is_scalar<Type>::value)
            {
                reset(other.lengths(), other.idx_, uninitialized, layout);
            }
            else
            {
                reset(other.lengths(), other.idx_, layout);
            }

            base::template operator=<>(other);
        }

        void reset(std::initializer_list<len_type> len,
                   initializer_matrix<len_type> idx,
                   const Type& value = Type(), layout layout = DEFAULT)
        {
            reset(len, idx, uninitialized, layout);
            std::uninitialized_fill_n(real_data_[0], storage_.size, value);
        }

        template <typename U,
            typename=detail::enable_if_container_of_t<U,len_type>>
        void reset(std::initializer_list<len_type> len,
                   std::initializer_list<U> idx,
                   const Type& value = Type(), layout layout = DEFAULT)
        {
            reset(len, idx, uninitialized, layout);
            std::uninitialized_fill_n(real_data_[0], storage_.size, value);
        }

        template <typename U, typename V,
            typename=detail::enable_if_t<
                detail::is_container_of<U,len_type>::value &&
                (detail::is_container_of_containers_of<V,len_type>::value ||
                 detail::is_matrix_of<V,len_type>::value)>>
        void reset(const U& len, const V& idx,
                   const Type& value = Type(), layout layout = DEFAULT)
        {
            reset(len, idx, uninitialized, layout);
            std::uninitialized_fill_n(real_data_[0], storage_.size, value);
        }

        void reset(std::initializer_list<len_type> len,
                   initializer_matrix<len_type> idx,
                   layout layout)
        {
            reset(len, idx, Type(), layout);
        }

        template <typename U,
            typename=detail::enable_if_container_of_t<U,len_type>>
        void reset(std::initializer_list<len_type> len,
                   std::initializer_list<U> idx,
                   layout layout)
        {
            reset(len, idx, Type(), layout);
        }

        template <typename U, typename V,
            typename=detail::enable_if_t<
                detail::is_container_of<U,len_type>::value &&
                (detail::is_container_of_containers_of<V,len_type>::value ||
                 detail::is_matrix_of<V,len_type>::value)>>
        void reset(const U& len, const V& idx, layout layout)
        {
            reset(len, idx, Type(), layout);
        }

        void reset(std::initializer_list<len_type> len,
                   initializer_matrix<len_type> idx,
                   uninitialized_t, layout layout = DEFAULT)
        {
            reset<std::initializer_list<len_type>,
                  initializer_matrix<len_type>,void>(len, idx, uninitialized, layout);
        }

        template <typename U,
            typename=detail::enable_if_container_of_t<U,len_type>>
        void reset(std::initializer_list<len_type> len,
                   std::initializer_list<U> idx,
                   uninitialized_t, layout layout = DEFAULT)
        {
            reset<std::initializer_list<len_type>,
                  std::initializer_list<U>,void>(len, idx, uninitialized, layout);
        }

        template <typename U, typename V,
            typename=detail::enable_if_t<
                detail::is_container_of<U,len_type>::value &&
                (detail::is_container_of_containers_of<V,len_type>::value ||
                 detail::is_matrix_of<V,len_type>::value)>>
        void reset(const U& len, const V& idx,
                   uninitialized_t, layout layout = DEFAULT)
        {
            len_type idx_ndim = detail::length(idx, 0);
            len_type nidx = detail::length(idx, 1);
            MARRAY_ASSERT((idx_ndim > 0 && nidx > 0) ||
                          (idx_ndim == 0 && nidx == 0));

            real_data_.reset({idx_ndim});
            real_idx_.reset({idx_ndim, nidx}, ROW_MAJOR);
            layout_ = layout;

            detail::set_idx(idx, real_idx_);

            base::reset(len, real_data_.cview(), real_idx_.cview(), layout);

            stride_type size = varray_view<Type>::size(dense_len_);
            storage_.size = size*idx_ndim;
            real_data_[0] = alloc_traits::allocate(storage_, storage_.size);
            for (len_type i = 1;i < idx_ndim;i++)
                real_data_[i] = real_data_[i-1] + size;
        }

        /***********************************************************************
         *
         * Swap
         *
         **********************************************************************/

        void swap(indexed_varray& other)
        {
            using std::swap;
            swap(real_data_, other.real_data_);
            swap(real_idx_,  other.real_idx_);
            swap(storage_,   other.storage_);
            swap(layout_,    other.layout_);
            base::swap(other);
        }

        friend void swap(indexed_varray& a, indexed_varray& b)
        {
            a.swap(b);
        }
};

}

#endif
