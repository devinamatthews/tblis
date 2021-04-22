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
        using base::factor_;
        layout layout_ = DEFAULT;
        struct : Allocator { stride_type size = 0; } storage_;

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

        indexed_varray(const detail::array_1d<len_type>& len,
                       const detail::array_2d<len_type>& idx,
                       const Type& value = Type(), layout layout = DEFAULT)
        {
            reset(len, idx, value, layout);
        }

        indexed_varray(const detail::array_1d<len_type>& len,
                       const detail::array_2d<len_type>& idx, layout layout)
        {
            reset(len, idx, Type(), layout);
        }

        indexed_varray(const detail::array_1d<len_type>& len,
                       const detail::array_2d<len_type>& idx,
                       uninitialized_t, layout layout = DEFAULT)
        {
            reset(len, idx, uninitialized, layout);
        }

        ~indexed_varray()
        {
            reset();
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
        using base::factors;
        using base::factor;
        using base::indices;
        using base::index;
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
                for (stride_type i = storage_.size;i --> 0;)
                {
                    alloc_traits::destroy(storage_, data_[0]+i);
                }
                alloc_traits::deallocate(storage_, data_[0], storage_.size);
                storage_.size = 0;
            }

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

        void reset(const detail::array_1d<len_type>& len,
                   const detail::array_2d<len_type>& idx,
                   const Type& value = Type(), layout layout = DEFAULT)
        {
            reset(len, idx, uninitialized, layout);
            if (storage_.size > 0)
                std::uninitialized_fill_n(data_[0], storage_.size, value);
        }

        void reset(const detail::array_1d<len_type>& len,
                   const detail::array_2d<len_type>& idx, layout layout)
        {
            reset(len, idx, Type(), layout);
        }

        void reset(const detail::array_1d<len_type>& len,
                   const detail::array_2d<len_type>& idx,
                   uninitialized_t, layout layout = DEFAULT)
        {
            unsigned total_dim = len.size();
            unsigned idx_dim = idx.length(1);
            unsigned dense_dim = total_dim - idx_dim;
            MARRAY_ASSERT(total_dim > idx_dim);

            unsigned num_idx = idx_dim == 0 ? 1 : idx.length(0);
            MARRAY_ASSERT(num_idx > 0);

            idx.slurp(idx_, ROW_MAJOR);
            len.slurp(dense_len_);
            idx_len_.assign(dense_len_.begin()+dense_dim, dense_len_.end());
            dense_len_.resize(dense_dim);
            dense_stride_ = varray_view<Type>::strides(dense_len_, layout);
            factor_.assign(num_idx, Type(1));

            layout_ = layout;
            data_.resize(num_idx);
            stride_type size = varray_view<Type>::size(dense_len_);
            storage_.size = size*num_idx;
            if (storage_.size > 0)
            {
                data_[0] = alloc_traits::allocate(storage_, storage_.size);
                for (len_type i = 1;i < num_idx;i++)
                    data_[i] = data_[i-1] + size;
            }
        }

        /***********************************************************************
         *
         * Swap
         *
         **********************************************************************/

        void swap(indexed_varray& other)
        {
            using std::swap;
            swap(storage_, other.storage_);
            swap(layout_, other.layout_);
            base::swap(other);
        }

        friend void swap(indexed_varray& a, indexed_varray& b)
        {
            a.swap(b);
        }
};

}

#endif
