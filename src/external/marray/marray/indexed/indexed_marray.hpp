#ifndef MARRAY_INDEXED_MARRAY_HPP
#define MARRAY_INDEXED_MARRAY_HPP

#include "indexed_marray_view.hpp"

namespace MArray
{

template <typename Type, typename Allocator>
class indexed_marray : public indexed_marray_base<Type, indexed_marray<Type, Allocator>, true>
{
    template <typename, typename, bool> friend class indexed_marray_base;
    template <typename> friend class indexed_marray_view;
    template <typename, typename> friend class indexed_marray;

    protected:
        typedef indexed_marray_base<Type, indexed_marray, true> base;
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
        layout layout_ = DEFAULT_LAYOUT;
        struct storage_s : Allocator { stride_type size = 0; } storage_;

    public:

        /***********************************************************************
         *
         * Constructors
         *
         **********************************************************************/

        indexed_marray()
        {
            reset();
        }

        indexed_marray(const indexed_marray& other)
        {
            reset(other);
        }

        indexed_marray(indexed_marray&& other)
        {
            reset(std::move(other));
        }

        template <typename U, typename A>
        indexed_marray(const indexed_marray<U, A>& other)
        {
            reset(other);
        }

        template <typename U, bool O, typename D>
        indexed_marray(const indexed_marray_base<U, D, O>& other, layout layout = DEFAULT_LAYOUT)
        {
            reset(other, layout);
        }

        indexed_marray(const array_1d<len_type>& len,
                       const array_2d<len_type>& idx,
                       const Type& value = Type(), layout layout = DEFAULT_LAYOUT)
        {
            reset(len, idx, value, layout);
        }

        indexed_marray(const array_1d<len_type>& len,
                       const array_2d<len_type>& idx, layout layout)
        {
            reset(len, idx, Type(), layout);
        }

        indexed_marray(const array_1d<len_type>& len,
                       const array_2d<len_type>& idx,
                       uninitialized_t, layout layout = DEFAULT_LAYOUT)
        {
            reset(len, idx, uninitialized, layout);
        }

        ~indexed_marray()
        {
            reset();
        }

        /***********************************************************************
         *
         * Base operations
         *
         **********************************************************************/

        indexed_marray& operator=(const indexed_marray& other)
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

            layout_ = DEFAULT_LAYOUT;
            base::reset();
        }

        void reset(indexed_marray&& other)
        {
            swap(other);
        }

        template <typename U, typename A>
        void reset(const indexed_marray<U, A>& other)
        {
            reset(other, other.layout_);
        }

        template <typename U, bool O, typename D>
        void reset(const indexed_marray_base<U, D, O>& other, layout layout = DEFAULT_LAYOUT)
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

        void reset(const array_1d<len_type>& len,
                   const array_2d<len_type>& idx,
                   const Type& value = Type(), layout layout = DEFAULT_LAYOUT)
        {
            reset(len, idx, uninitialized, layout);
            if (storage_.size > 0)
                std::uninitialized_fill_n(data_[0], storage_.size, value);
        }

        void reset(const array_1d<len_type>& len,
                   const array_2d<len_type>& idx, layout layout)
        {
            reset(len, idx, Type(), layout);
        }

        void reset(const array_1d<len_type>& len,
                   const array_2d<len_type>& idx,
                   uninitialized_t, layout layout = DEFAULT_LAYOUT)
        {
            int total_dim = len.size();
            int idx_dim = idx.length(1);
            int dense_dim = total_dim - idx_dim;
            MARRAY_ASSERT(total_dim > idx_dim);

            int num_idx = idx_dim == 0 ? 1 : idx.length(0);
            MARRAY_ASSERT(num_idx > 0);

            idx.slurp(idx_, ROW_MAJOR);
            len.slurp(dense_len_);
            idx_len_.assign(dense_len_.begin()+dense_dim, dense_len_.end());
            dense_len_.resize(dense_dim);
            dense_stride_ = marray_view<Type>::strides(dense_len_, layout);
            factor_.assign(num_indices(), Type(1));

            layout_ = layout;
            data_.resize(num_idx);
            stride_type size = marray_view<Type>::size(dense_len_);
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

        void swap(indexed_marray& other)
        {
            using std::swap;
            swap(storage_, other.storage_);
            swap(layout_, other.layout_);
            base::swap(other);
        }

        friend void swap(indexed_marray& a, indexed_marray& b)
        {
            a.swap(b);
        }
};

}

#endif //MARRAY_INDEXED_MARRAY_HPP
