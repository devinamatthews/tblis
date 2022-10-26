#ifndef MARRAY_INDEXED_MARRAY_VIEW_HPP
#define MARRAY_INDEXED_MARRAY_VIEW_HPP

#include "indexed_marray_base.hpp"

namespace MArray
{

template <typename Type>
class indexed_marray_view : public indexed_marray_base<Type, indexed_marray_view<Type>, false>
{
    template <typename, typename, bool> friend class indexed_marray_base;
    template <typename> friend class indexed_marray_view;
    template <typename, typename> friend class indexed_marray;

    protected:
        typedef indexed_marray_base<Type, indexed_marray_view, false> base;

        using base::data_;
        using base::idx_;
        using base::dense_len_;
        using base::idx_len_;
        using base::dense_stride_;
        using base::factor_;

    public:
        using typename base::value_type;
        using typename base::pointer;
        using typename base::const_pointer;
        using typename base::reference;
        using typename base::const_reference;

        /***********************************************************************
         *
         * Constructors
         *
         **********************************************************************/

        indexed_marray_view()
        {
            reset();
        }

        indexed_marray_view(const indexed_marray_view& other)
        {
            reset(other);
        }

        indexed_marray_view(indexed_marray_view&& other)
        {
            reset(std::move(other));
        }

        template <typename U, bool O, typename D>
        indexed_marray_view(const indexed_marray_base<U, D, O>& other)
        {
            reset(other);
        }

        template <typename U, bool O, typename D>
        indexed_marray_view(indexed_marray_base<U, D, O>&& other)
        {
            reset(other);
        }

        template <typename U, bool O, typename D>
        indexed_marray_view(indexed_marray_base<U, D, O>& other)
        {
            reset(other);
        }

        indexed_marray_view(const array_1d<len_type>& len,
                            const array_1d<pointer>& ptr,
                            const array_2d<len_type>& idx,
                            layout layout = DEFAULT_LAYOUT)
        {
            reset(len, ptr, idx, layout);
        }

        indexed_marray_view(const array_1d<len_type>& len,
                            const array_1d<pointer>& ptr,
                            const array_2d<len_type>& idx,
                            const array_1d<stride_type>& stride)
        {
            reset(len, ptr, idx, stride);
        }

        /***********************************************************************
         *
         * Base operations
         *
         **********************************************************************/

        indexed_marray_view& operator=(const indexed_marray_view& other)
        {
            return base::operator=(other);
        }

        using base::operator=;
        using base::reset;
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

        Type& factor(len_type idx)
        {
            return const_cast<Type&>(const_cast<const indexed_marray_view&>(*this).factor(idx));
        }

        /***********************************************************************
         *
         * Swap
         *
         **********************************************************************/

        void swap(indexed_marray_view& other)
        {
            base::swap(other);
        }

        friend void swap(indexed_marray_view& a, indexed_marray_view& b)
        {
            a.swap(b);
        }
};

}

#endif //MARRAY_INDEXED_MARRAY_VIEW_HPP
