#ifndef _MARRAY_INDEXED_VARRAY_VIEW_HPP_
#define _MARRAY_INDEXED_VARRAY_VIEW_HPP_

#include "indexed_varray_base.hpp"

namespace MArray
{

template <typename Type>
class indexed_varray_view : public indexed_varray_base<Type, indexed_varray_view<Type>, false>
{
    template <typename, typename, bool> friend class indexed_varray_base;
    template <typename> friend class indexed_varray_view;
    template <typename, typename> friend class indexed_varray;

    protected:
        typedef indexed_varray_base<Type, indexed_varray_view, false> base;

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

        indexed_varray_view()
        {
            reset();
        }

        indexed_varray_view(const indexed_varray_view& other)
        {
            reset(other);
        }

        indexed_varray_view(indexed_varray_view&& other)
        {
            reset(std::move(other));
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename indexed_varray_base<U, D, O>::cptr,pointer>>
        indexed_varray_view(const indexed_varray_base<U, D, O>& other)
        {
            reset(other);
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename indexed_varray_base<U, D, O>::pointer,pointer>>
        indexed_varray_view(indexed_varray_base<U, D, O>&& other)
        {
            reset(other);
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename indexed_varray_base<U, D, O>::pointer,pointer>>
        indexed_varray_view(indexed_varray_base<U, D, O>& other)
        {
            reset(other);
        }

        indexed_varray_view(const detail::array_1d<len_type>& len,
                            const detail::array_1d<pointer>& ptr,
                            const detail::array_2d<len_type>& idx,
                            layout layout = DEFAULT)
        {
            reset(len, ptr, idx, layout);
        }

        indexed_varray_view(const detail::array_1d<len_type>& len,
                            const detail::array_1d<pointer>& ptr,
                            const detail::array_2d<len_type>& idx,
                            const detail::array_1d<stride_type>& stride)
        {
            reset(len, ptr, idx, stride);
        }

        /***********************************************************************
         *
         * Base operations
         *
         **********************************************************************/

        indexed_varray_view& operator=(const indexed_varray_view& other)
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
            return const_cast<Type&>(const_cast<const indexed_varray_view&>(*this).factor(idx));
        }

        /***********************************************************************
         *
         * Swap
         *
         **********************************************************************/

        void swap(indexed_varray_view& other)
        {
            base::swap(other);
        }

        friend void swap(indexed_varray_view& a, indexed_varray_view& b)
        {
            a.swap(b);
        }
};

}

#endif
