#ifndef _MARRAY_MARRAY_VIEW_HPP_
#define _MARRAY_MARRAY_VIEW_HPP_

#include "marray_base.hpp"

namespace MArray
{

template <typename Type, int NDim>
class marray_view : public marray_base<Type, NDim, marray_view<Type, NDim>, false>
{
    protected:
        typedef marray_base<Type, NDim, marray_view, false> base;

        using base::len_;
        using base::stride_;
        using base::data_;

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

        /**
         * Construct an empty view.
         */
        marray_view() {}

        /**
         * Construct a view of the given tensor, view, or partially-indexed tensor.
         *
         * @param other     The tensor, view, or partially-indexed tensor to view.
         *                  If this is a mutable view (the value type is not
         *                  const-qualified), then `other` may not be a const-
         *                  qualified tensor instance or a view with a const-
         *                  qualified value type. May be either an lvalue- or
         *                  rvalue-reference.
         */
#if MARRAY_DOXYGEN
        marray_view(tensor_or_view_reference other)
#else
        marray_view(const marray_view& other)
#endif
        {
            reset(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename marray_base<U, NDim, D, O>::cptr,pointer>>
        marray_view(const marray_base<U, NDim, D, O>& other)
        {
            reset(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename marray_base<U, NDim, D, O>::pointer,pointer>>
        marray_view(marray_base<U, NDim, D, O>&& other)
        {
            reset(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_convertible_t<
                typename marray_base<U, NDim, D, O>::pointer,pointer>>
        marray_view(marray_base<U, NDim, D, O>& other)
        {
            reset(other);
        }

        template <typename U, int OldNDim, int NIndexed, typename... Dims,
            typename=detail::enable_if_convertible_t<U*,pointer>>
        marray_view(const marray_slice<U, OldNDim, NIndexed, Dims...>& other)
        {
            reset(other);
        }

        /**
         * Construct a view that wraps a raw data pointer, using the provided
         * shape and layout.
         *
         * @param len   The lengths of the tensor dimensions. May be any one-
         *              dimensional container whose elements are convertible to
         *              tensor lengths, including initializer lists.
         *
         * @param ptr   A pointer to the tensor element with all zero inidices.
         *              If this is a mutable view, then the pointer may not be
         *              const-qualified.
         *
         * @param layout    The layout to use, either #ROW_MAJOR or #COLUMN_MAJOR,
         *                  if not specified, the default layout is used.
         */
        marray_view(const detail::array_1d<len_type>& len, pointer ptr,
                    layout layout = DEFAULT)
        {
            reset(len, ptr, layout);
        }

        /**
         * Construct a view that wraps a raw data pointer, using the provided
         * shape and layout.
         *
         * @param len   The lengths of the tensor dimensions. May be any one-
         *              dimensional container whose elements are convertible to
         *              tensor lengths, including initializer lists.
         *
         * @param ptr   A pointer to the tensor element with all zero inidices.
         *              If this is a mutable view, then the pointer may not be
         *              const-qualified.
         *
         * @param stride    The strides along each dimension. The stride is the distance
         *                  in memory (in units of the value type) between successive
         *                  elements along this direction. In general, the strides need
         *                  not be defined such that elements have unique locations,
         *                  although such a view should not be written into. Strides may
         *                  also be negative. In this case, `ptr` still refers to the
         *                  location of the element with all zero indices, although this
         *                  is not the lowest address of any tensor element.
         */
        marray_view(const detail::array_1d<len_type>& len, pointer ptr,
                    const detail::array_1d<stride_type>& stride)
        {
            reset(len, ptr, stride);
        }

        /***********************************************************************
         *
         * Base operations
         *
         **********************************************************************/

        marray_view& operator=(const marray_view& other)
        {
            return base::operator=(other);
        }

        using base::operator=;
        using base::operator+=;
        using base::operator-=;
        using base::operator*=;
        using base::operator/=;
        using base::reset;
        using base::cview;
        using base::view;
        using base::cbegin;
        using base::begin;
        using base::cend;
        using base::end;
        using base::crbegin;
        using base::rbegin;
        using base::crend;
        using base::rend;
        using base::shifted;
        using base::shifted_up;
        using base::shifted_down;
        using base::permuted;
        using base::transposed;
        using base::T;
        using base::lowered;
        using base::cfront;
        using base::front;
        using base::cback;
        using base::back;
        using base::operator[];
        using base::operator();
        using base::cdata;
        using base::data;
        using base::length;
        using base::lengths;
        using base::stride;
        using base::strides;
        using base::dimension;
        using base::size;

        /***********************************************************************
         *
         * Mutating shift
         *
         **********************************************************************/

        /**
         * Shift this view along each dimension by the amount specified.
         *
         * An index `i` in the shifted view is equivalent to an index `i+n[i]`
         * in the original view.
         *
         * @param n The amount by which to shift for each dimension. May be any
         *          one-dimensional container type whose elements are convertible
         *          to a tensor length, including initializer lists.
         */
        void shift(const detail::array_1d<len_type>& n)
        {
            MARRAY_ASSERT(n.size() == NDim);

            std::array<len_type, NDim> n_;
            n.slurp(n_);

            for (auto dim : range(NDim))
                shift(dim, n_[dim]);
        }

        /**
         * Shift this view by the given amount.
         *
         * This overload is only available for vector views.
         * An index `i` in the shifted view is equivalent to an index `i+n`
         * in the original view.
         *
         * @param n The amount by which to shift.
         */
#if !MARRAY_DOXYGEN
        template <typename=void, int N=NDim, typename=detail::enable_if_t<N==1>>
#endif
        void shift(len_type n)
        {
            shift(0, n);
        }

        /**
         * Shift this view by the given amount along one dimension.
         *
         * Only for the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i+n` in the original view.
         *
         * @tparam Dim  The dimension along which to shift the view.
         *
         * @param n     The amount by which to shift.
         */
        template <int Dim>
        void shift(len_type n)
        {
            static_assert(Dim >= 0 && Dim < NDim, "Dim out of range");
            shift(Dim, n);
        }

        /**
         * Shift this view by the given amount along one dimension.
         *
         * Only for the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i+n` in the original view.
         *
         * @param dim   The dimension along which to shift the view.
         *
         * @param n     The amount by which to shift.
         */
        void shift(int dim, len_type n)
        {
            MARRAY_ASSERT(dim >= 0 && dim < NDim);
            data_ += n*stride_[dim];
        }

        /**
         * Shift this view "down".
         *
         * This overload is only available for vector views.
         * An index `i` in the shifted view
         * is equivalent to an index `i+n` in the original view,
         * where `n` is the vector length.
         */
#if !MARRAY_DOXYGEN
        template <typename=void, int N=NDim, typename=detail::enable_if_t<N==1>>
#endif
        void shift_down()
        {
            shift_down(0);
        }

        /**
         * Shift this view "down" along one dimension.
         *
         * Only for the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i+n` in the original view,
         * where `n` is the tensor length in that dimension.
         *
         * @tparam Dim  The dimension along which to shift the view.
         */
        template <int Dim>
        void shift_down()
        {
            shift_down(Dim);
        }

        /**
         * Shift this view "down" along one dimension.
         *
         * Only for the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i+n` in the original view,
         * where `n` is the tensor length in that dimension.
         *
         * @param dim  The dimension along which to shift the view.
         */
        void shift_down(int dim)
        {
            shift(dim, len_[dim]);
        }

        /**
         * Shift this view "up".
         *
         * This overload is only available for vector views.
         * An index `i` in the shifted view
         * is equivalent to an index `i-n` in the original view,
         * where `n` is the vector length.
         */
#if !MARRAY_DOXYGEN
        template <typename=void, int N=NDim, typename=detail::enable_if_t<N==1>>
#endif
        void shift_up()
        {
            shift_up(0);
        }

        /**
         * Shift this view "up" along one dimension.
         *
         * Only for the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i-n` in the original view,
         * where `n` is the tensor length in that dimension.
         *
         * @tparam Dim  The dimension along which to shift the view.
         */
        template <int Dim>
        void shift_up()
        {
            shift_up(Dim);
        }

        /**
         * Shift this view "up" along one dimension.
         *
         * Only for the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i-n` in the original view,
         * where `n` is the tensor length in that dimension.
         *
         * @param dim  The dimension along which to shift the view.
         */
        void shift_up(int dim)
        {
            shift(dim, -len_[dim]);
        }

        /***********************************************************************
         *
         * Mutating permute
         *
         **********************************************************************/

        /**
         * Permuted this view.
         *
         * Indexing into dimension `i` of the permuted view is equivalent to
         * indexing into dimension `perm[i]` of the original view.
         *
         * @param perm  The permutation vector. May be any
         *              one-dimensional container type whose elements are convertible
         *              to `int`, including initializer lists. The values must form
         *              a permutation of `[0,NDim)`, where `NDim` is the number of
         *              tensor dimensions.
         */
        void permute(const detail::array_1d<int>& perm)
        {
            MARRAY_ASSERT(perm.size() == NDim);

            std::array<len_type, NDim> len = len_;
            std::array<stride_type, NDim> stride = stride_;
            std::array<int, NDim> perm_;
            perm.slurp(perm_);

            for (auto i : range(NDim))
            {
                MARRAY_ASSERT(perm_[i] < NDim);
                for (auto j : range(i+1,NDim))
                    MARRAY_ASSERT(perm_[i] != perm_[j]);

                len_[i] = len[perm_[i]];
                stride_[i] = stride[perm_[i]];
            }
        }

        /**
         * Transpose this view.
         *
         * This overload is only available for matrix views.
         */
#if !MARRAY_DOXYGEN
        template <int N=NDim, typename=detail::enable_if_t<N==2>>
#endif
        void transpose()
        {
            permute({1, 0});
        }

        /***********************************************************************
         *
         * Mutating reversal
         *
         **********************************************************************/

        /**
         * Reverse the order of the indices along each dimension.
         *
         * An index of `i` in the reversed view corresponds to an index of
         * `n-1-i` in the original view, where `n` is the tensor length along
         * that dimension.
         */
        void reverse()
        {
            for (auto i : range(NDim)) reverse(i);
        }

        /**
         * Reverse the order of the indices along the specified dimension.
         *
         * Only for the indicated dimension, an index of `i` in the reversed view corresponds to an index of
         * `n-1-i` in the original view, where `n` is the tensor length along
         * that dimension.
         *
         * @tparam Dim  The dimension along which to reverse the indices.
         */
        template <int Dim>
        void reverse()
        {
            reverse(Dim);
        }

        /**
         * Reverse the order of the indices along the specified dimension.
         *
         * Only for the indicated dimension, an index of `i` in the reversed view corresponds to an index of
         * `n-1-i` in the original view, where `n` is the tensor length along
         * that dimension.
         *
         * @param dim  The dimension along which to reverse the indices.
         */
        void reverse(int dim)
        {
            MARRAY_ASSERT(dim >= 0 && dim < NDim);
            data_ += (len_[dim]-1)*stride_[dim];
            stride_[dim] = -stride_[dim];
        }

        /***********************************************************************
         *
         * Basic setters
         *
         **********************************************************************/

        /**
         * Set the data pointer.
         *
         * This is the location of the element with all zero indices.
         */
        pointer data(pointer ptr)
        {
            std::swap(ptr, data_);
            return ptr;
        }

        /**
         * Set the tensor length along the specified dimension.
         *
         * @tparam Dim  The dimension whose length to change.
         *
         * @param len   The new length.
         */
        template <int Dim>
        len_type length(len_type len)
        {
            static_assert(Dim >= 0 && Dim < NDim, "Dim out of range");
            return length(Dim, len);
        }

        /**
         * Set the tensor length along the specified dimension.
         *
         * @param dim   The dimension whose length to change.
         *
         * @param len   The new length.
         */
        len_type length(int dim, len_type len)
        {
            MARRAY_ASSERT(dim >= 0 && dim < NDim);
            std::swap(len, len_[dim]);
            return len;
        }

        /**
         * Set the tensor stride along the specified dimension.
         *
         * @tparam Dim  The dimension whose stride to change.
         *
         * @param stride   The new stride.
         */
        template <int Dim>
        stride_type stride(stride_type s)
        {
            static_assert(Dim >= 0 && Dim < NDim, "Dim out of range");
            return stride(Dim, s);
        }

        /**
         * Set the tensor stride along the specified dimension.
         *
         * @param dim   The dimension whose stride to change.
         *
         * @param stride   The new stride.
         */
        stride_type stride(int dim, stride_type s)
        {
            MARRAY_ASSERT(dim >= 0 && dim < NDim);
            std::swap(s, stride_[dim]);
            return s;
        }

        /***********************************************************************
         *
         * Swap
         *
         **********************************************************************/

        /**
         * Swap this view with another.
         *
         * @param other The view to swap with.
         */
        void swap(marray_view& other)
        {
            base::swap(other);
        }

        /**
         * Swap two views.
         *
         * @param a      The first view to swap.
         *
         * @param b      The second view to swap.
         */
        friend void swap(marray_view& a, marray_view& b)
        {
            a.swap(b);
        }
};

}

#endif
