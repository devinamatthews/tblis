#ifndef _MARRAY_MARRAY_VIEW_HPP_
#define _MARRAY_MARRAY_VIEW_HPP_

#include "marray_base.hpp"

namespace MArray
{

template <typename Type, unsigned NDim>
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

        marray_view() {}

        marray_view(const marray_view& other)
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

        template <typename U, unsigned OldNDim, unsigned NIndexed, typename... Dims,
            typename=detail::enable_if_convertible_t<U*,pointer>>
        marray_view(const marray_slice<U, OldNDim, NIndexed, Dims...>& other)
        {
            reset(other);
        }

        marray_view(const detail::array_1d<len_type>& len, pointer ptr,
                    layout layout = DEFAULT)
        {
            reset(len, ptr, layout);
        }
    
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

        void shift(const detail::array_1d<len_type>& n_)
        {
            MARRAY_ASSERT(n_.size() == NDim);
            
            std::array<len_type, NDim> n;
            n_.slurp(n);
            
            for (unsigned dim = 0;dim < NDim;dim++)
                shift(dim, n[dim]);
        }

        template <typename=void, unsigned N=NDim, typename=detail::enable_if_t<N==1>>
        void shift(len_type n)
        {
            shift(0, n);
        }

        template <unsigned Dim>
        void shift(len_type n)
        {
            static_assert(Dim < NDim, "Dim out of range");
            shift(Dim, n);
        }

        void shift(unsigned dim, len_type n)
        {
            MARRAY_ASSERT(dim < NDim);
            data_ += n*stride_[dim];
        }

        template <typename=void, unsigned N=NDim, typename=detail::enable_if_t<N==1>>
        void shift_down()
        {
            shift_down(0);
        }

        template <unsigned Dim>
        void shift_down()
        {
            shift_down(Dim);
        }

        void shift_down(unsigned dim)
        {
            shift(dim, len_[dim]);
        }

        template <unsigned Dim>
        void shift_up()
        {
            shift_up(Dim);
        }

        template <typename=void, unsigned N=NDim, typename=detail::enable_if_t<N==1>>
        void shift_up()
        {
            shift_up(0);
        }

        void shift_up(unsigned dim)
        {
            shift(dim, -len_[dim]);
        }

        /***********************************************************************
         *
         * Mutating permute
         *
         **********************************************************************/

        void permute(const detail::array_1d<unsigned>& perm_)
        {
            MARRAY_ASSERT(perm_.size() == NDim);
            
            std::array<len_type, NDim> len = len_;
            std::array<stride_type, NDim> stride = stride_;
            std::array<unsigned, NDim> perm;
            perm_.slurp(perm);

            auto it = perm.begin();
            for (unsigned i = 0;i < NDim;i++)
            {
                MARRAY_ASSERT(perm[i] < NDim);
                for (unsigned j = i+1;j < NDim;j++)
                    MARRAY_ASSERT(perm[i] != perm[j]);

                len_[i] = len[perm[i]];
                stride_[i] = stride[perm[i]];
            }
        }

        template <unsigned N=NDim, typename=detail::enable_if_t<N==2>>
        void transpose()
        {
            permute({1, 0});
        }

        /***********************************************************************
         *
         * Mutating reversal
         *
         **********************************************************************/

        void reverse()
        {
            for (unsigned i = 0;i < NDim;i++) reverse(i);
        }

        template <unsigned Dim>
        void reverse()
        {
            reverse(Dim);
        }

        void reverse(unsigned dim)
        {
            MARRAY_ASSERT(dim < NDim);
            data_ += (len_[dim]-1)*stride_[dim];
            stride_[dim] = -stride_[dim];
        }

        /***********************************************************************
         *
         * Basic setters
         *
         **********************************************************************/

        pointer data(pointer ptr)
        {
            std::swap(ptr, data_);
            return ptr;
        }

        template <unsigned Dim>
        len_type length(len_type len)
        {
            static_assert(Dim < NDim, "Dim out of range");
            return length(Dim, len);
        }

        len_type length(unsigned dim, len_type len)
        {
            MARRAY_ASSERT(dim < NDim);
            std::swap(len, len_[dim]);
            return len;
        }

        template <unsigned Dim>
        stride_type stride(stride_type s)
        {
            static_assert(Dim < NDim, "Dim out of range");
            return stride(Dim, s);
        }

        stride_type stride(unsigned dim, stride_type s)
        {
            MARRAY_ASSERT(dim < NDim);
            std::swap(s, stride_[dim]);
            return s;
        }

        /***********************************************************************
         *
         * Swap
         *
         **********************************************************************/

        void swap(marray_view& other)
        {
            base::swap(other);
        }

        friend void swap(marray_view& a, marray_view& b)
        {
            a.swap(b);
        }
};

}

#endif
