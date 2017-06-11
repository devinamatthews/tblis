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

        template <typename U>
        marray_view(std::initializer_list<U> len, pointer ptr, layout layout = DEFAULT)
        {
            reset(len, ptr, layout);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,len_type>>
        marray_view(const U& len, pointer ptr, layout layout = DEFAULT)
        {
            reset(len, ptr, layout);
        }

        template <typename U, typename V>
        marray_view(std::initializer_list<U> len, pointer ptr,
                    std::initializer_list<V> stride)
        {
            reset(len, ptr, stride);
        }

        template <typename U, typename V, typename=
                  detail::enable_if_t<detail::is_container_of<U,len_type>::value &&
                                      detail::is_container_of<V,stride_type>::value>>
        marray_view(const U& len, pointer ptr, const V& stride)
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

        template <typename U>
        void shift(std::initializer_list<U> n)
        {
            shift<std::initializer_list<U>>(n);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,len_type>>
        void shift(const U& n)
        {
            MARRAY_ASSERT(n.size() == NDim);
            auto it = n.begin();
            for (unsigned dim = 0;dim < NDim;dim++)
            {
                shift(dim, *it);
                ++it;
            }
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

        void permute(const std::array<unsigned, NDim>& perm)
        {
            permute<>(perm);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        void permute(const U& perm)
        {
            std::array<len_type, NDim> len = len_;
            std::array<stride_type, NDim> stride = stride_;

            auto it = perm.begin();
            for (unsigned i = 0;i < NDim;i++)
            {
                MARRAY_ASSERT((unsigned)*it < NDim);
                for (auto it2 = perm.begin();it2 != it;++it2)
                    MARRAY_ASSERT(*it != *it2);

                len_[i] = len[*it];
                stride_[i] = stride[*it];
                ++it;
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
        void reversed()
        {
            reverse(Dim);
        }

        void reversed(unsigned dim)
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
