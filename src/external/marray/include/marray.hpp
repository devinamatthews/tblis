#ifndef _MARRAY_MARRAY_HPP_
#define _MARRAY_MARRAY_HPP_

#include "marray_view.hpp"

namespace MArray
{

template <typename Type, unsigned NDim, typename Allocator>
class marray : public marray_base<Type, NDim, marray<Type, NDim, Allocator>, true>
{
    template <typename, unsigned, typename, bool> friend class marray_base;
    template <typename, unsigned> friend class marray_view;
    template <typename, unsigned, typename> friend class marray;
    template <typename, typename, bool> friend class varray_base;
    template <typename> friend class varray_view;
    template <typename, typename> friend class varray;

    protected:
        typedef marray_base<Type, NDim, marray, true> base;
        typedef std::allocator_traits<Allocator> alloc_traits;

        using base::len_;
        using base::stride_;
        using base::data_;
        struct : Allocator { stride_type size = 0; } storage_;
        layout layout_ = DEFAULT;

    public:
        using typename base::value_type;
        using typename base::pointer;
        using typename base::const_pointer;
        using typename base::reference;
        using typename base::const_reference;
        using typename base::initializer_type;

        /***********************************************************************
         *
         * Constructors
         *
         **********************************************************************/

        marray() {}

        marray(const marray& other)
        {
            reset(other);
        }

        marray(marray&& other)
        {
            reset(std::move(other));
        }

        template <typename U, typename A,
            typename=detail::enable_if_assignable_t<reference,U>>
        marray(const marray<U, NDim, A>& other)
        {
            reset(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_assignable_t<reference,U>>
        marray(const marray_base<U, NDim, D, O>& other, layout layout = DEFAULT)
        {
            reset(other, layout);
        }

        template <typename U, unsigned OldNDim, unsigned NIndexed, typename... Dims,
            typename=detail::enable_if_assignable_t<reference,U>>
        marray(const marray_slice<U, OldNDim, NIndexed, Dims...>& other, layout layout = DEFAULT)
        {
            reset(other, layout);
        }

        explicit marray(const detail::array_1d<len_type>& len,
                        const Type& val=Type(), layout layout = DEFAULT)
        {
            reset(len, val, layout);
        }

        marray(const detail::array_1d<len_type>& len, layout layout)
        {
            reset(len, Type(), layout);
        }

        marray(const detail::array_1d<len_type>& len, uninitialized_t,
               layout layout = DEFAULT)
        {
            reset(len, uninitialized, layout);
        }

        marray(initializer_type data, layout layout = DEFAULT)
        {
            reset(data, layout);
        }

        ~marray()
        {
            reset();
        }

        /***********************************************************************
         *
         * Base operations
         *
         **********************************************************************/

        marray& operator=(const marray& other)
        {
            return base::operator=(other);
        }

        using base::operator=;
        using base::operator+=;
        using base::operator-=;
        using base::operator*=;
        using base::operator/=;
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
         * Reset
         *
         **********************************************************************/

        void reset()
        {
            if (data_)
            {
                for (stride_type i = storage_.size;i --> 0;)
                {
                    alloc_traits::destroy(storage_, data_+i);
                }
                alloc_traits::deallocate(storage_, data_, storage_.size);
                storage_.size = 0;
            }

            base::reset();
            layout_ = DEFAULT;
        }

        void reset(marray&& other)
        {
            swap(other);
        }

        template <typename U, typename A,
            typename=detail::enable_if_assignable_t<reference, U>>
        void reset(const marray<U, NDim, A>& other)
        {
            reset(other, other.layout_);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_assignable_t<reference, U>>
        void reset(const marray_base<U, NDim, D, O>& other, layout layout = DEFAULT)
        {
            if (std::is_scalar<Type>::value)
            {
                reset(other.len_, uninitialized, layout);
            }
            else
            {
                reset(other.len_, layout);
            }

            base::template operator=<>(other);
        }

        template <typename U, unsigned OldNDim, unsigned NIndexed, typename... Dims,
            typename=detail::enable_if_assignable_t<reference, U>>
        void reset(const marray_slice<U, OldNDim, NIndexed, Dims...>& other, layout layout = DEFAULT)
        {
            reset(other.view(), layout);
        }

        void reset(const detail::array_1d<len_type>& len,
                   const Type& val=Type(), layout layout = DEFAULT)
        {
            reset(len, uninitialized, layout);
            std::uninitialized_fill_n(data_, storage_.size, val);
        }

        void reset(const detail::array_1d<len_type>& len, layout layout)
        {
            reset(len, Type(), layout);
        }

        void reset(const detail::array_1d<len_type>& len, uninitialized_t,
                   layout layout = DEFAULT)
        {
            reset();

            layout_ = layout;
            storage_.size = size(len);
            base::reset(len, alloc_traits::allocate(storage_, storage_.size),
                        base::strides(len, layout));
        }

        void reset(initializer_type data, layout layout = DEFAULT)
        {
            reset();

            layout_ = layout;
            base::set_lengths(0, len_, data);
            storage_.size = size(len_);
            data_ = alloc_traits::allocate(storage_, storage_.size);
            stride_ = base::strides(len_, layout);
            base::set_data(0, data_, data);
        }

        /***********************************************************************
         *
         * Resize
         *
         **********************************************************************/

        void resize(const detail::array_1d<len_type>& len,
                    const Type& val=Type())
        {
            marray a(std::move(*this));
            reset(len, val, layout_);
            marray_view<Type, NDim> b(*this);

            /*
             * It is OK to change the geometry of 'a' even if it is not
             * a view since it is about to go out of scope.
             */
            for (unsigned i = 0;i < NDim;i++)
            {
                len_type l = std::min(a.length(i), b.length(i));
                a.len_[i] = l;
                b.length(i, l);
            }

            b = a;
        }

        /***********************************************************************
         *
         * Push/pop
         *
         **********************************************************************/

        template <unsigned Dim=0, unsigned N=NDim, typename=detail::enable_if_t<N==1>>
        void push_back(const Type& x)
        {
            static_assert(Dim == 0, "Dim out of range");
            resize({len_[0]+1});
            back() = x;
        }

        template <unsigned N=NDim, typename=detail::enable_if_t<N==1>>
        void push_back(unsigned dim, const Type& x)
        {
            MARRAY_ASSERT(dim == 0);
            push_back(x);
        }

        template <unsigned Dim, typename U, typename D, bool O, unsigned N=NDim,
            typename=detail::enable_if_assignable_t<reference, U>>
        void push_back(const marray_base<U, NDim-1, D, O>& x)
        {
            push_back(Dim, x);
        }

        template <typename U, typename D, bool O, unsigned N=NDim,
            typename=detail::enable_if_assignable_t<reference, U>>
        void push_back(unsigned dim, const marray_base<U, NDim-1, D, O>& x)
        {
            MARRAY_ASSERT(dim < NDim);

            for (unsigned i = 0, j = 0;i < NDim;i++)
            {
                MARRAY_ASSERT(i == dim || len_[i] == x.length(j++));
            }

            std::array<len_type, NDim> len = len_;
            len[dim]++;
            resize(len);
            this->template back<NDim>(dim) = x;
        }

        template <typename=void, unsigned N=NDim>
        typename std::enable_if<N==1>::type
        pop_back()
        {
            resize({len_[0]-1});
        }

        template <unsigned Dim, unsigned N=NDim>
        void pop_back()
        {
            pop_back(Dim);
        }

        void pop_back(unsigned dim)
        {
            MARRAY_ASSERT(dim < NDim);
            MARRAY_ASSERT(len_[dim] > 0);

            std::array<len_type, NDim> len = len_;
            len[dim]--;
            resize(len);
        }

        /***********************************************************************
         *
         * Swap
         *
         **********************************************************************/

        void swap(marray& other)
        {
            using std::swap;
            swap(storage_, other.storage_);
            swap(layout_, other.layout_);
            base::swap(other);
        }

        friend void swap(marray& a, marray& b)
        {
            a.swap(b);
        }
};

}

#endif
