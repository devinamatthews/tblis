#ifndef _MARRAY_VARRAY_HPP_
#define _MARRAY_VARRAY_HPP_

#include "varray_view.hpp"

namespace MArray
{

template <typename Type, typename Allocator>
class varray : public varray_base<Type, varray<Type, Allocator>, true>
{
    template <typename, int, typename, bool> friend class marray_base;
    template <typename, int> friend class marray_view;
    template <typename, int, typename> friend class marray;
    template <typename, typename, bool> friend class varray_base;
    template <typename> friend class varray_view;
    template <typename, typename> friend class varray;

    protected:
        typedef varray_base<Type, varray, true> base;
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

        /***********************************************************************
         *
         * Constructors
         *
         **********************************************************************/

        varray() {}

        varray(const varray& other)
        {
            reset(other);
        }

        varray(varray&& other)
        {
            reset(std::move(other));
        }

        template <typename U, typename A,
            typename=detail::enable_if_assignable_t<reference,U>>
        varray(const varray<U, A>& other)
        {
            reset(other);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_assignable_t<reference,U>>
        varray(const varray_base<U, D, O>& other, layout layout = DEFAULT)
        {
            reset(other, layout);
        }

        template <typename Type_=Type>
        explicit varray(detail::len_type_init len)
        {
            reset(len, Type(), DEFAULT);
        }

        explicit varray(detail::array_1d<len_type> len, const Type& val=Type(), layout layout = DEFAULT)
        {
            reset(len, val, layout);
        }

        explicit varray(detail::array_1d<len_type> len, layout layout)
        {
            reset(len, Type(), layout);
        }

        explicit varray(detail::array_1d<len_type> len, uninitialized_t, layout layout = DEFAULT)
        {
            reset(len, uninitialized, layout);
        }

        ~varray()
        {
            reset();
        }

        /***********************************************************************
         *
         * Base operations
         *
         **********************************************************************/

        varray& operator=(const varray& other)
        {
            return base::operator=(other);
        }

        using base::operator=;
        using base::cview;
        using base::view;
        using base::fix;
        using base::shifted;
        using base::shifted_up;
        using base::shifted_down;
        using base::permuted;
        using base::lowered;
        using base::cfront;
        using base::front;
        using base::cback;
        using base::back;
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

        void reset(varray&& other)
        {
            swap(other);
        }

        template <typename U, typename A,
            typename=detail::enable_if_assignable_t<reference,U>>
        void reset(const varray<U, A>& other)
        {
            reset(other, other.layout_);
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_assignable_t<reference,U>>
        void reset(const varray_base<U, D, O>& other, layout layout = DEFAULT)
        {
            if (std::is_scalar<Type>::value)
            {
                reset(other.lengths(), uninitialized, layout);
            }
            else
            {
                reset(other.lengths(), Type(), layout);
            }

            base::template operator=<>(other);
        }

        template <typename Type_=Type>
        void reset(detail::len_type_init len)
        {
            reset(len, Type(), DEFAULT);
        }

        void reset(detail::array_1d<len_type> len, const Type& val=Type(), layout layout = DEFAULT)
        {
            reset(len, uninitialized, layout);
            std::uninitialized_fill_n(data_, storage_.size, val);
        }

        void reset(detail::array_1d<len_type> len, layout layout)
        {
            reset(len, Type(), layout);
        }

        void reset(detail::array_1d<len_type> len, uninitialized_t, layout layout = DEFAULT)
        {
            MARRAY_ASSERT(len.size() > 0);

            reset();

            storage_.size = size(len);
            base::reset(len, alloc_traits::allocate(storage_, storage_.size),
                        strides(len, layout));
        }

        /***********************************************************************
         *
         * Resize
         *
         **********************************************************************/

        void resize(detail::array_1d<len_type> len, const Type& val=Type())
        {
            MARRAY_ASSERT(len.size() == dimension());

            len_vector new_len;
            len.slurp(new_len);

            if (new_len == len_) return;

            varray a(std::move(*this));
            reset(len, val, layout_);
            auto b = view();

            /*
             * It is OK to change the geometry of 'a' even if it is not
             * a view since it is about to go out of scope.
             */
            for (auto i : range(dimension()))
            {
                len_type len = std::min(a.length(i), b.length(i));
                a.len_[i] = len;
                b.length(i, len);
            }

            b = a;
        }

        /***********************************************************************
         *
         * Push/pop
         *
         **********************************************************************/

        void push_back(const Type& x)
        {
            MARRAY_ASSERT(dimension() == 1);
            resize({len_[0]+1});
            back() = x;
        }

        template <typename U, typename D, bool O,
            typename=detail::enable_if_assignable_t<reference,U>>
        void push_back(int dim, const varray_base<U, D, O>& x)
        {
            MARRAY_ASSERT(x.dimension()+1 == dimension());
            MARRAY_ASSERT(dim >= 0 && dim < dimension());

            for (int i = 0, j = 0;i < dimension();i++)
            {
                (void)j;
                MARRAY_ASSERT(i == dim || len_[i] == x.length(j++));
            }

            len_vector len = len_;
            len[dim]++;
            resize(len);
            back(dim) = x;
        }

        void pop_back()
        {
            MARRAY_ASSERT(dimension() == 1);
            MARRAY_ASSERT(len_[0] > 0);
            resize({len_[0]-1});
        }

        void pop_back(int dim)
        {
            MARRAY_ASSERT(dim >= 0 && dim < dimension());
            MARRAY_ASSERT(len_[dim] > 0);

            len_vector len = len_;
            len[dim]--;
            resize(len);
        }

        /***********************************************************************
         *
         * Swap
         *
         **********************************************************************/

        void swap(varray& other)
        {
            using std::swap;
            swap(storage_, other.storage_);
            swap(layout_, other.layout_);
            base::swap(other);
        }

        friend void swap(varray& a, varray& b)
        {
            a.swap(b);
        }
};

}

#endif
