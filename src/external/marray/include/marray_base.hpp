#ifndef _MARRAY_MARRAY_BASE_HPP_
#define _MARRAY_MARRAY_BASE_HPP_

#include "utility.hpp"
#include "range.hpp"

namespace MArray
{

template <typename Type, unsigned NDim, unsigned NIndexed, typename... Dims>
class marray_slice;

template <typename Type, unsigned NDim, typename Derived, bool Owner>
class marray_base;

template <typename Type, unsigned NDim>
class marray_view;

template <typename Type, unsigned NDim, typename Allocator=std::allocator<Type>>
class marray;

template <typename Type, typename Derived, bool Owner>
class varray_base;

template <typename Type>
class varray_view;

template <typename Type, typename Allocator=std::allocator<Type>>
class varray;

template <typename Expr>
struct is_expression_arg_or_scalar;

}

#include "expression.hpp"
#include "marray_slice.hpp"
#include "miterator.hpp"

namespace MArray
{

namespace detail
{

template <typename Type, unsigned NDim>
struct initializer_type;

template <typename Type>
struct initializer_type<Type, 1u>
{
    typedef std::initializer_list<Type> type;
};

template <typename Type, unsigned NDim>
struct initializer_type
{
    typedef std::initializer_list<typename initializer_type<Type, NDim-1>::type> type;
};

template <typename Func, typename Arg, typename... Args>
detail::enable_if_t<sizeof...(Args) && std::is_same<decltype(std::declval<Func&&>()(std::declval<Arg&&>(), std::declval<Args&&>()...)),void>::value>
call(Func&& f, Arg&& arg, Args&&... args)
{
    f(std::forward<Arg>(arg), std::forward<Args>(args)...);
}

template <typename Func, typename Arg, typename... Args>
detail::enable_if_t<std::is_same<decltype(std::declval<Func&&>()(std::declval<Arg&&>())),void>::value>
call(Func&& f, Arg&& arg, Args&&... args)
{
    f(std::forward<Arg>(arg));
}

}

template <typename Type, unsigned NDim, typename Derived, bool Owner>
class marray_base
{
    static_assert(NDim > 0, "NDim must be positive");

    template <typename, unsigned, typename, bool> friend class marray_base;
    template <typename, unsigned> friend class marray_view;
    template <typename, unsigned, typename> friend class marray;
    template <typename, typename, bool> friend class varray_base;
    template <typename> friend class varray_view;
    template <typename, typename> friend class varray;

    public:
        typedef Type value_type;
        typedef Type* pointer;
        typedef const Type* const_pointer;
        typedef Type& reference;
        typedef const Type& const_reference;
        typedef typename detail::initializer_type<Type, NDim>::type
            initializer_type;

        typedef typename std::conditional<Owner,const Type,Type>::type ctype;
        typedef ctype& cref;
        typedef ctype* cptr;

    protected:
        std::array<len_type, NDim> len_ = {};
        std::array<stride_type, NDim> stride_ = {};
        pointer data_ = nullptr;

        /***********************************************************************
         *
         * Reset
         *
         **********************************************************************/

        void reset()
        {
            data_ = nullptr;
            len_.fill(0);
            stride_.fill(0);
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename marray_base<U, NDim, D, O>::cptr,pointer>>
        void reset(const marray_base<U, NDim, D, O>& other)
        {
            reset(const_cast<marray_base<U, NDim, D, O>&>(other));
        }

        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename marray_base<U, NDim, D, O>::pointer,pointer>>
        void reset(marray_base<U, NDim, D, O>& other)
        {
            data_ = other.data_;
            len_ = other.len_;
            stride_ = other.stride_;
        }

        template <typename U, unsigned OldNDim, unsigned NIndexed, typename... Dims,
            typename=detail::enable_if_convertible_t<U*,pointer>>
        void reset(const marray_slice<U, OldNDim, NIndexed, Dims...>& other)
        {
            reset(other.view());
        }

        void reset(std::initializer_list<len_type> len, pointer ptr, layout layout = DEFAULT)
        {
            reset<std::initializer_list<len_type>>(len, ptr, layout);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,len_type>>
        void reset(const U& len, pointer ptr, layout layout = DEFAULT)
        {
            reset(len, ptr, strides(len, layout));
        }

        void reset(std::initializer_list<len_type> len, pointer ptr,
                   std::initializer_list<stride_type> stride)
        {
            reset<std::initializer_list<len_type>,
                  std::initializer_list<stride_type>>(len, ptr, stride);
        }

        template <typename U, typename V, typename=detail::enable_if_t<
            detail::is_container_of<U,len_type>::value &&
            detail::is_container_of<V,stride_type>::value>>
        void reset(const U& len, pointer ptr, const V& stride)
        {
            data_ = ptr;
            std::copy_n(len.begin(), NDim, len_.begin());
            std::copy_n(stride.begin(), NDim, stride_.begin());
        }

        void reset(initializer_type data, layout layout = DEFAULT)
        {
            set_lengths(0, len_, data);
            stride_ = strides(len_, layout);
            set_data(0, data_, data);
        }

        /***********************************************************************
         *
         * Private helper functions
         *
         **********************************************************************/

        template <typename Ptr, typename Func, unsigned... I>
        void for_each_element(Func&& f, detail::integer_sequence<unsigned, I...>) const
        {
            miterator<NDim, 1> it(len_, stride_);
            Ptr ptr = const_cast<Ptr>(data_);
            while (it.next(ptr)) detail::call(std::forward<Func>(f), *ptr, it.position()[I]...);
        }

        void set_lengths(unsigned i, std::array<len_type, NDim>& len, std::initializer_list<Type> data)
        {
            len[i] = data.size();
        }

        template <typename U>
        void set_lengths(unsigned i, std::array<len_type, NDim>& len, std::initializer_list<U> data)
        {
            len[i] = data.size();
            set_lengths(i+1, len, *data.begin());
        }

        void set_data(unsigned i, pointer ptr, std::initializer_list<Type> data)
        {
            auto it = data.begin();
            for (len_type j = 0;j < len_[i];j++)
            {
                ptr[j*stride_[i]] = *it;
                ++it;
            }
        }

        template <typename U>
        void set_data(unsigned i, pointer ptr, std::initializer_list<U> data)
        {
            auto it = data.begin();
            for (len_type j = 0;j < len_[i];j++)
            {
                set_data(i+1, ptr + j*stride_[i], *it);
                ++it;
            }
        }

        void swap(marray_base& other)
        {
            using std::swap;
            swap(data_, other.data_);
            swap(len_, other.len_);
            swap(stride_, other.stride_);
        }

    public:

        /***********************************************************************
         *
         * Static helper functions
         *
         **********************************************************************/

        static std::array<stride_type, NDim>
        strides(std::initializer_list<len_type> len, layout layout = DEFAULT)
        {
            return strides<std::initializer_list<len_type>>(len, layout);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,len_type>>
        static std::array<stride_type, NDim>
        strides(const U& len, layout layout = DEFAULT)
        {
            //TODO: add alignment option

            std::array<stride_type, NDim> stride;

            if (layout == ROW_MAJOR)
            {
                /*
                 * Some monkeying around has to be done to support len as a
                 * std::initializer_list (or other forward iterator range)
                 */
                auto it = len.begin(); ++it;
                std::copy_n(it, NDim-1, stride.begin());
                stride[NDim-1] = 1;
                for (unsigned i = NDim;i --> 1;)
                {
                    stride[i-1] *= stride[i];
                }
            }
            else
            {
                stride[0] = 1;
                auto it = len.begin();
                for (unsigned i = 1;i < NDim;i++)
                {
                    stride[i] = stride[i-1]*(*it);
                    ++it;
                }
            }

            return stride;
        }

        static stride_type size(std::initializer_list<len_type> len)
        {
            return size<std::initializer_list<len_type>>(len);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,len_type>>
        static stride_type size(const U& len)
        {
            //TODO: add alignment option

            stride_type s = 1;
            for (auto& l : len) s *= l;
            return s;
        }

        /***********************************************************************
         *
         * Operators
         *
         **********************************************************************/

        Derived& operator=(const marray_base& other)
        {
            return operator=<>(other);
        }

        Derived& operator=(initializer_type data)
        {
            std::array<len_type, NDim> len;
            set_lengths(0, len, data);
            MARRAY_ASSERT(len == len_);
            set_data(0, data_, data);
            return static_cast<Derived&>(*this);
        }

        template <typename Expression,
            typename=detail::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
        Derived& operator=(const Expression& other)
        {
            assign_expr(*this, other);
            return static_cast<Derived&>(*this);
        }

        template <typename Expression, bool O=Owner,
            typename=detail::enable_if_t<!O && is_expression_arg_or_scalar<Expression>::value>>
        const Derived& operator=(const Expression& other) const
        {
            assign_expr(*this, other);
            return static_cast<const Derived&>(*this);
        }

        template <typename Expression,
            typename=detail::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
        Derived& operator+=(const Expression& other)
        {
            *this = *this + other;
            return static_cast<Derived&>(*this);
        }

        template <typename Expression, bool O=Owner,
            typename=detail::enable_if_t<!O && is_expression_arg_or_scalar<Expression>::value>>
        const Derived& operator+=(const Expression& other) const
        {
            *this = *this + other;
            return static_cast<const Derived&>(*this);
        }

        template <typename Expression,
            typename=detail::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
        Derived& operator-=(const Expression& other)
        {
            *this = *this - other;
            return static_cast<Derived&>(*this);
        }

        template <typename Expression, bool O=Owner,
            typename=detail::enable_if_t<!O && is_expression_arg_or_scalar<Expression>::value>>
        const Derived& operator-=(const Expression& other) const
        {
            *this = *this - other;
            return static_cast<const Derived&>(*this);
        }

        template <typename Expression,
            typename=detail::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
        Derived& operator*=(const Expression& other)
        {
            *this = *this * other;
            return static_cast<Derived&>(*this);
        }

        template <typename Expression, bool O=Owner,
            typename=detail::enable_if_t<!O && is_expression_arg_or_scalar<Expression>::value>>
        const Derived& operator*=(const Expression& other) const
        {
            *this = *this * other;
            return static_cast<const Derived&>(*this);
        }

        template <typename Expression,
            typename=detail::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
        Derived& operator/=(const Expression& other)
        {
            *this = *this / other;
            return static_cast<Derived&>(*this);
        }

        template <typename Expression, bool O=Owner,
            typename=detail::enable_if_t<!O && is_expression_arg_or_scalar<Expression>::value>>
        const Derived& operator/=(const Expression& other) const
        {
            *this = *this / other;
            return static_cast<const Derived&>(*this);
        }

        template <typename U, unsigned N, typename D, bool O>
        detail::enable_if_t<N==NDim, bool>
        operator==(const marray_base<U, N, D, O>& other) const
        {
            if (len_ != other.len_) return false;

            miterator<NDim, 2> it(len_, stride_, other.stride_);

            auto a = data_;
            auto b = other.data_;
            while (it.next(a, b))
            {
                if (*a != *b) return false;
            }

            return true;
        }

        template <typename U, unsigned N, typename D, bool O>
        detail::enable_if_t<N!=NDim, bool>
        operator==(const marray_base<U, N, D, O>&) const
        {
            return false;
        }

        template <typename U, unsigned N, typename D, bool O>
        bool operator!=(const marray_base<U, N, D, O>& other) const
        {
            return !(*this == other);
        }

        /***********************************************************************
         *
         * Views
         *
         **********************************************************************/

        marray_view<const Type, NDim> cview() const
        {
            return const_cast<marray_base&>(*this).view();
        }

        marray_view<ctype, NDim> view() const
        {
            return const_cast<marray_base&>(*this).view();
        }

        marray_view<Type, NDim> view()
        {
            return *this;
        }

        friend marray_view<const Type, NDim> cview(const marray_base& x)
        {
            return x.view();
        }

        friend marray_view<ctype, NDim> view(const marray_base& x)
        {
            return x.view();
        }

        friend marray_view<Type, NDim> view(marray_base& x)
        {
            return x.view();
        }

        /***********************************************************************
         *
         * Shifting
         *
         **********************************************************************/

        marray_view<ctype, NDim> shifted(std::initializer_list<len_type> n) const
        {
            return const_cast<marray_base&>(*this).shifted(n);
        }

        marray_view<Type, NDim> shifted(std::initializer_list<len_type> n)
        {
            return shifted<std::initializer_list<len_type>>(n);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,len_type>>
        marray_view<ctype, NDim> shifted(const U& n) const
        {
            return const_cast<marray_base&>(*this).shifted(n);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,len_type>>
        marray_view<Type, NDim> shifted(const U& n)
        {
            marray_view<Type,NDim> r(*this);
            r.shift(n);
            return r;
        }

        template <typename=void, unsigned N=NDim, typename=detail::enable_if_t<N==1>>
        marray_view<ctype,1> shifted(len_type n) const
        {
            return const_cast<marray_base&>(*this).shifted(n);
        }

        template <typename=void, unsigned N=NDim, typename=detail::enable_if_t<N==1>>
        marray_view<Type,1> shifted(len_type n)
        {
            return shifted(0, n);
        }

        template <unsigned Dim>
        marray_view<ctype, NDim> shifted(len_type n) const
        {
            return const_cast<marray_base&>(*this).shifted<Dim>(n);
        }

        template <unsigned Dim>
        marray_view<Type, NDim> shifted(len_type n)
        {
            return shifted(Dim, n);
        }

        marray_view<ctype, NDim> shifted(unsigned dim, len_type n) const
        {
            return const_cast<marray_base&>(*this).shifted(dim, n);
        }

        marray_view<Type, NDim> shifted(unsigned dim, len_type n)
        {
            marray_view<Type,NDim> r(*this);
            r.shift(dim, n);
            return r;
        }

        template <typename=void, unsigned N=NDim, typename=detail::enable_if_t<N==1>>
        marray_view<ctype,1> shifted_down() const
        {
            return const_cast<marray_base&>(*this).shifted_down();
        }

        template <typename=void, unsigned N=NDim, typename=detail::enable_if_t<N==1>>
        marray_view<Type,1> shifted_down()
        {
            return shifted_down(0);
        }

        template <unsigned Dim>
        marray_view<ctype,NDim> shifted_down() const
        {
            return const_cast<marray_base&>(*this).shifted_down<Dim>();
        }

        template <unsigned Dim>
        marray_view<Type,NDim> shifted_down()
        {
            return shifted_down(Dim);
        }

        marray_view<ctype,NDim> shifted_down(unsigned dim) const
        {
            return const_cast<marray_base&>(*this).shifted_down(dim);
        }

        marray_view<Type,NDim> shifted_down(unsigned dim)
        {
            return shifted(dim, len_[dim]);
        }

        template <typename=void, unsigned N=NDim, typename=detail::enable_if_t<N==1>>
        marray_view<ctype,1> shifted_up() const
        {
            return const_cast<marray_base&>(*this).shifted_up();
        }

        template <typename=void, unsigned N=NDim, typename=detail::enable_if_t<N==1>>
        marray_view<Type,1> shifted_up()
        {
            return shifted_up(0);
        }

        template <unsigned Dim>
        marray_view<ctype,NDim> shifted_up() const
        {
            return const_cast<marray_base&>(*this).shifted_up<Dim>();
        }

        template <unsigned Dim>
        marray_view<Type,NDim> shifted_up()
        {
            return shifted_up(Dim);
        }

        marray_view<ctype,NDim> shifted_up(unsigned dim) const
        {
            return const_cast<marray_base&>(*this).shifted_up(dim);
        }

        marray_view<Type,NDim> shifted_up(unsigned dim)
        {
            return shifted(dim, -len_[dim]);
        }

        /***********************************************************************
         *
         * Permutation
         *
         **********************************************************************/

        marray_view<ctype,NDim> permuted(std::initializer_list<unsigned> perm) const
        {
            return const_cast<marray_base&>(*this).permuted(perm);
        }

        marray_view<Type,NDim> permuted(std::initializer_list<unsigned> perm)
        {
            return permuted<std::initializer_list<unsigned>>(perm);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        marray_view<ctype,NDim> permuted(const U& perm) const
        {
            return const_cast<marray_base&>(*this).permuted<U>(perm);
        }

        template <typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        marray_view<Type,NDim> permuted(const U& perm)
        {
            marray_view<Type,NDim> r(*this);
            r.permute(perm);
            return r;
        }

        template <unsigned N=NDim, typename=detail::enable_if_t<N==2>>
        marray_view<ctype, NDim> transposed() const
        {
            return const_cast<marray_base&>(*this).transposed();
        }

        template <unsigned N=NDim, typename=detail::enable_if_t<N==2>>
        marray_view<Type, NDim> transposed()
        {
            return permuted({1, 0});
        }

        template <unsigned N=NDim, typename=detail::enable_if_t<N==2>>
        marray_view<ctype, NDim> T() const
        {
            return const_cast<marray_base&>(*this).T();
        }

        template <unsigned N=NDim, typename=detail::enable_if_t<N==2>>
        marray_view<Type, NDim> T()
        {
            return transposed();
        }

        /***********************************************************************
         *
         * Dimension change
         *
         **********************************************************************/

        template <unsigned NewNDim>
        marray_view<ctype, NewNDim> lowered(std::initializer_list<unsigned> split) const
        {
            return const_cast<marray_base&>(*this).lowered<NewNDim>(split);
        }

        template <unsigned NewNDim>
        marray_view<Type, NewNDim> lowered(std::initializer_list<unsigned> split)
        {
            return lowered<NewNDim,std::initializer_list<unsigned>>(split);
        }

        template <unsigned NewNDim, typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        marray_view<ctype, NewNDim> lowered(const U& split) const
        {
            return const_cast<marray_base&>(*this).lowered<NewNDim,U>(split);
        }

        template <unsigned NewNDim, typename U, typename=detail::enable_if_container_of_t<U,unsigned>>
        marray_view<Type, NewNDim> lowered(const U& split_)
        {
            static_assert(NewNDim > 0 && NewNDim <= NDim,
                          "Cannot split into this number of dimensions");

            constexpr unsigned NSplit = NewNDim-1;
            MARRAY_ASSERT(split_.size() == NSplit);

            std::array<unsigned, NSplit> split;
            std::copy_n(split_.begin(), NSplit, split.begin());

            for (unsigned i = 0;i < NSplit;i++)
            {
                MARRAY_ASSERT(split[i] <= NDim);
                if (i != 0) MARRAY_ASSERT(split[i-1] <= split[i]);
            }

            std::array<len_type, NSplit+1> newlen;
            std::array<stride_type, NSplit+1> newstride;

            for (unsigned i = 0;i <= NSplit;i++)
            {
                int begin = (i == 0 ? 0 : split[i-1]);
                int end = (i == NSplit ? NDim-1 : split[i]-1);
                if (begin > end) continue;

                if (stride_[begin] < stride_[end])
                {
                    newlen[i] = len_[end];
                    newstride[i] = stride_[begin];
                    for (int j = begin;j < end;j++)
                    {
                        MARRAY_ASSERT(stride_[j+1] == stride_[j]*len_[j]);
                        newlen[i] *= len_[j];
                    }
                }
                else
                {
                    newlen[i] = len_[end];
                    newstride[i] = stride_[end];
                    for (int j = begin;j < end;j++)
                    {
                        MARRAY_ASSERT(stride_[j] == stride_[j+1]*len_[j+1]);
                        newlen[i] *= len_[j];
                    }
                }
            }

            return {newlen, data_, newstride};
        }

        /***********************************************************************
         *
         * Reversal
         *
         **********************************************************************/

        marray_view<ctype, NDim> reversed() const
        {
            return const_cast<marray_base&>(*this).reversed();
        }

        marray_view<Type, NDim> reversed()
        {
            marray_view<Type,NDim> r(*this);
            r.reverse();
            return r;
        }

        template <unsigned Dim>
        marray_view<ctype, NDim> reversed() const
        {
            return const_cast<marray_base&>(*this).reversed<Dim>();
        }

        template <unsigned Dim>
        marray_view<Type, NDim> reversed()
        {
            return reversed(Dim);
        }

        marray_view<ctype, NDim> reversed(unsigned dim) const
        {
            return const_cast<marray_base&>(*this).reversed(dim);
        }

        marray_view<Type, NDim> reversed(unsigned dim)
        {
            marray_view<Type,NDim> r(*this);
            r.reverse(dim);
            return r;
        }

        /***********************************************************************
         *
         * Slices
         *
         **********************************************************************/

        template <unsigned Dim=0, unsigned N=NDim>
        detail::enable_if_t<N==1, const_reference>
        cfront() const
        {
            return const_cast<marray_base&>(*this).front<Dim>();
        }

        template <unsigned Dim=0, unsigned N=NDim>
        detail::enable_if_t<N==1, cref>
        front() const
        {
            return const_cast<marray_base&>(*this).front<Dim>();
        }

        template <unsigned Dim=0, unsigned N=NDim>
        detail::enable_if_t<N==1, reference>
        front()
        {
            static_assert(Dim == 0, "Dim out of range");
            MARRAY_ASSERT(len_[0] > 0);
            return data_[0];
        }

        template <unsigned N=NDim>
        detail::enable_if_t<N==1, const_reference>
        cfront(unsigned dim) const
        {
            return const_cast<marray_base&>(*this).front(dim);
        }

        template <unsigned N=NDim>
        detail::enable_if_t<N==1, cref>
        front(unsigned dim) const
        {
            return const_cast<marray_base&>(*this).front(dim);
        }

        template <unsigned N=NDim>
        detail::enable_if_t<N==1, reference>
        front(unsigned dim)
        {
            MARRAY_ASSERT(dim == 0);
            return front();
        }

        template <unsigned Dim, unsigned N=NDim>
        detail::enable_if_t<N!=1, marray_view<const Type, NDim-1>>
        cfront() const
        {
            return const_cast<marray_base&>(*this).front<Dim>();
        }

        template <unsigned Dim, unsigned N=NDim>
        detail::enable_if_t<N!=1, marray_view<ctype, NDim-1>>
        front() const
        {
            return const_cast<marray_base&>(*this).front<Dim>();
        }

        template <unsigned Dim, unsigned N=NDim>
        detail::enable_if_t<N!=1, marray_view<Type, NDim-1>>
        front()
        {
            return front<N>(Dim);
        }

        template <unsigned N=NDim>
        detail::enable_if_t<N!=1, marray_view<const Type, NDim-1>>
        cfront(unsigned dim) const
        {
            return const_cast<marray_base&>(*this).front(dim);
        }

        template <unsigned N=NDim>
        detail::enable_if_t<N!=1, marray_view<ctype, NDim-1>>
        front(unsigned dim) const
        {
            return const_cast<marray_base&>(*this).front(dim);
        }

        template <unsigned N=NDim>
        detail::enable_if_t<N!=1, marray_view<Type, NDim-1>>
        front(unsigned dim)
        {
            MARRAY_ASSERT(dim < NDim);
            MARRAY_ASSERT(len_[dim] > 0);

            std::array<len_type, NDim-1> len;
            std::array<stride_type, NDim-1> stride;

            std::copy_n(len_.begin(), dim, len.begin());
            std::copy_n(len_.begin()+dim+1, NDim-dim-1, len.begin()+dim);
            std::copy_n(stride_.begin(), dim, stride.begin());
            std::copy_n(stride_.begin()+dim+1, NDim-dim-1, stride.begin()+dim);

            return {len, data_, stride};
        }

        template <unsigned Dim=0, unsigned N=NDim>
        detail::enable_if_t<N==1, const_reference>
        cback() const
        {
            return const_cast<marray_base&>(*this).back<Dim>();
        }

        template <unsigned Dim=0, unsigned N=NDim>
        detail::enable_if_t<N==1, cref>
        back() const
        {
            return const_cast<marray_base&>(*this).back<Dim>();
        }

        template <unsigned Dim=0, unsigned N=NDim>
        detail::enable_if_t<N==1, reference>
        back()
        {
            static_assert(Dim == 0, "Dim out of range");
            MARRAY_ASSERT(len_[0] > 0);
            return data_[(len_[0]-1)*stride_[0]];
        }

        template <unsigned N=NDim>
        detail::enable_if_t<N==1, const_reference>
        cback(unsigned dim) const
        {
            return const_cast<marray_base&>(*this).back(dim);
        }

        template <unsigned N=NDim>
        detail::enable_if_t<N==1, cref>
        back(unsigned dim) const
        {
            return const_cast<marray_base&>(*this).back(dim);
        }

        template <unsigned N=NDim>
        detail::enable_if_t<N==1, reference>
        back(unsigned dim)
        {
            MARRAY_ASSERT(dim == 0);
            return back();
        }

        template <unsigned Dim, unsigned N=NDim>
        detail::enable_if_t<N!=1, marray_view<const Type, NDim-1>>
        cback() const
        {
            return const_cast<marray_base&>(*this).back<Dim>();
        }

        template <unsigned Dim, unsigned N=NDim>
        detail::enable_if_t<N!=1, marray_view<ctype, NDim-1>>
        back() const
        {
            return const_cast<marray_base&>(*this).back<Dim>();
        }

        template <unsigned Dim, unsigned N=NDim>
        detail::enable_if_t<N!=1, marray_view<Type, NDim-1>>
        back()
        {
            return back<N>(Dim);
        }

        template <unsigned N=NDim>
        detail::enable_if_t<N!=1, marray_view<const Type, NDim-1>>
        cback(unsigned dim) const
        {
            return const_cast<marray_base&>(*this).back(dim);
        }

        template <unsigned N=NDim>
        detail::enable_if_t<N!=1, marray_view<ctype, NDim-1>>
        back(unsigned dim) const
        {
            return const_cast<marray_base&>(*this).back(dim);
        }

        template <unsigned N=NDim>
        detail::enable_if_t<N!=1, marray_view<Type, NDim-1>>
        back(unsigned dim)
        {
            MARRAY_ASSERT(dim < NDim);
            MARRAY_ASSERT(len_[dim] > 0);

            std::array<len_type, NDim-1> len;
            std::array<stride_type, NDim-1> stride;

            std::copy_n(len_.begin(), dim, len.begin());
            std::copy_n(len_.begin()+dim+1, NDim-dim-1, len.begin()+dim);
            std::copy_n(stride_.begin(), dim, stride.begin());
            std::copy_n(stride_.begin()+dim+1, NDim-dim-1, stride.begin()+dim);

            return {len, data_ + (len_[dim]-1)*stride_[dim], stride};
        }

        /***********************************************************************
         *
         * Indexing
         *
         **********************************************************************/

        template <unsigned N=NDim>
        detail::enable_if_t<N==1, cref>
        operator[](len_type i) const
        {
            return const_cast<marray_base&>(*this)[i];
        }

        template <unsigned N=NDim>
        detail::enable_if_t<N==1, reference>
        operator[](len_type i)
        {
            MARRAY_ASSERT(i < len_[0]);
            return data_[i*stride_[0]];
        }

        template <unsigned N=NDim>
        detail::enable_if_t<N!=1, marray_slice<ctype, NDim, 1>>
        operator[](len_type i) const
        {
            MARRAY_ASSERT(i < len_[0]);
            return {*this, i};
        }

        template <unsigned N=NDim>
        detail::enable_if_t<N!=1, marray_slice<Type, NDim, 1>>
        operator[](len_type i)
        {
            MARRAY_ASSERT(i < len_[0]);
            return {*this, i};
        }

        template <typename I>
        marray_slice<ctype, NDim, 1, slice_dim>
        operator[](const range_t<I>& x) const
        {
            MARRAY_ASSERT(x.size() >= 0);
            MARRAY_ASSERT(x.front() >= 0);
            MARRAY_ASSERT(x.front()+x.size() <= len_[0]);
            return {*this, x};
        }

        template <typename I>
        marray_slice<Type, NDim, 1, slice_dim>
        operator[](const range_t<I>& x)
        {
            MARRAY_ASSERT(x.size() >= 0);
            MARRAY_ASSERT(x.front() >= 0);
            MARRAY_ASSERT(x.front()+x.size() <= len_[0]);
            return {*this, x};
        }

        marray_slice<ctype, NDim, 1, slice_dim>
        operator[](all_t) const
        {
            return {*this, range(len_[0])};
        }

        marray_slice<Type, NDim, 1, slice_dim>
        operator[](all_t)
        {
            return {*this, range(len_[0])};
        }

        marray_slice<ctype, NDim, 0, bcast_dim>
        operator[](bcast_t) const
        {
            return {*this, slice::bcast, len_[0]};
        }

        marray_slice<Type, NDim, 0, bcast_dim>
        operator[](bcast_t)
        {
            return {*this, slice::bcast, len_[0]};
        }

        template <typename Arg, typename=
            detail::enable_if_t<detail::is_index_or_slice<Arg>::value>>
        auto operator()(Arg&& arg) const ->
        decltype((*this)[std::forward<Arg>(arg)])
        {
            return (*this)[std::forward<Arg>(arg)];
        }

        template <typename Arg, typename=
            detail::enable_if_t<detail::is_index_or_slice<Arg>::value>>
        auto operator()(Arg&& arg) ->
        decltype((*this)[std::forward<Arg>(arg)])
        {
            return (*this)[std::forward<Arg>(arg)];
        }

        template <typename Arg, typename... Args, typename=
            detail::enable_if_t<sizeof...(Args) &&
                detail::are_indices_or_slices<Arg, Args...>::value>>
        auto operator()(Arg&& arg, Args&&... args) const ->
        decltype((*this)[std::forward<Arg>(arg)](std::forward<Args>(args)...))
        {
            return (*this)[std::forward<Arg>(arg)](std::forward<Args>(args)...);
        }

        template <typename Arg, typename... Args, typename=
            detail::enable_if_t<sizeof...(Args) &&
                detail::are_indices_or_slices<Arg, Args...>::value>>
        auto operator()(Arg&& arg, Args&&... args) ->
        decltype((*this)[std::forward<Arg>(arg)](std::forward<Args>(args)...))
        {
            return (*this)[std::forward<Arg>(arg)](std::forward<Args>(args)...);
        }

        /***********************************************************************
         *
         * Iteration
         *
         **********************************************************************/

        template <typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_element<cptr>(std::forward<Func>(f), detail::static_range<unsigned, NDim>{});
        }

        template <typename Func>
        void for_each_element(Func&& f)
        {
            for_each_element<pointer>(std::forward<Func>(f), detail::static_range<unsigned, NDim>{});
        }

        /***********************************************************************
         *
         * Basic getters
         *
         **********************************************************************/

        const_pointer cdata() const
        {
            return const_cast<marray_base&>(*this).data();
        }

        cptr data() const
        {
            return const_cast<marray_base&>(*this).data();
        }

        pointer data()
        {
            return data_;
        }

        template <typename=void, unsigned N=NDim, typename=detail::enable_if_t<N==1>>
        len_type length() const
        {
            return len_[0];
        }

        template <unsigned Dim>
        len_type length() const
        {
            static_assert(Dim < NDim, "Dim out of range");
            return len_[Dim];
        }

        len_type length(unsigned dim) const
        {
            MARRAY_ASSERT(dim < NDim);
            return len_[dim];
        }

        const std::array<len_type, NDim>& lengths() const
        {
            return len_;
        }

        template <typename=void, unsigned N=NDim, typename=detail::enable_if_t<N==1>>
        stride_type stride() const
        {
            return stride_[0];
        }

        template <unsigned Dim>
        stride_type stride() const
        {
            static_assert(Dim < NDim, "Dim out of range");
            return stride_[Dim];
        }

        stride_type stride(unsigned dim) const
        {
            MARRAY_ASSERT(dim < NDim);
            return stride_[dim];
        }

        const std::array<stride_type, NDim>& strides() const
        {
            return stride_;
        }

        static constexpr unsigned dimension()
        {
            return NDim;
        }

        friend std::ostream& operator<<(std::ostream& os, const marray_base& x)
        {
            for (unsigned i = 0;i < NDim-1;i++)
            {
                for (unsigned j = 0;j < i;j++) os << ' ';
                os << "{\n";
            }

            std::array<len_type,NDim-1> idx = {};
            auto data = x.data_;

            for (bool done = false;!done;)
            {
                for (unsigned i = 0;i < NDim-1;i++) os << ' ';
                os << '{';
                len_type n = x.len_[NDim-1];
                if (n > 0)
                {
                    for (len_type i = 0;i < n-1;i++)
                        os << data[i*x.stride_[NDim-1]] << ", ";
                    os << data[(n-1)*x.stride_[NDim-1]];
                }
                os << "}";

                for (unsigned i = NDim-1;i --> 0;)
                {
                    idx[i]++;
                    data += x.stride_[i];

                    if (idx[i] >= x.len_[i])
                    {
                        data -= idx[i]*x.stride_[i];
                        idx[i] = 0;
                        os << "\n";
                        for (unsigned j = 0;j < i;j++) os << ' ';
                        os << '}';
                        if (i == 0) done = true;
                    }
                    else
                    {
                        os << ",\n";
                        for (unsigned j = i+1;j < NDim-1;j++)
                        {
                            for (unsigned k = 0;k < j;k++) os << ' ';
                            os << "{\n";
                        }
                        break;
                    }
                }

                if (NDim == 1) break;
            }

            return os;
        }
};

}

#endif
