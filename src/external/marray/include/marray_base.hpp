#ifndef _MARRAY_MARRAY_BASE_HPP_
#define _MARRAY_MARRAY_BASE_HPP_

#include "utility.hpp"
#include "range.hpp"
#include "expression.hpp"

namespace MArray
{

template <typename Type, int NDim, int NIndexed, typename... Dims>
class marray_slice;

/**
 * Tensor base class.
 */
template <typename Type, int NDim, typename Derived, bool Owner>
class marray_base;

/**
 * A tensor (multi-dimensional array) view, which may either be mutable or immutable.
 *
 * @tparam Type     The type of the tensor elements. The view is immutable if this is const-qualified.
 *
 * @tparam NDim     The number of tensor dimensions, must be positive.
 */
template <typename Type, int NDim>
class marray_view;

/**
 * A tensor (multi-dimensional array) container.
 *
 * @tparam Type         The type of the tensor elements.
 *
 * @tparam NDim         The number of tensor dimensions, must be positive.
 *
 * @tparam Allocator    An allocator. If not specified, `std::allocator<Type>` is used.
 */
template <typename Type, int NDim, typename Allocator=std::allocator<Type>>
class marray;

template <typename Type, typename Derived, bool Owner>
class varray_base;

template <typename Type>
class varray_view;

template <typename Type, typename Allocator=std::allocator<Type>>
class varray;

template <typename Expr>
struct is_expression_arg_or_scalar;

/*
 * Convenient names for 1- and 2-dimensional array types.
 */
template <typename Type> using row_view = marray_view<Type, 1>;
template <typename Type, typename Allocator=std::allocator<Type>> using row = marray<Type, 1, Allocator>;

template <typename Type> using matrix_view = marray_view<Type, 2>;
template <typename Type, typename Allocator=std::allocator<Type>> using matrix = marray<Type, 2, Allocator>;

namespace detail
{

struct len_type_wrapper
{
    len_type value;

    len_type_wrapper(  signed     short value) : value(value) {}
    len_type_wrapper(unsigned     short value) : value(value) {}
    len_type_wrapper(  signed       int value) : value(value) {}
    len_type_wrapper(unsigned       int value) : value(value) {}
    len_type_wrapper(  signed      long value) : value(value) {}
    len_type_wrapper(unsigned      long value) : value(value) {}
    len_type_wrapper(  signed long long value) : value(value) {}
    len_type_wrapper(unsigned long long value) : value(value) {}

    operator len_type() const { return value; }
};

typedef std::initializer_list<len_type_wrapper> len_type_init;

template <typename Type, int NDim>
struct initializer_type;

template <typename Type>
struct initializer_type<Type, 0u>
{
    typedef Type type;
};

template <typename Type, int NDim>
struct initializer_type
{
    typedef std::initializer_list<
        typename initializer_type<Type, NDim-1>::type> type;
};

template <typename Func, typename Arg, typename... Args>
detail::enable_if_t<sizeof...(Args) &&
                    std::is_same<decltype(std::declval<Func&&>()(
                        std::declval<Arg&&>(),
                        std::declval<Args&&>()...)),void>::value>
call(Func&& f, Arg&& arg, Args&&... args)
{
    f(std::forward<Arg>(arg), std::forward<Args>(args)...);
}

template <typename Func, typename Arg, typename... Args>
detail::enable_if_t<std::is_same<decltype(std::declval<Func&&>()(
                        std::declval<Arg&&>())),void>::value>
call(Func&& f, Arg&& arg, Args&&...)
{
    f(std::forward<Arg>(arg));
}

template <typename Array>
class marray_iterator
{
    protected:
        Array* array_ = nullptr;
        len_type i_ = 0;

    public:
        typedef std::random_access_iterator_tag iterator_category;
        typedef decltype((*array_)[0]) value_type;
        typedef len_type difference_type;
        typedef typename std::remove_reference<value_type>::type* pointer;
        typedef value_type& reference;

        marray_iterator(Array& array, len_type i)
        : array_(&array), i_(i) {}

        bool operator==(const marray_iterator& other) const
        {
            return i_ == other.i_;
        }

        bool operator!=(const marray_iterator& other) const
        {
            return !(*this == other);
        }

        value_type operator*() const
        {
            return (*array_)[i_];
        }

        marray_iterator& operator++()
        {
            i_++;
            return *this;
        }

        marray_iterator operator++(int)
        {
            marray_iterator old(*this);
            i_++;
            return old;
        }

        marray_iterator& operator--()
        {
            i_--;
            return *this;
        }

        marray_iterator operator--(int)
        {
            marray_iterator old(*this);
            i_--;
            return old;
        }

        marray_iterator& operator+=(difference_type n)
        {
            i_ += n;
            return *this;
        }

        marray_iterator operator+(difference_type n) const
        {
            return marray_iterator(*array_, i_+n);
        }

        friend marray_iterator operator+(difference_type n, const marray_iterator& i)
        {
            return marray_iterator(*i.array_, i.i_+n);
        }

        marray_iterator& operator-=(difference_type n)
        {
            i_ -= n;
            return *this;
        }

        marray_iterator operator-(difference_type n) const
        {
            return marray_iterator(*array_, i_-n);
        }

        difference_type operator-(const marray_iterator& other) const
        {
            return i_ - other.i_;
        }

        bool operator<(const marray_iterator& other) const
        {
            return i_ < other.i_;
        }

        bool operator<=(const marray_iterator& other) const
        {
            return !(other < *this);
        }

        bool operator>(const marray_iterator& other) const
        {
            return other < *this;
        }

        bool operator>=(const marray_iterator& other) const
        {
            return !(*this < other);
        }

        value_type operator[](difference_type n) const
        {
            return (*array_)[i_+n];
        }

        friend void swap(marray_iterator& a, marray_iterator& b)
        {
            using std::swap;
            swap(a.array_, b.array_);
            swap(a.i_, b.i_);
        }
};

template <typename T>
struct is_row : std::false_type {};

template <typename T, typename D, bool O>
struct is_row<marray_base<T, 1, D, O>> : std::true_type {};

template <typename T>
struct is_row<marray_view<T, 1>> : std::true_type {};

template <typename T, typename A>
struct is_row<marray<T, 1, A>> : std::true_type {};

template <typename T, typename U, typename=void>
struct is_row_of : std::false_type {};

template <typename T, typename U>
struct is_row_of<T, U, enable_if_t<is_row<T>::value &&
    std::is_assignable<U&,typename T::value_type>::value>> : std::true_type {};

template <typename T>
struct is_matrix : std::false_type {};

template <typename T, typename D, bool O>
struct is_matrix<marray_base<T, 2, D, O>> : std::true_type {};

template <typename T>
struct is_matrix<marray_view<T, 2>> : std::true_type {};

template <typename T, typename A>
struct is_matrix<marray<T, 2, A>> : std::true_type {};

template <typename T, typename U, typename=void>
struct is_matrix_of : std::false_type {};

template <typename T, typename U>
struct is_matrix_of<T, U, enable_if_t<is_matrix<T>::value &&
    std::is_assignable<U&,typename T::value_type>::value>> : std::true_type {};

template <typename T, typename U, typename V=void>
using enable_if_matrix_of_t = enable_if_t<is_matrix_of<T,U>::value, V>;

template <typename T, typename U>
struct is_1d_container_of :
    std::integral_constant<bool, is_row_of<T,U>::value ||
                                 is_container_of<T,U>::value> {};

template <typename T, typename U, typename V=void>
using enable_if_1d_container_of_t = enable_if_t<is_1d_container_of<T,U>::value, V>;

template <typename T, typename U>
struct is_2d_container_of :
    std::integral_constant<bool, is_matrix_of<T,U>::value ||
                                 is_container_of_containers_of<T,U>::value> {};

template <typename T, typename U, typename V=void>
using enable_if_2d_container_of_t = enable_if_t<is_2d_container_of<T,U>::value, V>;

template <typename T>
enable_if_t<is_container<T>::value,len_type>
length(const T& len)
{
    return len.size();
}

template <typename T>
enable_if_t<is_row<T>::value,len_type>
length(const T& len)
{
    return len.length(0);
}

template <typename T>
enable_if_t<is_container_of_containers<T>::value,len_type>
length(const T& len, int dim)
{
    if (dim == 0) return len.size();
    else
    {
        MARRAY_ASSERT(dim == 1);
        auto it = len.begin();
        if (it == len.end()) return 0;
        len_type l = it->size();
        while (++it != len.end()) MARRAY_ASSERT((len_type)it->size() == l);
        return l;
    }
}

template <typename T>
enable_if_t<is_matrix<T>::value,len_type>
length(const T& len, int dim)
{
    return len.length(dim);
}

template <typename T>
class array_1d
{
    protected:
        struct adaptor_base
        {
            len_type len;

            adaptor_base(len_type len) : len(len) {}

            virtual ~adaptor_base() {}

            virtual void slurp(T*) const = 0;

            virtual adaptor_base& copy(adaptor_base& other) = 0;
        };

        template <typename U>
        struct adaptor : adaptor_base
        {
            U data;
            using adaptor_base::len;

            adaptor(U data)
            : adaptor_base(detail::length(data)), data(std::move(data)) {}

            virtual void slurp(T* x) const override
            {
                std::copy_n(data.begin(), len, x);
            }

            virtual adaptor_base& copy(adaptor_base& other) override
            {
            	return *(new (static_cast<adaptor*>(&other)) adaptor(*this));
            }
        };

        constexpr static size_t _s1 = sizeof(adaptor<T*&>);
        constexpr static size_t _s2 = sizeof(adaptor<short_vector<T,MARRAY_OPT_NDIM>>);
        constexpr static size_t max_adaptor_size = _s1 > _s2 ? _s1 : _s2;

        std::aligned_storage_t<max_adaptor_size> raw_adaptor_;
        adaptor_base& adaptor_;

        template <typename U>
        adaptor_base& adapt(U&& data)
        {
            return *(new (&raw_adaptor_) adaptor<U>(data));
        }

    public:
        array_1d()
        : adaptor_(adapt(std::array<T,0>{})) {}

        array_1d(const array_1d& other)
        : adaptor_(other.adaptor_.copy(reinterpret_cast<adaptor_base&>(raw_adaptor_))) {}

        template <typename... Args, typename =
            detail::enable_if_t<detail::are_convertible<T,Args...>::value>>
        array_1d(Args&&... args)
        : adaptor_(adapt(short_vector<T,MARRAY_OPT_NDIM>{(T)args...})) {}

        template <typename U, typename=detail::enable_if_1d_container_of_t<U,T>>
        array_1d(const U& data)
        : adaptor_(adapt(data)) {}

        ~array_1d() { adaptor_.~adaptor_base(); }

        template <size_t N>
        void slurp(std::array<T, N>& x) const
        {
            MARRAY_ASSERT((len_type)N >= size());
            adaptor_.slurp(x.data());
        }

        void slurp(std::vector<T>& x) const
        {
            x.resize(size());
            adaptor_.slurp(x.data());
        }

        template <size_t N>
        void slurp(short_vector<T, N>& x) const
        {
            x.resize(size());
            adaptor_.slurp(x.data());
        }

        void slurp(row<T>& x) const
        {
            x.reset({size()});
            adaptor_.slurp(x.data());
        }

        len_type size() const { return adaptor_.len; }
};

template <typename T>
class array_2d
{
    protected:
        struct adaptor_base
        {
            std::array<len_type,2> len;

            adaptor_base(len_type len0, len_type len1) : len{len0, len1} {}

            virtual ~adaptor_base() {}

            virtual void slurp(T* x, len_type rs, len_type cs) const = 0;

            virtual void slurp(std::vector<std::vector<T>>&) const = 0;
        };

        template <typename U>
        struct adaptor : adaptor_base
        {
            static constexpr bool IsMatrix = is_matrix<typename std::decay<U>::type>::value;

            U data;
            using adaptor_base::len;

            adaptor(U data)
            : adaptor_base(detail::length(data, 0), detail::length(data, 1)),
              data(data) {}

            template <bool IsMatrix_ = IsMatrix>
            typename std::enable_if<!IsMatrix_>::type
            do_slurp(T* x, len_type rs, len_type cs) const
            {
                int i = 0;
                for (auto it = data.begin(), end = data.end();it != end;++it)
                {
                    int j = 0;
                    for (auto it2 = it->begin(), end2 = it->end();it2 != end2;++it2)
                    {
                        x[i*rs + j*cs] = *it2;
                        j++;
                    }
                    i++;
                }
            }

            template <bool IsMatrix_ = IsMatrix>
            typename std::enable_if<!IsMatrix_>::type
            do_slurp(std::vector<std::vector<T>>& x) const
            {
                x.clear();
                for (auto it = data.begin(), end = data.end();it != end;++it)
                {
                    x.emplace_back(it->begin(), it->end());
                }
            }

            template <bool IsMatrix_ = IsMatrix>
            typename std::enable_if<IsMatrix_>::type
            do_slurp(T* x, len_type rs, len_type cs) const
            {
                for (len_type i = 0;i < len[0];i++)
                {
                    for (len_type j = 0;j < len[1];j++)
                    {
                        x[i*rs + j*cs] = data[i][j];
                    }
                }
            }

            template <bool IsMatrix_ = IsMatrix>
            typename std::enable_if<IsMatrix_>::type
            do_slurp(std::vector<std::vector<T>>& x) const
            {
                x.resize(len[0]);
                for (len_type i = 0;i < len[0];i++)
                {
                    x[i].resize(len[1]);
                    for (len_type j = 0;j < len[1];j++)
                    {
                        x[i][j] = data[i][j];
                    }
                }
            }

            virtual void slurp(T* x, len_type rs, len_type cs) const override
            {
                do_slurp(x, rs, cs);
            }

            virtual void slurp(std::vector<std::vector<T>>& x) const override
            {
                do_slurp(x);
            }
        };

        constexpr static size_t _s1 = sizeof(adaptor<std::initializer_list<std::initializer_list<int>>>);
        constexpr static size_t _s2 = sizeof(adaptor<const matrix<int>&>);
        constexpr static size_t max_adaptor_size = _s1 > _s2 ? _s1 : _s2;

        std::aligned_storage_t<max_adaptor_size> raw_adaptor_;
        adaptor_base& adaptor_;

        template <typename U>
        adaptor_base& adapt(U data)
        {
            return *(new (&raw_adaptor_) adaptor<U>(data));
        }

        adaptor_base& adapt(const array_2d& other)
        {
            memcpy(&raw_adaptor_, &other.raw_adaptor_, sizeof(raw_adaptor_));

            return *reinterpret_cast<adaptor_base*>(
                reinterpret_cast<char*>(this) +
                (reinterpret_cast<const char*>(&other.adaptor_) -
                 reinterpret_cast<const char*>(&other)));
        }

    public:
        array_2d(const array_2d& other)
        : adaptor_(adapt(other)) {}

        array_2d(std::initializer_list<std::initializer_list<T>> data)
        : adaptor_(adapt<std::initializer_list<std::initializer_list<T>>>(data)) {}

        template <typename U, typename=detail::enable_if_assignable_t<T&,U>>
        array_2d(std::initializer_list<std::initializer_list<U>> data)
        : adaptor_(adapt<std::initializer_list<std::initializer_list<U>>>(data)) {}

        template <typename U, typename=detail::enable_if_1d_container_of_t<U,T>>
        array_2d(std::initializer_list<U> data)
        : adaptor_(adapt<std::initializer_list<U>>(data)) {}

        template <typename U, typename=detail::enable_if_2d_container_of_t<U,T>>
        array_2d(const U& data)
        : adaptor_(adapt<const U&>(data)) {}

        ~array_2d() { adaptor_.~adaptor_base(); }

        void slurp(std::vector<std::vector<T>>& x) const { adaptor_.slurp(x); }

        template <size_t M, size_t N>
        void slurp(std::array<std::array<T, N>, M>& x) const
        {
            MARRAY_ASSERT((len_type)M >= length(0));
            MARRAY_ASSERT((len_type)N >= length(1));
            adaptor_.slurp(&x[0][0], N, 1);
        }

        template <size_t M, size_t N>
        void slurp(short_vector<std::array<T, N>, M>& x) const
        {
            x.resize(length(0));
            MARRAY_ASSERT((len_type)N >= length(1));
            adaptor_.slurp(&x[0][0], N, 1);
        }

        void slurp(matrix<T>& x, layout layout = DEFAULT) const
        {
            x.reset({length(0), length(1)}, layout);
            adaptor_.slurp(x.data(), x.stride(0), x.stride(1));
        }

        len_type length(int dim) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < 2);
            return adaptor_.len[dim];
        }
};

}
}

#include "expression.hpp"
#include "marray_slice.hpp"
#include "miterator.hpp"

namespace MArray
{

template <typename Type, int NDim, typename Derived, bool Owner>
class marray_base
{
    static_assert(NDim > 0, "NDim must be positive");

    template <typename, int, typename, bool> friend class marray_base;
    template <typename, int> friend class marray_view;
    template <typename, int, typename> friend class marray;
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
        typedef detail::marray_iterator<marray_base> iterator;
        typedef detail::marray_iterator<const marray_base> const_iterator;
        typedef std::reverse_iterator<iterator> reverse_iterator;
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

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

        /**
         * Reset to an empty view.
         */
        void reset()
        {
            data_ = nullptr;
            len_ = {};
            stride_ = {};
        }

        /**
         * Reset to a view of the given tensor, view, or partially-indexed tensor.
         *
         * @param other     The tensor, view, or partially-indexed tensor to view.
         *                  If this is a mutable view (the value type is not
         *                  const-qualified), then `other` may not be a const-
         *                  qualified tensor instance or a view with a const-
         *                  qualified value type. May be either an lvalue- or
         *                  rvalue-reference.
         */
#if MARRAY_DOXYGEN
        void reset(tensor_or_view_reference other)
#else
        template <typename U, bool O, typename D,
            typename=detail::enable_if_convertible_t<
                typename marray_base<U, NDim, D, O>::cptr,pointer>>
        void reset(const marray_base<U, NDim, D, O>& other)
#endif
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

        template <typename U, int OldNDim, int NIndexed,
            typename... Dims, typename=detail::enable_if_convertible_t<U*,pointer>>
        void reset(const marray_slice<U, OldNDim, NIndexed, Dims...>& other)
        {
            reset(other.view());
        }

        /**
         * Reset to a view that wraps a raw data pointer, using the provided
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
        void reset(const detail::array_1d<len_type>& len, pointer ptr,
                   layout layout = DEFAULT)
        {
            reset(len, ptr, strides(len, layout));
        }

        /**
         * Reset to a view that wraps a raw data pointer, using the provided
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
        void reset(const detail::array_1d<len_type>& len, pointer ptr,
                   const detail::array_1d<stride_type>& stride)
        {
            data_ = ptr;
            len.slurp(len_);
            stride.slurp(stride_);
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

        template <typename Ptr, typename Func, int... I>
        void for_each_element(Func&& f,
                              detail::integer_sequence<int, I...>) const
        {
            miterator<NDim, 1> it(len_, stride_);
            Ptr ptr = const_cast<Ptr>(data_);
            while (it.next(ptr))
                detail::call(std::forward<Func>(f), *ptr, it.position()[I]...);
        }

        void set_lengths(int i, std::array<len_type, NDim>& len,
                         std::initializer_list<Type> data)
        {
            len[i] = data.size();
        }

        template <typename U>
        void set_lengths(int i, std::array<len_type, NDim>& len,
                         std::initializer_list<U> data)
        {
            len[i] = data.size();
            set_lengths(i+1, len, *data.begin());
        }

        void set_data(int i, pointer ptr, std::initializer_list<Type> data)
        {
            auto it = data.begin();
            for (len_type j = 0;j < len_[i];j++)
            {
                ptr[j*stride_[i]] = *it;
                ++it;
            }
        }

        template <typename U>
        void set_data(int i, pointer ptr, std::initializer_list<U> data)
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

        /**
         * Return the strides for a hypothetical tensor with the given lengths and layout.
         *
         * @param len       The lengths of the hypothetical tensor.
         *
         * @param layout    The layout to use, either #ROW_MAJOR or #COLUMN_MAJOR.
         *                  If omitted, the default layout is used.
         */
        static std::array<stride_type, NDim>
        strides(const detail::array_1d<len_type>& len, layout layout = DEFAULT)
        {
            //TODO: add alignment option

            MARRAY_ASSERT(len.size() == NDim);

            std::array<len_type, NDim> len_;
            len.slurp(len_);
            std::array<stride_type, NDim> stride;

            if (layout == ROW_MAJOR)
            {
                stride[NDim-1] = 1;
                for (auto i : reversed_range(NDim-1))
                    stride[i] = stride[i+1]*len_[i+1];
            }
            else
            {
                stride[0] = 1;
                for (auto i : range(1,NDim))
                    stride[i] = stride[i-1]*len_[i-1];
            }

            return stride;
        }

        /**
         * Return the number of elements in a hypothetical tensor with the given lengths.
         *
         * @param len       The lengths of the hypothetical tensor.
         *
         * @return          The number of elements, which is equal to the product of the lengths.
         */
        static stride_type size(const detail::array_1d<len_type>& len)
        {
            //TODO: add alignment option

            len_vector len_;
            len.slurp(len_);

            stride_type s = 1;
            for (auto& l : len_) s *= l;
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

        /**
         * Set the tensor data using a nested initializer list.
         *
         * @param data  A nested initializer list. The number of levels must be
         *              equal to the number of dimensions, and the supplied initializer
         *              lists must be "dense", i.e. every element must be specified.
         *
         * @return      *this
         */
        Derived& operator=(initializer_type data)
        {
            std::array<len_type, NDim> len;
            set_lengths(0, len, data);
            MARRAY_ASSERT(len == len_);
            set_data(0, data_, data);
            return static_cast<Derived&>(*this);
        }

        /**
         * Set the tensor elements to the result of the specified expression.
         *
         * For a tensor ([marray](@ref MArray::marray)), the instance must not be
         * const-qualified. For a tensor view (marray_view),
         * the value type must not be const-qualified.
         *
         * @param other     An expression object; either the result of one or
         *                  more mathematical operations on a set of tensors, a
         *                  single tensor or tensor view, or a scalar. The dimensions of
         *                  the expression or tensor must either match those of this tensor
         *                  or be broadcast-compatible.
         *
         * @return      *this
         */
#if !MARRAY_DOXYGEN
        template <typename Expression,
            typename=detail::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
#endif
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

        /**
         * Increment the elements by the result of the specified expression.
         *
         * For a tensor ([marray](@ref MArray::marray)), the instance must not be
         * const-qualified. For a tensor view (marray_view),
         * the value type must not be const-qualified.
         *
         * @param other     An expression object; either the result of one or
         *                  more mathematical operations on a set of tensors, a
         *                  single tensor or tensor view, or a scalar. The dimensions of
         *                  the expression or tensor must either match those of this tensor
         *                  or be broadcast-compatible.
         *
         * @return      *this
         */
#if !MARRAY_DOXYGEN
        template <typename Expression,
            typename=detail::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
#endif
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

        /**
         * Decrement the elements by the result of the specified expression.
         *
         * For a tensor ([marray](@ref MArray::marray)), the instance must not be
         * const-qualified. For a tensor view (marray_view),
         * the value type must not be const-qualified.
         *
         * @param other     An expression object; either the result of one or
         *                  more mathematical operations on a set of tensors, a
         *                  single tensor or tensor view, or a scalar. The dimensions of
         *                  the expression or tensor must either match those of this tensor
         *                  or be broadcast-compatible.
         *
         * @return      *this
         */
#if !MARRAY_DOXYGEN
        template <typename Expression,
            typename=detail::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
#endif
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

        /**
         * Perform an element-wise multiplication by the result of the specified expression.
         *
         * For a tensor ([marray](@ref MArray::marray)), the instance must not be
         * const-qualified. For a tensor view (marray_view),
         * the value type must not be const-qualified.
         *
         * @param other     An expression object; either the result of one or
         *                  more mathematical operations on a set of tensors, a
         *                  single tensor or tensor view, or a scalar. The dimensions of
         *                  the expression or tensor must either match those of this tensor
         *                  or be broadcast-compatible.
         *
         * @return      *this
         */
#if !MARRAY_DOXYGEN
        template <typename Expression,
            typename=detail::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
#endif
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

        /**
         * Perform an element-wise division by the result of the specified expression.
         *
         * For a tensor ([marray](@ref MArray::marray)), the instance must not be
         * const-qualified. For a tensor view (marray_view),
         * the value type must not be const-qualified.
         *
         * @param other     An expression object; either the result of one or
         *                  more mathematical operations on a set of tensors, a
         *                  single tensor or tensor view, or a scalar. The dimensions of
         *                  the expression or tensor must either match those of this tensor
         *                  or be broadcast-compatible.
         *
         * @return      *this
         */
#if !MARRAY_DOXYGEN
        template <typename Expression,
            typename=detail::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
#endif
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

        /**
         * Return true if this tensor is the same size and shape and has the same elements
         * as another tensor.
         *
         * @param other     A tensor or tensor view against which to check.
         *
         * @return          True if all elements match, false otherwise. If the tensors
         *                  are not the same size and shape, then false.
         */
#if MARRAY_DOXYGEN
        bool
#else
        template <typename U, int N, typename D, bool O>
        detail::enable_if_t<N==NDim, bool>
#endif
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

        template <typename U, int N, typename D, bool O>
        detail::enable_if_t<N!=NDim, bool>
        operator==(const marray_base<U, N, D, O>&) const
        {
            return false;
        }

        /**
         * Return false if this tensor is the same size and shape and has the same elements
         * as another tensor.
         *
         * @param other     A tensor or tensor view against which to check.
         *
         * @return          False if all elements match, true otherwise. If the tensors
         *                  are not the same size and shape, then true.
         */
#if !MARRAY_DOXYGEN
        template <typename U, int N, typename D, bool O>
#endif
        bool operator!=(const marray_base<U, N, D, O>& other) const
        {
            return !(*this == other);
        }

        /***********************************************************************
         *
         * Views
         *
         **********************************************************************/

        /**
         * Return an immutable view of this tensor.
         *
         * @return an immutable view.
         */
#if MARRAY_DOXYGEN
        immutable_view
#else
        marray_view<const Type, NDim>
#endif
        cview() const
        {
            return const_cast<marray_base&>(*this).view();
        }

        /**
         * Return a view of this tensor.
         *
         * @return  A possibly-mutable tensor view. For a tensor
         *          ([marray](@ref MArray::marray)), the returned view is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view (marray_view),
         *          the returned view is mutable if the value type is not
         *          const-qualified.
         */
#if MARRAY_DOXYGEN
        possible_immutable_view
#else
        marray_view<ctype, NDim>
#endif
        view() const
        {
            return const_cast<marray_base&>(*this).view();
        }

        /**
         * Return a mutable view of this tensor.
         *
         * @return a mutable view.
         */
#if MARRAY_DOXYGEN
        mutable_view
#else
        marray_view<Type, NDim>
#endif
        view()
        {
            return *this;
        }

        /**
         * Return an immutable view of the given tensor.
         *
         * @param x The tensor to view.
         *
         * @return an immutable view.
         */
        friend
#if MARRAY_DOXYGEN
        immutable_view
#else
        marray_view<const Type, NDim>
#endif
        cview(const marray_base& x)
        {
            return x.view();
        }

        /**
         * Return a view of the given tensor.
         *
         * @param x The tensor to view.
         *
         * @return  A possibly-mutable tensor view. For a tensor
         *          ([marray](@ref MArray::marray)), the returned view is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view (marray_view),
         *          the returned view is mutable if the value type is not
         *          const-qualified.
         */
        friend
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        marray_view<ctype, NDim>
#endif
        view(const marray_base& x)
        {
            return x.view();
        }

        /**
         * Return a mutable view of the given tensor.
         *
         * @param x The tensor to view.
         *
         * @return a mutable view.
         */
        friend
#if MARRAY_DOXYGEN
        mutable_view
#else
        marray_view<Type, NDim>
#endif
        view(marray_base& x)
        {
            return x.view();
        }

        /***********************************************************************
         *
         * Iterators
         *
         **********************************************************************/

        const_iterator cbegin() const
        {
            return const_iterator{*this, 0};
        }

        const_iterator begin() const
        {
            return const_iterator{*this, 0};
        }

        iterator begin()
        {
            return iterator{*this, 0};
        }

        const_iterator cend() const
        {
            return const_iterator{*this, len_[0]};
        }

        const_iterator end() const
        {
            return const_iterator{*this, len_[0]};
        }

        iterator end()
        {
            return iterator{*this, len_[0]};
        }

        const_reverse_iterator crbegin() const
        {
            return const_reverse_iterator{end()};
        }

        const_reverse_iterator rbegin() const
        {
            return const_reverse_iterator{end()};
        }

        reverse_iterator rbegin()
        {
            return reverse_iterator{end()};
        }

        const_reverse_iterator crend() const
        {
            return const_reverse_iterator{begin()};
        }

        const_reverse_iterator rend() const
        {
            return const_reverse_iterator{begin()};
        }

        reverse_iterator rend()
        {
            return reverse_iterator{begin()};
        }

        /***********************************************************************
         *
         * Shifting
         *
         **********************************************************************/

        marray_view<ctype, NDim> shifted(const detail::array_1d<len_type>& n) const
        {
            return const_cast<marray_base&>(*this).shifted(n);
        }

        /**
         * Return a view that references elements whose indices are
         * shifted by the given amount along each dimension.
         *
         * An index `i` in the shifted view is equivalent to an index `i+n[i]`
         * in the original tensor or tensor view.
         *
         * @param n The amount by which to shift for each dimension. May be any
         *          one-dimensional container type whose elements are convertible
         *          to a tensor length, including initializer lists.
         *
         * @return  A possibly-mutable tensor view. For a tensor
         *          ([marray](@ref MArray::marray)), the returned view is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view (marray_view),
         *          the returned view is mutable if the value type is not
         *          const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        marray_view<Type, NDim>
#endif
        shifted(const detail::array_1d<len_type>& n)
        {
            marray_view<Type,NDim> r(*this);
            r.shift(n);
            return r;
        }

        template <typename=void, int N=NDim, typename=detail::enable_if_t<N==1>>
        marray_view<ctype,1> shifted(len_type n) const
        {
            return const_cast<marray_base&>(*this).shifted(n);
        }

        /**
         * Return a view that references elements whose indices are
         * shifted by the given amount.
         *
         * This overload is only available for vectors or vector views.
         * An index `i` in the shifted view is equivalent to an index `i+n`
         * in the original tensor or tensor view.
         *
         * @param n The amount by which to shift.
         *
         * @return  A possibly-mutable tensor view. For a tensor
         *          ([marray](@ref MArray::marray)), the returned view is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view (marray_view),
         *          the returned view is mutable if the value type is not
         *          const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        template <typename=void, int N=NDim, typename=detail::enable_if_t<N==1>>
        marray_view<Type,1>
#endif
        shifted(len_type n)
        {
            return shifted(0, n);
        }

        template <int Dim>
        marray_view<ctype, NDim> shifted(len_type n) const
        {
            return const_cast<marray_base&>(*this).shifted<Dim>(n);
        }

        /**
         * Return a view that references elements whose indices are
         * shifted by the given amount along one dimension.
         *
         * Only for the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i+n` in the original tensor or tensor view.
         *
         * @tparam Dim  The dimension along which to shift the returned view.
         *
         * @param n     The amount by which to shift.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view (marray_view),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
        template <int Dim>
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        marray_view<Type, NDim>
#endif
        shifted(len_type n)
        {
            return shifted(Dim, n);
        }

        marray_view<ctype, NDim> shifted(int dim, len_type n) const
        {
            return const_cast<marray_base&>(*this).shifted(dim, n);
        }

        /**
         * Return a view that references elements whose indices are
         * shifted by the given amount along one dimension.
         *
         * Only for the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i+n` in the original tensor or tensor view.
         *
         * @param dim   The dimension along which to shift the returned view.
         *
         * @param n     The amount by which to shift.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view (marray_view),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        marray_view<Type, NDim>
#endif
        shifted(int dim, len_type n)
        {
            marray_view<Type,NDim> r(*this);
            r.shift(dim, n);
            return r;
        }

        template <typename=void, int N=NDim, typename=detail::enable_if_t<N==1>>
        marray_view<ctype,1> shifted_down() const
        {
            return const_cast<marray_base&>(*this).shifted_down();
        }

        /**
         * Return a view that references elements whose indices are
         * shifted "down".
         *
         * This overload is only available for vectors and vector views.
         * An index `i` in the shifted view
         * is equivalent to an index `i+n` in the original vector or vector view,
         * where `n` is the vector length.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view (marray_view),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        template <typename=void, int N=NDim, typename=detail::enable_if_t<N==1>>
        marray_view<Type,1>
#endif
        shifted_down()
        {
            return shifted_down(0);
        }

        template <int Dim>
        marray_view<ctype,NDim> shifted_down() const
        {
            return const_cast<marray_base&>(*this).shifted_down<Dim>();
        }

        /**
         * Return a view that references elements whose indices are
         * shifted "down" along one dimension.
         *
         * Only for the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i+n` in the original tensor or tensor view,
         * where `n` is the tensor length in that dimension.
         *
         * @tparam Dim  The dimension along which to shift the returned view.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view (marray_view),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
        template <int Dim>
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        marray_view<Type,NDim>
#endif
        shifted_down()
        {
            return shifted_down(Dim);
        }

        marray_view<ctype,NDim> shifted_down(int dim) const
        {
            return const_cast<marray_base&>(*this).shifted_down(dim);
        }

        /**
         * Return a view that references elements whose indices are
         * shifted "down" along one dimension.
         *
         * Only for the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i+n` in the original tensor or tensor view,
         * where `n` is the tensor length in that dimension.
         *
         * @param dim   The dimension along which to shift the returned view.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view (marray_view),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        marray_view<Type,NDim>
#endif
        shifted_down(int dim)
        {
            return shifted(dim, len_[dim]);
        }

        template <typename=void, int N=NDim, typename=detail::enable_if_t<N==1>>
        marray_view<ctype,1> shifted_up() const
        {
            return const_cast<marray_base&>(*this).shifted_up();
        }

        /**
         * Return a view that references elements whose indices are
         * shifted "up".
         *
         * This overload is only available for vectors and vector views.
         * An index `i` in the shifted view
         * is equivalent to an index `i-n` in the original vector or vector view,
         * where `n` is the vector length.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view (marray_view),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        template <typename=void, int N=NDim, typename=detail::enable_if_t<N==1>>
        marray_view<Type,1>
#endif
        shifted_up()
        {
            return shifted_up(0);
        }

        template <int Dim>
        marray_view<ctype,NDim> shifted_up() const
        {
            return const_cast<marray_base&>(*this).shifted_up<Dim>();
        }

        /**
         * Return a view that references elements whose indices are
         * shifted "up" along one dimension.
         *
         * Only for the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i-n` in the original tensor or tensor view,
         * where `n` is the tensor length in that dimension.
         *
         * @tparam Dim  The dimension along which to shift the returned view.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view (marray_view),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
        template <int Dim>
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        marray_view<Type,NDim>
#endif
        shifted_up()
        {
            return shifted_up(Dim);
        }

        marray_view<ctype,NDim> shifted_up(int dim) const
        {
            return const_cast<marray_base&>(*this).shifted_up(dim);
        }

        /**
         * Return a view that references elements whose indices are
         * shifted "up" along one dimension.
         *
         * Only for the specified dimension, an index `i` in the shifted view
         * is equivalent to an index `i-n` in the original tensor or tensor view,
         * where `n` is the tensor length in that dimension.
         *
         * @param dim   The dimension along which to shift the returned view.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view (marray_view),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        marray_view<Type,NDim>
#endif
        shifted_up(int dim)
        {
            return shifted(dim, -len_[dim]);
        }

        /***********************************************************************
         *
         * Permutation
         *
         **********************************************************************/

        marray_view<ctype,NDim> permuted(const detail::array_1d<int>& perm) const
        {
            return const_cast<marray_base&>(*this).permuted(perm);
        }

        /**
         * Return a permuted view.
         *
         * Indexing into dimension `i` of the permuted view is equivalent to
         * indexing into dimension `perm[i]` of the original tensor or tensor
         * view.
         *
         * @param perm  The permutation vector. May be any
         *              one-dimensional container type whose elements are convertible
         *              to `int`, including initializer lists. The values must form
         *              a permutation of `[0,NDim)`, where `NDim` is the number of
         *              tensor dimensions.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view (marray_view),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        marray_view<Type,NDim>
#endif
        permuted(const detail::array_1d<int>& perm)
        {
            marray_view<Type,NDim> r(*this);
            r.permute(perm);
            return r;
        }

        template <int N=NDim, typename=detail::enable_if_t<N==2>>
        marray_view<ctype, NDim> transposed() const
        {
            return const_cast<marray_base&>(*this).transposed();
        }

        /**
         * Return a transposed view.
         *
         * This overload is only available for matrices and matrix views.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view (marray_view),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        template <int N=NDim, typename=detail::enable_if_t<N==2>>
        marray_view<Type, NDim>
#endif
        transposed()
        {
            return permuted({1, 0});
        }

        template <int N=NDim, typename=detail::enable_if_t<N==2>>
        marray_view<ctype, NDim> T() const
        {
            return const_cast<marray_base&>(*this).T();
        }

        /**
         * Return a transposed view.
         *
         * This overload is only available for matrices and matrix views.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view (marray_view),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        template <int N=NDim, typename=detail::enable_if_t<N==2>>
        marray_view<Type, NDim>
#endif
        T()
        {
            return transposed();
        }

        /***********************************************************************
         *
         * Dimension change
         *
         **********************************************************************/

        template <int NewNDim>
        marray_view<ctype, NewNDim> lowered(const detail::array_1d<int>& split) const
        {
            return const_cast<marray_base&>(*this).lowered<NewNDim>(split);
        }

        /**
         * Return a view of lower dimensionality.
         *
         * The values along each lowered dimension (which corresponds to one or
         * more dimensions in the original tensor or tensor view) must have a
         * consistent stride.
         *
         * @tparam NewNDim  The number of dimensions in the lowered view.
         *
         * @param split The "split" or "pivot" vector. The number of split points/pivots
         *              must be equal to the number of dimensions in the lowered view
         *              minus one. Dimensions `[0,split[0])` correspond to the
         *              first dimension of the return view, dimensions `[split[NewNDim-1],NDim)`
         *              correspond to the last dimension of the returned view, and
         *              dimensions `[split[i-1],split[i])` correspond to the `i`th
         *              dimension of the return view otherwise, where `NDim` is the
         *              dimensionality of the original tensor. The split points must be
         *              in increasing order. May be any
         *              one-dimensional container type whose elements are convertible
         *              to `int`, including initializer lists.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view (marray_view),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
        template <int NewNDim>
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        marray_view<Type, NewNDim>
#endif
        lowered(const detail::array_1d<int>& split)
        {
            static_assert(NewNDim > 0 && NewNDim <= NDim,
                          "Cannot split into this number of dimensions");

            constexpr auto NSplit = NewNDim-1;
            MARRAY_ASSERT(split.size() == NSplit);

            std::array<int, NSplit> split_;
            split.slurp(split_);

            for (auto i : range(NSplit))
            {
                MARRAY_ASSERT(split_[i] <= NDim);
                if (i != 0) MARRAY_ASSERT(split_[i-1] <= split_[i]);
            }

            std::array<len_type, NSplit+1> newlen;
            std::array<stride_type, NSplit+1> newstride;

            for (auto i : range(NSplit+1))
            {
                auto begin = (i == 0 ? 0 : split_[i-1]);
                auto end = (i == NSplit ? NDim-1 : split_[i]-1);
                if (begin > end) continue;

                if (stride_[begin] < stride_[end] ||
                    (stride_[begin] == stride_[end] && len_[begin] == 1))
                {
                    newlen[i] = len_[end];
                    newstride[i] = stride_[begin];
                    for (auto j : range(begin,end))
                    {
                        MARRAY_ASSERT(stride_[j+1] == stride_[j]*len_[j]);
                        newlen[i] *= len_[j];
                    }
                }
                else
                {
                    newlen[i] = len_[end];
                    newstride[i] = stride_[end];
                    for (auto j : range(begin,end))
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

        /**
         * Return a view where the order of the indices along each dimension has
         * been reversed.
         *
         * An index of `i` in the reversed view corresponds to an index of
         * `n-1-i` in the original tensor or view, where `n` is the tensor length along
         * that dimension.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view (marray_view),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        marray_view<Type, NDim>
#endif
        reversed()
        {
            marray_view<Type,NDim> r(*this);
            r.reverse();
            return r;
        }

        template <int Dim>
        marray_view<ctype, NDim> reversed() const
        {
            return const_cast<marray_base&>(*this).reversed<Dim>();
        }

        /**
         * Return a view where the order of the indices along the given dimension has
         * been reversed.
         *
         * Only for the indicated dimension, an index of `i` in the reversed tensor corresponds to an index of
         * `n-1-i` in the original tensor, where `n` is the tensor length along
         * that dimension.
         *
         * @tparam Dim  The dimension along which to reverse the indices.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view (marray_view),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
        template <int Dim>
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        marray_view<Type, NDim>
#endif
        reversed()
        {
            return reversed(Dim);
        }

        marray_view<ctype, NDim> reversed(int dim) const
        {
            return const_cast<marray_base&>(*this).reversed(dim);
        }

        /**
         * Return a view where the order of the indices along the given dimension has
         * been reversed.
         *
         * Only for the indicated dimension, an index of `i` in the reversed tensor corresponds to an index of
         * `n-1-i` in the original tensor, where `n` is the tensor length along
         * that dimension.
         *
         * @param dim   The dimension along which to reverse the indices.
         *
         * @return      A possibly-mutable tensor view. For a tensor
         *              ([marray](@ref MArray::marray)), the returned view is
         *              mutable if the instance is not const-qualified.
         *              For a tensor view (marray_view),
         *              the returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        marray_view<Type, NDim>
#endif
        reversed(int dim)
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

        /**
         * Return an immutable reference to the first element in this vector.
         *
         * This overload is only avaialble for vectors and vector views.
         *
         * @return  An immutable reference.
         */
#if MARRAY_DOXYGEN
        immutable_reference
#else
        template <int Dim=0, int N=NDim>
        detail::enable_if_t<N==1, const_reference>
#endif
        cfront() const
        {
            return const_cast<marray_base&>(*this).front<Dim>();
        }

        /**
         * Return a reference to the first element in this vector.
         *
         * This overload is only avaialble for vectors and vector views.
         *
         * @return  For a tensor ([marray](@ref MArray::marray)), the returned reference is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view (marray_view),
         *          the returned reference is mutable if the value type is not
         *          const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_reference
#else
        template <int Dim=0, int N=NDim>
        detail::enable_if_t<N==1, cref>
#endif
        front() const
        {
            return const_cast<marray_base&>(*this).front<Dim>();
        }

        /**
         * Return a mutable reference to the first element in this vector.
         *
         * This overload is only avaialble for vectors and vector views.
         *
         * @return  A mutable reference.
         */
#if MARRAY_DOXYGEN
        mutable_reference
#else
        template <int Dim=0, int N=NDim>
        detail::enable_if_t<N==1, reference>
#endif
        front()
        {
            static_assert(Dim == 0, "Dim out of range");
            MARRAY_ASSERT(len_[0] > 0);
            return data_[0];
        }

        template <int N=NDim>
        detail::enable_if_t<N==1, const_reference>
        cfront(int dim) const
        {
            return const_cast<marray_base&>(*this).front(dim);
        }

        template <int N=NDim>
        detail::enable_if_t<N==1, cref>
        front(int dim) const
        {
            return const_cast<marray_base&>(*this).front(dim);
        }

        template <int N=NDim>
        detail::enable_if_t<N==1, reference>
        front(int dim)
        {
            (void)dim;
            MARRAY_ASSERT(dim == 0);
            return front();
        }

        /**
         * Return an immutable reference or view to the first element or face along the specified dimension.
         *
         * If the tensor has more than one dimension, a tensor face is returned.
         * Otherwise, a reference to an element is returned.
         *
         * @tparam Dim   The dimension along which to extract the reference or face.
         *
         * @return      an immutable reference or view.
         */
#ifdef MARRAY_DOXYGEN
        template <int Dim>
        immutable_reference_or_view
#else
        template <int Dim, int N=NDim>
        detail::enable_if_t<N!=1, marray_view<const Type, NDim-1>>
#endif
        cfront() const
        {
            return const_cast<marray_base&>(*this).front<Dim>();
        }

        /**
         * Return a reference view to the first element or face along the specified dimension.
         *
         * If the tensor has more than one dimension, a tensor face is returned.
         * Otherwise, a reference to an element is returned.
         *
         * @tparam Dim  The dimension along which to extract the reference or face.
         *
         * @return  For a tensor ([marray](@ref MArray::marray)), the returned reference or view is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view (marray_view),
         *          the returned reference or view is mutable if the value type is not
         *          const-qualified.
         */
#ifdef MARRAY_DOXYGEN
        template <int Dim>
        possibly_mutable_reference_or_view
#else
        template <int Dim, int N=NDim>
        detail::enable_if_t<N!=1, marray_view<ctype, NDim-1>>
#endif
        front() const
        {
            return const_cast<marray_base&>(*this).front<Dim>();
        }

        /**
         * Return a mutable reference or view to the first element or face along the specified dimension.
         *
         * If the tensor has more than one dimension, a tensor face is returned.
         * Otherwise, a reference to an element is returned.
         *
         * @tparam Dim   The dimension along which to extract the reference or face.
         *
         * @return      A mutable reference or view.
         */
#ifdef MARRAY_DOXYGEN
        template <int Dim>
        mutable_reference_or_view
#else
        template <int Dim, int N=NDim>
        detail::enable_if_t<N!=1, marray_view<Type, NDim-1>>
#endif
        front()
        {
            return front<N>(Dim);
        }

        /**
         * Return an immutable reference or view to the first element or face along the specified dimension.
         *
         * @param dim   The dimension along which to extract the element or face.
         *
         * @return      An immutable reference or view.
         */
#ifdef MARRAY_DOXYGEN
        immutable_reference_or_view
#else
        template <int N=NDim>
        detail::enable_if_t<N!=1, marray_view<const Type, NDim-1>>
#endif
        cfront(int dim) const
        {
            return const_cast<marray_base&>(*this).front(dim);
        }

        /**
         * Return a reference or view to the first element or face along the specified dimension.
         *
         * @param dim   The dimension along which to extract the element or face.
         *
         * @return  For a tensor ([marray](@ref MArray::marray)), the returned reference or view is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view (marray_view),
         *          the returned reference or view is mutable if the value type is not
         *          const-qualified.
         */
#ifdef MARRAY_DOXYGEN
        possibly_mutable_reference_or_view
#else
        template <int N=NDim>
        detail::enable_if_t<N!=1, marray_view<ctype, NDim-1>>
#endif
        front(int dim) const
        {
            return const_cast<marray_base&>(*this).front(dim);
        }

        /**
         * Return a mutable reference or view to the first element or face along the specified dimension.
         *
         * @param dim   The dimension along which to extract the element or face.
         *
         * @return      A mutable reference or view.
         */
#ifdef MARRAY_DOXYGEN
        mutable_reference_or_view
#else
        template <int N=NDim>
        detail::enable_if_t<N!=1, marray_view<Type, NDim-1>>
#endif
        front(int dim)
        {
            MARRAY_ASSERT(dim >= 0 && dim < NDim);
            MARRAY_ASSERT(len_[dim] > 0);

            std::array<len_type, NDim-1> len;
            std::array<stride_type, NDim-1> stride;

            std::copy_n(len_.begin(), dim, len.begin());
            std::copy_n(len_.begin()+dim+1, NDim-dim-1, len.begin()+dim);
            std::copy_n(stride_.begin(), dim, stride.begin());
            std::copy_n(stride_.begin()+dim+1, NDim-dim-1, stride.begin()+dim);

            return {len, data_, stride};
        }

        /**
         * Return an immutable reference to the last element in this vector.
         *
         * This overload is only avaialble for vectors and vector views.
         *
         * @return  An immutable reference.
         */
#ifdef MARRAY_DOXYGEN
        immutable_reference
#else
        template <int Dim=0, int N=NDim>
        detail::enable_if_t<N==1, const_reference>
#endif
        cback() const
        {
            return const_cast<marray_base&>(*this).back<Dim>();
        }

        /**
         * Return a reference to the last element in this vector.
         *
         * This overload is only avaialble for vectors and vector views.
         *
         * @return  For a tensor ([marray](@ref MArray::marray)), the returned reference is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view (marray_view),
         *          the returned reference is mutable if the value type is not
         *          const-qualified.
         */
#ifdef MARRAY_DOXYGEN
        possibly_mutable_reference
#else
        template <int Dim=0, int N=NDim>
        detail::enable_if_t<N==1, cref>
#endif
        back() const
        {
            return const_cast<marray_base&>(*this).back<Dim>();
        }

        /**
         * Return a mutable reference to the last element in this vector.
         *
         * This overload is only avaialble for vectors and vector views.
         *
         * @return  A mutable reference.
         */
#ifdef MARRAY_DOXYGEN
        mutable_reference
#else
        template <int Dim=0, int N=NDim>
        detail::enable_if_t<N==1, reference>
#endif
        back()
        {
            static_assert(Dim == 0, "Dim out of range");
            MARRAY_ASSERT(len_[0] > 0);
            return data_[(len_[0]-1)*stride_[0]];
        }

        template <int N=NDim>
        detail::enable_if_t<N==1, const_reference>
        cback(int dim) const
        {
            return const_cast<marray_base&>(*this).back(dim);
        }

        template <int N=NDim>
        detail::enable_if_t<N==1, cref>
        back(int dim) const
        {
            return const_cast<marray_base&>(*this).back(dim);
        }

        template <int N=NDim>
        detail::enable_if_t<N==1, reference>
        back(int dim)
        {
            (void)dim;
            MARRAY_ASSERT(dim == 0);
            return back();
        }

        /**
         * Return an immutable reference or view to the last element or face along the specified dimension.
         *
         * @tparam Dim   The dimension along which to extract the element or face.
         *
         * @return      An immutable reference or view.
         */
#ifdef MARRAY_DOXYGEN
        template <int Dim>
        immutable_reference_or_view
#else
        template <int Dim, int N=NDim>
        detail::enable_if_t<N!=1, marray_view<const Type, NDim-1>>
#endif
        cback() const
        {
            return const_cast<marray_base&>(*this).back<Dim>();
        }

        /**
         * Return a reference or view to the last element or face along the specified dimension.
         *
         * @tparam Dim   The dimension along which to extract the element or face.
         *
         * @return  For a tensor ([marray](@ref MArray::marray)), the returned reference or view is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view (marray_view),
         *          the returned reference or view is mutable if the value type is not
         *          const-qualified.
         */
#ifdef MARRAY_DOXYGEN
        template <int Dim>
        possibly_mutable_reference_or_view
#else
        template <int Dim, int N=NDim>
        detail::enable_if_t<N!=1, marray_view<ctype, NDim-1>>
#endif
        back() const
        {
            return const_cast<marray_base&>(*this).back<Dim>();
        }

        /**
         * Return a mutable reference or view to the last element or face along the specified dimension.
         *
         * @tparam Dim   The dimension along which to extract the element or face.
         *
         * @return      A mutable reference or view.
         */
#ifdef MARRAY_DOXYGEN
        template <int Dim>
        possibly_mutable_reference_or_view
#else
        template <int Dim, int N=NDim>
        detail::enable_if_t<N!=1, marray_view<Type, NDim-1>>
#endif
        back()
        {
            return back<N>(Dim);
        }

        /**
         * Return an immutable reference or view to the last element or face along the specified dimension.
         *
         * @param dim   The dimension along which to extract the element or face.
         *
         * @return      An immutable reference or view.
         */
#ifdef MARRAY_DOXYGEN
        immutable_reference_or_view
#else
        template <int N=NDim>
        detail::enable_if_t<N!=1, marray_view<const Type, NDim-1>>
#endif
        cback(int dim) const
        {
            return const_cast<marray_base&>(*this).back(dim);
        }

        /**
         * Return a reference or view to the last element or face along the specified dimension.
         *
         * @param dim   The dimension along which to extract the element or face.
         *
         * @return  For a tensor ([marray](@ref MArray::marray)), the returned reference or view is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view (marray_view),
         *          the returned reference or view is mutable if the value type is not
         *          const-qualified.
         */
#ifdef MARRAY_DOXYGEN
        possibly_mutable_reference_or_view
#else
        template <int N=NDim>
        detail::enable_if_t<N!=1, marray_view<ctype, NDim-1>>
#endif
        back(int dim) const
        {
            return const_cast<marray_base&>(*this).back(dim);
        }

        /**
         * Return a mutable reference or view to the last element or face along the specified dimension.
         *
         * @param dim   The dimension along which to extract the element or face.
         *
         * @return      A mutable reference or view.
         */
#ifdef MARRAY_DOXYGEN
        mutable_reference_or_view
#else
        template <int N=NDim>
        detail::enable_if_t<N!=1, marray_view<Type, NDim-1>>
#endif
        back(int dim)
        {
            MARRAY_ASSERT(dim >= 0 && dim < NDim);
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

        template <int N=NDim>
        detail::enable_if_t<N==1, cref>
        operator[](len_type i) const
        {
            return const_cast<marray_base&>(*this)[i];
        }

        /**
         * Return a reference or subtensor.
         *
         * The overloaded [] operators may be applied multiple times in any
         * combination.
         *
         * This overload specifies a particular index along a dimension, and so
         * reduces the dimensionality of the resulting view by one. If specific
         * indices are given for all dimensions then the result is a reference to
         * the specified tensor element. If this overload is mixed with others
         * that specify ranges of indices, then the result is a subtensor view.
         *
         * For a tensor ([marray](@ref MArray::marray)),
         * the final view or reference is mutable if the instance is not const-qualified.
         * For a tensor view (marray_view), the final
         * view or reference is mutable if the value type is not const-qualified.
         *
         * @param i     The specified index. The dimension to which this index
         *              refers depends on how many [] operators have been applied.
         *              The first [] refers to the first dimension and so on.
         *
         * @return      If all indices have been explicitly specified, a reference
         *              to the indicated tensor element. Otherwise, a temporary
         *              indexing object which can be converted to a tensor view.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_reference_or_view
#else
        template <int N=NDim>
        detail::enable_if_t<N==1, reference>
#endif
        operator[](len_type i)
        {
            MARRAY_ASSERT(i < len_[0]);
            return data_[i*stride_[0]];
        }

        template <int N=NDim>
        detail::enable_if_t<N!=1, marray_slice<ctype, NDim, 1>>
        operator[](len_type i) const
        {
            MARRAY_ASSERT(i < len_[0]);
            return {*this, i};
        }

        template <int N=NDim>
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
            MARRAY_ASSERT_RANGE_IN(x, 0, len_[0]);
            return {*this, x};
        }

        /**
         * Return a subtensor.
         *
         * The overloaded [] operators may be applied multiple times in any
         * combination.
         *
         * This overload specifies a range of indices along a dimension, and so
         * always produces a subtensor view. If mixed with [] operators that specify
         * particular indices, then the dimensionality of the resulting view is reduced.
         * An index of `i` in the resulting view, along the current dimension, corresponds
         * to an index of `i+x.front()` in the original tensor or tensor view, and the length
         * of the resulting view along this dimension is `x.size()`.
         *
         * For a tensor ([marray](@ref MArray::marray)),
         * the final reference or view is mutable if the instance is not const-qualified.
         * For a tensor view (marray_view), the final
         * reference or view is mutable if the value type is not const-qualified.
         *
         * @param x     The specified range of indices. The dimension to which this range
         *              refers depends on how many [] operators have been applied.
         *              The first [] refers to the first dimension and so on. The specified
         *              range must not exceed the bounds of this tensor or tensor view.
         *
         * @return      A temporary indexing object which can be converted to a tensor view.
         */
        template <typename I>
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        marray_slice<Type, NDim, 1, slice_dim>
#endif
        operator[](const range_t<I>& x)
        {
            MARRAY_ASSERT_RANGE_IN(x, 0, len_[0]);
            return {*this, x};
        }

        marray_slice<ctype, NDim, 1, slice_dim>
        operator[](all_t) const
        {
            return {*this, range(len_[0])};
        }

        /**
         * Return a subtensor.
         *
         * The overloaded [] operators may be applied multiple times in any
         * combination.
         *
         * This overload is triggered using the special #slice::all value.
         *
         * This overload specifies all indices along a dimension, and so
         * always produces a subtensor view. If mixed with [] operators that specify
         * particular indices, then the dimensionality of the resulting view is reduced.
         * Indices in the resulting view, along the current dimension, are unchanged.
         *
         * For a tensor ([marray](@ref MArray::marray)),
         * the final reference or view is mutable if the instance is not const-qualified.
         * For a tensor view (marray_view), the final
         * reference or view is mutable if the value type is not const-qualified.
         *
         * @return      A temporary indexing object which can be converted to a tensor view.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        marray_slice<Type, NDim, 1, slice_dim>
#endif
        operator[](all_t)
        {
            return {*this, range(len_[0])};
        }

        marray_slice<ctype, NDim, 0, bcast_dim>
        operator[](bcast_t) const
        {
            return {*this, slice::bcast};
        }

        marray_slice<Type, NDim, 0, bcast_dim>
        operator[](bcast_t)
        {
            return {*this, slice::bcast};
        }

        cref operator()(detail::array_1d<len_type> idx) const
        {
            return const_cast<marray_base&>(*this)(idx);
        }

        /**
         * Return a reference to the indicated element.
         *
         * @param idx   The indices of the desired element. The number of indices must be
         *              equal to the number of dimensions. May be any
         *              one-dimensional container type whose elements are convertible
         *              to a tensor length, including initializer lists.
         *
         * @return  For a tensor ([marray](@ref MArray::marray)), the returned reference is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view (marray_view),
         *          the returned reference is mutable if the value type is not
         *          const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_reference
#else
        reference
#endif
        operator()(detail::array_1d<len_type> idx)
        {
            MARRAY_ASSERT(idx.size() == dimension());

            std::array<len_type,NDim> idx_;
            idx.slurp(idx_);

            auto ptr = data();
            for (auto i : range(NDim))
                ptr += idx_[i]*stride(i);

            return *ptr;
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

        /**
         * Return a reference to the indicated element or subtensor.
         *
         * This function is equivalent to calling the [] operator multiple times with
         * the given arguments.
         *
         * @param indices   An index or index range for each tensor dimension.
         *
         * @return  For a tensor ([marray](@ref MArray::marray)), the returned reference or view is
         *          mutable if the instance is not const-qualified.
         *          For a tensor view (marray_view),
         *          the returned reference or view is mutable if the value type is not
         *          const-qualified.
         */
#if MARRAY_DOXYGEN
        template <typename... Indices>
        possibly_mutable_reference_or_view
        operator(Indices&&... indices)
#else
        template <typename Arg, typename... Args, typename=
            detail::enable_if_t<sizeof...(Args) &&
                detail::are_indices_or_slices<Arg, Args...>::value>>
        auto operator()(Arg&& arg, Args&&... args) ->
        decltype((*this)[std::forward<Arg>(arg)](std::forward<Args>(args)...))
#endif
        {
            return (*this)[std::forward<Arg>(arg)](std::forward<Args>(args)...);
        }

        /***********************************************************************
         *
         * Iteration
         *
         **********************************************************************/

        /**
         * Iterate over the elements and call a function.
         *
         * @param f   A function or functor callable as either `f(e)` or
         *            `f(e, i0, i1, ...)` where `e` is a tensor element and
         *            `i0, i1, ...` are indices, one for each dimension.
         *            For a tensor ([marray](@ref MArray::marray)),
         *            the elements are mutable if the instance is not const-qualified.
         *            For a tensor view (marray_view),
         *            the elements are mutable if the value type is not
         *            const-qualified.
         */
        template <typename Func>
        void for_each_element(Func&& f) const
        {
            for_each_element<cptr>(std::forward<Func>(f),
                                   detail::static_range<int, NDim>{});
        }

        template <typename Func>
        void for_each_element(Func&& f)
        {
            for_each_element<pointer>(std::forward<Func>(f),
                                      detail::static_range<int, NDim>{});
        }

        /***********************************************************************
         *
         * Basic getters
         *
         **********************************************************************/

        /**
         * Return an immutable pointer to the tensor data.
         *
         * @return An immutable pointer that points to the element with
         *         all zero indices.
         */
#if MARRAY_DOXYGEN
        immutable_pointer
#else
        const_pointer
#endif
        cdata() const
        {
            return const_cast<marray_base&>(*this).data();
        }

        /**
         * Return a pointer to the tensor data.
         *
         * @return A pointer that points to the element with
         *         all zero indices. For a tensor ([marray](@ref MArray::marray)),
         *            the returned pointer is immutable if the instance is const-qualified.
         *            For a tensor view (marray_view),
         *            the returned pointer is immutable if the value type is
         *            const-qualified.
         */
#if MARRAY_DOXYGEN
        possible_mutable_pointer
#else
        cptr
#endif
        data() const
        {
            return const_cast<marray_base&>(*this).data();
        }

        /**
         * Return a mutable pointer to the tensor data.
         *
         * @return A mutable pointer that points to the element with
         *         all zero indices.
         */
#if MARRAY_DOXYGEN
        mutable_pointer
#else
        pointer
#endif
        data()
        {
            return data_;
        }

        /**
         * Return the length of the vector.
         *
         * This overload is only available for vectors and vector views.
         *
         * @return  The vector length, or number of elements.
         */
#if !MARRAY_DOXYGEN
        template <typename=void, int N=NDim, typename=detail::enable_if_t<N==1>>
#endif
        len_type length() const
        {
            return len_[0];
        }

        /**
         * Return the tensor length along the specified dimension.
         *
         * @tparam Dim  A dimension.
         *
         * @return      The length of the specified dimension.
         */
        template <int Dim>
        len_type length() const
        {
            static_assert(Dim >= 0 && Dim < NDim, "Dim out of range");
            return len_[Dim];
        }

        /**
         * Return the tensor length along the specified dimension.
         *
         * @param dim   A dimension.
         *
         * @return      The length of the specified dimension.
         */
        len_type length(int dim) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < NDim);
            return len_[dim];
        }

        /**
         * Return the tensor lengths.
         *
         * @return The lengths of the tensor; immutable.
         */
        const std::array<len_type, NDim>& lengths() const
        {
            return len_;
        }

        /**
         * Return the stride of the vector.
         *
         * This overload is only available for vectors and vector views.
         *
         * @return  The vector stride, or the distance between consecutive elements.
         */
#if !MARRAY_DOXYGEN
        template <typename=void, int N=NDim, typename=detail::enable_if_t<N==1>>
#endif
        stride_type stride() const
        {
            return stride_[0];
        }

        /**
         * Return the tensor stride along the specified dimension.
         *
         * @tparam Dim  A dimension.
         *
         * @return      The stride of the specified dimension.
         */
        template <int Dim>
        stride_type stride() const
        {
            static_assert(Dim >= 0 && Dim < NDim, "Dim out of range");
            return stride_[Dim];
        }

        /**
         * Return the tensor stride along the specified dimension.
         *
         * @param dim   A dimension.
         *
         * @return      The stride of the specified dimension.
         */
        stride_type stride(int dim) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < NDim);
            return stride_[dim];
        }

        /**
         * Return the tensor strides.
         *
         * @return The strides of the tensor; immutable.
         */
        const std::array<stride_type, NDim>& strides() const
        {
            return stride_;
        }

        /**
         * Return the number of dimensions.
         *
         * @return The number of dimensions.
         */
        static constexpr auto dimension()
        {
            return NDim;
        }

        friend std::ostream& operator<<(std::ostream& os, const marray_base& x)
        {
            for (auto i : range(NDim-1))
                os << std::string(i, ' ') << "{\n";

            std::array<len_type,NDim-1> idx = {};
            auto data = x.data_;

            for (bool done = false;!done;)
            {
                os << std::string(NDim-1, ' ') << '{';
                auto n = x.len_[NDim-1];
                if (n > 0)
                {
                    for (auto i : range(n-1))
                        os << data[i*x.stride_[NDim-1]] << ", ";
                    os << data[(n-1)*x.stride_[NDim-1]];
                }
                os << "}";

                for (auto i : reversed_range(NDim-1))
                {
                    idx[i]++;
                    data += x.stride_[i];

                    if (idx[i] >= x.len_[i])
                    {
                        data -= idx[i]*x.stride_[i];
                        idx[i] = 0;
                        os << "\n" << std::string(i, ' ') << '}';
                        if (i == 0) done = true;
                    }
                    else
                    {
                        os << ",\n";
                        for (auto j : range(i+1,NDim-1))
                            os << std::string(j, ' ') << "{\n";
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
