#ifndef _MARRAY_MARRAY_HPP_
#define _MARRAY_MARRAY_HPP_

#include "marray_view.hpp"

namespace MArray
{

template <typename Type, int NDim, typename Allocator>
class marray : public marray_base<Type, NDim, marray<Type, NDim, Allocator>, true>
{
    template <typename, int, typename, bool> friend class marray_base;
    template <typename, int> friend class marray_view;
    template <typename, int, typename> friend class marray;
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

        /**
         * Construct a tensor with all zero lengths, and which has no elements.
         */
        marray() {}

        /**
         * Copy constructor.
         *
         * @param other The tensor to copy from.
         */
        marray(const marray& other)
        {
            reset(other);
        }

        /**
         * Move constructor.
         *
         * @param other The tensor to move from. It is left in the same state as if #reset() was called.
         */
        marray(marray&& other)
        {
            reset(std::move(other));
        }

        /**
         * Copy constructor.
         *
         * @param other     The tensor, view, or partially-indexed tensor to copy.
         *
         * @param layout    The layout to use for the copied data, either #ROW_MAJOR
         *                  or #COLUMN_MAJOR. If not specified, and `other`
         *                  is a tensor (#marray), then inherit its layout. If not specified
         *                  and `other` is a view (#marray_view) or partially-indexed tensor,
         *                  then use the default layout.
         */
#if MARRAY_DOXYGEN
        marray(const tensor_or_view& other, layout layout = INHERIT_OR_DEFAULT)
#else
        template <typename U, int OldNDim, int NIndexed, typename... Dims,
            typename=detail::enable_if_assignable_t<reference,U>>
        marray(const marray_slice<U, OldNDim, NIndexed, Dims...>& other, layout layout = DEFAULT)
#endif
        {
            reset(other, layout);
        }

        /* Inherit docs */
        template <typename U, typename A,
            typename=detail::enable_if_assignable_t<reference,U>>
        marray(const marray<U, NDim, A>& other)
        {
            reset(other);
        }

        /* Inherit docs */
        template <typename U, typename D, bool O,
            typename=detail::enable_if_assignable_t<reference,U>>
        marray(const marray_base<U, NDim, D, O>& other, layout layout = DEFAULT)
        {
            reset(other, layout);
        }

        /**
         * Create a tensor of the specified shape.
         *
         * This constructor should be called using uniform initialization syntax:
         *
         * @code{.cpp}
         * marray<4> my_tensor{3, 9, 2, 10};
         * @endcode
         *
         * The default layout is used.
         *
         * @param len       An initializer list of lengths, one for each dimension.
         */
#if MARRAY_DOXYGEN
        explicit marray(std::initializer_list<len_type> len)
#else
        template <typename Type_=Type>
        explicit marray(detail::len_type_init len)
#endif
        {
            reset(len, Type(), DEFAULT);
        }

        /**
         * Create a tensor of the specified shape and with the given layout and fill value.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists, except when
         *              neither `val` or `layout` is given. In this case, use
         *              uniform initialization syntax.
         *
         * @param val       Initialize all elements to this value. If not specified,
         *                  use value-initialization.
         *
         * @param layout    The layout to use, either #ROW_MAJOR or #COLUMN_MAJOR.
         *                  If not specified, use the default layout.
         */
        explicit marray(detail::array_1d<len_type> len,
                        const Type& val=Type(), layout layout = DEFAULT)
        {
            reset(len, val, layout);
        }

        /**
         * Create a tensor of the specified shape and with the given layout.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param layout    The layout to use, either #ROW_MAJOR or #COLUMN_MAJOR.
         */
        explicit marray(detail::array_1d<len_type> len, layout layout)
        {
            reset(len, Type(), layout);
        }

        /**
         * Create a tensor of the specified shape and with the given layout and
         * without initializing tensor elements.
         *
         * This overload is triggered using the special #uninitialized value.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param layout    The layout to use, either #ROW_MAJOR or #COLUMN_MAJOR.
         *                  If not specified, use the
         */
        explicit marray(detail::array_1d<len_type> len, uninitialized_t,
                        layout layout = DEFAULT)
        {
            reset(len, uninitialized, layout);
        }

        /**
         * Create a tensor with the specified data.
         *
         * @param data   A nested initializer list containing the tensor data.
         *               The level of nesting must match the number of dimensions.
         *               The lengths of the dimensions are inferred from the
         *               initializer; the provided data must be "dense", i.e. there
         *               cannot be missing values.
         *
         * @param layout    The layout to use, either #ROW_MAJOR or #COLUMN_MAJOR.
         *                  If not specified, use the default layout.
         */
#if MARRAY_DOXYGEN
        marray(initializer data, layout layout = DEFAULT)
#else
        template <int NDim_=NDim>
        marray(initializer_type data, layout layout = DEFAULT, std::enable_if_t<(NDim_>1)>* = nullptr)
#endif
        {
            reset(data, layout);
        }

        /**
         * Create a tensor from the given expression.
         *
         * @param other   A mathematical expression of one or more tensors.
         *                The shape of the new tensor is deduced from the tensors
         *                involved in the expression, and so should be well-defined.
         *                This means, in particular, that an implicitly broadcasted
         *                expression (#slice::bcast) is not valid. The default
         *                layout is used.
         */
#if !MARRAY_DOXYGEN
        template <typename Expression,
            typename=detail::enable_if_t<is_expression<Expression>::value>>
#endif
        marray(const Expression& other)
        {
            typedef typename expression_type<detail::decay_t<Expression>>::type expr_type;

            static_assert(NDim == expr_dimension<expr_type>::value,
                          "Dimensionality of the expression must equal that of the target");

            reset(get_expr_lengths<NDim>(other), uninitialized);
            assign_expr(*this, other);
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

        marray& operator=(initializer_type other)
        {
            return base::operator=(other);
        }

#if !MARRAY_DOXYGEN
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
#endif

        /**
         * The number of elements in the tensor.
         *
         * Since the elements are arranged contiguously, this is also the size
         * of the underlying storage.
         *
         * @return The number of elements in the tensor.
         */
        stride_type size() const
        {
            return storage_.size;
        }

        /***********************************************************************
         *
         * Reset
         *
         **********************************************************************/

        /**
         * Reset the tensor to an empty state, with all lengths zero.
         */
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

        /**
         * Re-initialize the tensor by moving the data from another tensor.
         *
         * @param other The tensor from which to move. It is left in the state as if #reset() were called on it.
         */
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

        /**
         * Re-initialize the tensor by copying the data from another tensor or view.
         *
         * @param other     The tensor, view, or partially-indexed tensor from which to copy.
         *
         * @param layout    The layout to use for the copied data, either #ROW_MAJOR
         *                  or #COLUMN_MAJOR. If not specified, and `other`
         *                  is a tensor (#marray), then inherit its layout. If not specified
         *                  and `other` is a view (#marray_view) or partially-indexed tensor,
         *                  then use the default layout.
         */
#if MARRAY_DOXYGEN
        void reset(const tensor_or_view& other, layout layout = DEFAULT)
#else
        template <typename U, int OldNDim, int NIndexed, typename... Dims,
            typename=detail::enable_if_assignable_t<reference, U>>
        void reset(const marray_slice<U, OldNDim, NIndexed, Dims...>& other, layout layout = DEFAULT)
#endif
        {
            reset(other.view(), layout);
        }

        template <typename Type_=Type>
        void reset(detail::len_type_init len)
        {
            reset(len, Type(), DEFAULT);
        }

        /**
         * Reset the tensor to the specified shape and with the given layout and fill value.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param val       Initialize all elements to this value. If not specified,
         *                  use value-initialization.
         *
         * @param layout    The layout to use, either #ROW_MAJOR or #COLUMN_MAJOR.
         *                  If not specified, use the default layout.
         */
        void reset(const detail::array_1d<len_type>& len,
                   const Type& val=Type(), layout layout = DEFAULT)
        {
            reset(len, uninitialized, layout);
            std::uninitialized_fill_n(data_, storage_.size, val);
        }

        /**
         * Reset the tensor to the specified shape and with the given layout.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param layout    The layout to use, either #ROW_MAJOR or #COLUMN_MAJOR.
         *                  If not specified, use the default layout.
         */
        void reset(const detail::array_1d<len_type>& len, layout layout)
        {
            reset(len, Type(), layout);
        }

        /**
         * Reset the tensor to the specified shape and with the given layout
         * and without intializing tensor elements.
         *
         * This overload is triggered using the special #uninitialized value.
         *
         * @param len   The length of each dimension. May be any one-dimensional
         *              container type with elements convertible to tensor
         *              lengths, including initializer lists.
         *
         * @param layout    The layout to use, either #ROW_MAJOR or #COLUMN_MAJOR.
         *                  If not specified, use the default layout.
         */
        void reset(const detail::array_1d<len_type>& len, uninitialized_t,
                   layout layout = DEFAULT)
        {
            reset();

            layout_ = layout;
            storage_.size = size(len);
            base::reset(len, alloc_traits::allocate(storage_, storage_.size),
                        base::strides(len, layout));
        }

        /**
         * Reset a tensor from the specified data.
         *
         * @param data   A nested initializer list containing the tensor data.
         *               The level of nesting must match the number of dimensions.
         *               The lengths of the dimensions are inferred from the
         *               initializer; the provided data must be "dense", i.e. there
         *               cannot be missing values.
         *
         * @param layout    The layout to use, either #ROW_MAJOR or #COLUMN_MAJOR.
         *                  If not specified, use the default layout.
         */
#if MARRAY_DOXYGEN
        void
#else
        template <int NDim_=NDim>
        std::enable_if_t<(NDim_>1)>
#endif
        reset(initializer_type data, layout layout = DEFAULT)
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

        /**
         * Resize the tensor.
         *
         * After resizing, any elements whose indicies fall within the bounds of
         * the new shape are retained. Any new elements are initialized as requested.
         *
         * This function always reallocates and copies data, so any pointers or
         * references to elements are invalidated.
         *
         * @param len   The lengths of the dimensions are resizing.
         *
         * @param val   The value to use for initializing new element (those
         *              whose indices fall outside the bounds of the original
         *              shape). If not specified, value-initialization is used.
         */
        void resize(const detail::array_1d<len_type>& len,
                    const Type& val=Type())
        {
            std::array<len_type, NDim> new_len;
            len.slurp(new_len);

            if (new_len == len_) return;

            marray a(std::move(*this));
            reset(len, val, layout_);
            marray_view<Type, NDim> b(*this);

            /*
             * It is OK to change the geometry of 'a' even if it is not
             * a view since it is about to go out of scope.
             */
            for (auto i : range(NDim))
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

        /**
         * Append a new value to the end of the vector.
         *
         * This overload is only available for vectors.
         *
         * The length of the vector is increased by one.
         *
         * This function always reallocates and copies data, so any pointers or
         * references to elements are invalidated.
         *
         * @param x     The value to append.
         */
#if !MARRAY_DOXYGEN
        template <int Dim=0, int N=NDim, typename=detail::enable_if_t<N==1>>
#endif
        void push_back(const Type& x)
        {
            static_assert(Dim == 0, "Dim out of range");
            resize({len_[0]+1});
            back() = x;
        }

        template <int N=NDim, typename=detail::enable_if_t<N==1>>
        void push_back(int dim, const Type& x)
        {
            (void)dim;
            MARRAY_ASSERT(dim == 0);
            push_back(x);
        }

        /**
         * Append a new element or face to the end of the specified dimension.
         *
         * The length of the tensor in the specified dimension
         * is increased by one.
         *
         * This function always reallocates and copies data, so any pointers or
         * references to elements are invalidated.
         *
         * @tparam Dim  The dimension along which to append.
         *
         * @param x     The element (if this is a vector) or face to append.
         */
#if MARRAY_DOXYGEN
        template <int Dim>
        void push_back(const element_or_tensor_or_view& x)
#else
        template <int Dim, typename U, typename D, bool O, int N=NDim,
            typename=detail::enable_if_assignable_t<reference, U>>
        void push_back(const marray_base<U, NDim-1, D, O>& x)
#endif
        {
            push_back(Dim, x);
        }

        /**
         * Append a new element or face to the end of the specified dimension.
         *
         * The length of the tensor in the specified dimension
         * is increased by one.
         *
         * This function always reallocates and copies data, so any pointers or
         * references to elements are invalidated.
         *
         * @param dim   The dimension along which to append.
         *
         * @param x     The element (if this is a vector) or face to append.
         */
#if MARRAY_DOXYGEN
        void push_back(int dim, const element_or_tensor_or_view& x)
#else
        template <typename U, typename D, bool O, int N=NDim,
            typename=detail::enable_if_assignable_t<reference, U>>
        void push_back(int dim, const marray_base<U, NDim-1, D, O>& x)
#endif
        {
            MARRAY_ASSERT(dim >= 0 && dim < NDim);

            for (int i = 0, j = 0;i < NDim;i++)
            {
                (void)j;
                MARRAY_ASSERT(i == dim || len_[i] == x.length(j++));
            }

            std::array<len_type, NDim> len = len_;
            len[dim]++;
            resize(len);
            this->template back<NDim>(dim) = x;
        }

        /**
         * Remove a the element at the end of the vector.
         *
         * This overload is only available for vectors.
         *
         * The length of the vector is decresed by one.
         *
         * This function always reallocates and copies data, so any pointers or
         * references to elements are invalidated.
         */
#if MARRAY_DOXYGEN
        void
#else
        template <typename=void, int N=NDim>
        typename std::enable_if<N==1>::type
#endif
        pop_back()
        {
            resize({len_[0]-1});
        }

        /**
         * Remove the element or face at the end of the specified dimension.
         *
         * The length of the tensor in the specified dimension
         * is decreased by one.
         *
         * This function always reallocates and copies data, so any pointers or
         * references to elements are invalidated.
         *
         * @tparam Dim  The dimension along which to remove an element or face.
         */
#if MARRAY_DOXYGEN
        template <int Dim>
#else
        template <int Dim, int N=NDim>
#endif
        void pop_back()
        {
            pop_back(Dim);
        }

        /**
         * Remove the element or face at the end of the specified dimension.
         *
         * The length of the tensor in the specified dimension
         * is decreased by one.
         *
         * This function always reallocates and copies data, so any pointers or
         * references to elements are invalidated.
         *
         * @param dim   The dimension along which to remove an element or face.
         */
        void pop_back(int dim)
        {
            MARRAY_ASSERT(dim >= 0 && dim < NDim);
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

        /**
         * Swap the shape, data, and layout of this tensor with those of another.
         *
         * @param other     The tensor to swap with.
         */
        void swap(marray& other)
        {
            using std::swap;
            swap(storage_, other.storage_);
            swap(layout_, other.layout_);
            base::swap(other);
        }

        /**
         * Swap the shape, data, and layout of two tensors
         *
         * @param a     The first tensor to swap.
         *
         * @param b     The second tensor to swap.
         */
        friend void swap(marray& a, marray& b)
        {
            a.swap(b);
        }
};

}

#endif
