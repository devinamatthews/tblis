#ifndef MARRAY_MARRAY_SLICE_HPP
#define MARRAY_MARRAY_SLICE_HPP

#include <utility>

#include "range.hpp"
#include "marray_iterator.hpp"

#include "fwd/expression_fwd.hpp"
#include "fwd/marray_fwd.hpp"

namespace MArray
{

struct bcast_dim {};

struct slice_dim
{
    int dim;
    len_type len;
    len_type off;
    stride_type stride;

    slice_dim(int dim, len_type len, len_type off, stride_type stride)
    : dim(dim), len(len), off(off), stride(stride) {}
};

template <typename Type, int NDim, int NIndexed, typename... Dims>
class marray_slice
{
    template <typename, int, typename, bool> friend class marray_base;
    template <typename, int, typename> friend class marray;
    template <typename, int> friend class marray_view;
    template <typename, int, int, typename...> friend class marray_slice;

    public:
        typedef typename marray_view<Type, NDim>::value_type value_type;
        typedef typename marray_view<Type, NDim>::const_pointer const_pointer;
        typedef typename marray_view<Type, NDim>::pointer pointer;
        typedef typename marray_view<Type, NDim>::const_reference const_reference;
        typedef typename marray_view<Type, NDim>::reference reference;

    protected:
        pointer data_;
        const len_type* base_;
        const len_type* len_;
        const stride_type* stride_;
#ifdef MARRAY_ENABLE_ASSERTS
        const_pointer bbox_data_;
        const len_type* bbox_len_;
        const len_type* bbox_off_;
        const stride_type* bbox_stride_;
#endif
        std::tuple<Dims...> dims_;

        static constexpr int DimsLeft = NDim - NIndexed;
        static constexpr int CurDim = NIndexed-1;
        static constexpr int NextDim = NIndexed;
        static constexpr int NSliced = sizeof...(Dims);
        static constexpr int NewNDim = (... || std::is_same_v<Dims,bcast_dim>) ? 0 : NSliced + DimsLeft;

        marray_slice(const marray_slice& other) = default;

        template <typename Array>
        marray_slice(Array&& array, len_type i)
        : data_(array.data() + i*array.stride(CurDim)),
          base_(array.bases().data()),
          len_(array.lengths().data()),
          stride_(array.strides().data()),
#ifdef MARRAY_ENABLE_ASSERTS
          bbox_data_(array.bbox_data_),
          bbox_len_(array.bbox_len_.data()),
          bbox_off_(array.bbox_off_.data()),
          bbox_stride_(array.bbox_stride_.data()),
#endif
          dims_()
        {}

        template <typename Array, typename I>
        marray_slice(Array&& array, const range_t<I>& slice)
        : data_(array.data() + slice.front()*array.stride(CurDim)),
          base_(array.bases().data()),
          len_(array.lengths().data()),
          stride_(array.strides().data()),
#ifdef MARRAY_ENABLE_ASSERTS
          bbox_data_(array.bbox_data_),
          bbox_len_(array.bbox_len_.data()),
          bbox_off_(array.bbox_off_.data()),
          bbox_stride_(array.bbox_stride_.data()),
#endif
          dims_(slice_dim{CurDim,
                          (len_type)slice.size(),
                          (len_type)std::min(slice.front(), slice.back()),
                          (stride_type)slice.step()*array.stride(CurDim)})
        {}

        template <typename Array>
        marray_slice(Array&& array, bcast_t)
        : data_(array.data()),
          base_(array.bases().data()),
          len_(array.lengths().data()),
          stride_(array.strides().data()),
#ifdef MARRAY_ENABLE_ASSERTS
          bbox_data_(array.bbox_data_),
          bbox_len_(array.bbox_len_.data()),
          bbox_off_(array.bbox_off_.data()),
          bbox_stride_(array.bbox_stride_.data()),
#endif
          dims_(bcast_dim{})
        {}

        marray_slice(const marray_slice<Type, NDim, NIndexed-1, Dims...>& parent, len_type i)
        : data_(parent.data_ + i*parent.stride_[CurDim]),
          base_(parent.base_),
          len_(parent.len_),
          stride_(parent.stride_),
#ifdef MARRAY_ENABLE_ASSERTS
          bbox_data_(parent.bbox_data_),
          bbox_len_(parent.bbox_len_),
          bbox_off_(parent.bbox_off_),
          bbox_stride_(parent.bbox_stride_),
#endif
          dims_(parent.dims_)
        {}

        template <typename... OldDims, typename I>
        marray_slice(const marray_slice<Type, NDim, NIndexed-1, OldDims...>& parent, const range_t<I>& slice)
        : data_(parent.data_ + slice.front()*parent.stride_[CurDim]),
          base_(parent.base_),
          len_(parent.len_),
          stride_(parent.stride_),
#ifdef MARRAY_ENABLE_ASSERTS
          bbox_data_(parent.bbox_data_),
          bbox_len_(parent.bbox_len_),
          bbox_off_(parent.bbox_off_),
          bbox_stride_(parent.bbox_stride_),
#endif
          dims_(std::tuple_cat(parent.dims_,
                std::make_tuple(slice_dim{CurDim,
                                          (len_type)slice.size(),
                                          (len_type)std::min(slice.front(), slice.back()),
                                          (stride_type)slice.step()*parent.stride_[CurDim]})))
        {}

        template <typename... OldDims>
        marray_slice(const marray_slice<Type, NDim, NIndexed, OldDims...>& parent,
                     bcast_t)
        : data_(parent.data_),
          base_(parent.base_),
          len_(parent.len_),
          stride_(parent.stride_),
#ifdef MARRAY_ENABLE_ASSERTS
          bbox_data_(parent.bbox_data_),
          bbox_len_(parent.bbox_len_),
          bbox_off_(parent.bbox_off_),
          bbox_stride_(parent.bbox_stride_),
#endif
          dims_(std::tuple_cat(parent.dims_, std::make_tuple(bcast_dim{})))
        {}

        const marray_slice& operator()() const
        {
            return *this;
        }

        template <typename T, int N, size_t... I>
        marray_view<T,N> view_(std::index_sequence<I...>) const
        {
            static_assert(NewNDim, "Cannot create a view with broadcasted dimensons");
            static_assert(N == DYNAMIC || N == NewNDim);

            marray_view<T,N> ret;

            if constexpr (N == DYNAMIC)
            {
                ret.base_.resize(NewNDim);
                ret.len_.resize(NewNDim);
                ret.stride_.resize(NewNDim);
            }

            (
                (
                    ret.base_[I] = base(dim<I>().dim),
                    ret.len_[I] = dim<I>().len,
                    ret.stride_[I] = dim<I>().stride
                ),
                ...
            );

            std::copy_n(base_+NextDim, DimsLeft, ret.base_.begin()+NSliced);
            std::copy_n(len_+NextDim, DimsLeft, ret.len_.begin()+NSliced);
            std::copy_n(stride_+NextDim, DimsLeft, ret.stride_.begin()+NSliced);
            ret.data_ = data();

#ifdef MARRAY_ENABLE_ASSERTS

            if constexpr (N == DYNAMIC)
            {
                ret.bbox_len_.resize(NewNDim);
                ret.bbox_off_.resize(NewNDim);
                ret.bbox_stride_.resize(NewNDim);
            }

            (
                (
                    ret.bbox_len_[I] = bbox_len_[dim<I>().dim],
                    ret.bbox_stride_[I] = bbox_stride_[dim<I>().dim],
                    ret.bbox_off_[I] = bbox_off_[dim<I>().dim] +
                        dim<I>().off * std::abs(dim<I>().stride) / ret.bbox_stride_[I]
                ),
                ...
            );

            std::copy_n(bbox_len_+NextDim, DimsLeft, ret.bbox_len_.begin()+NSliced);
            std::copy_n(bbox_off_+NextDim, DimsLeft, ret.bbox_off_.begin()+NSliced);
            std::copy_n(bbox_stride_+NextDim, DimsLeft, ret.bbox_stride_.begin()+NSliced);
            ret.bbox_data_ = bbox_data_;

#endif

            return ret;
        }

    public:
        /**
         * Assign the partially-indexed portion of a tensor or tensor view to the result of an expression
         *
         * @param other The expression to use in the assignment.
         *
         * @returns *this
         */
#if !MARRAY_DOXYGEN
        template <typename Expression,
            typename=std::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
#endif
        const marray_slice& operator=(const Expression& other) const
        {
            assign_expr(*this, other);
            return *this;
        }

        /* Inherit docs */
        const marray_slice& operator=(const marray_slice& other) const
        {
            return operator=<>(other);
        }

        /**
         * Increment the partially-indexed portion of a tensor or tensor view by the result of an expression
         *
         * @param other The expression by which to increment.
         *
         * @returns *this
         */
#if !MARRAY_DOXYGEN
        template <typename Expression,
            typename=std::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
#endif
        const marray_slice& operator+=(const Expression& other) const
        {
            return *this = *this + other;
        }

        /**
         * Decrement the partially-indexed portion of a tensor or tensor view by the result of an expression
         *
         * @param other The expression by which to decrement.
         *
         * @returns *this
         */
#if !MARRAY_DOXYGEN
        template <typename Expression,
            typename=std::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
#endif
        const marray_slice& operator-=(const Expression& other) const
        {
            return *this = *this - other;
        }

        /**
         * Multiply the partially-indexed portion of a tensor or tensor view by the result of an expression
         *
         * @param other The expression by which to multiply.
         *
         * @returns *this
         */
#if !MARRAY_DOXYGEN
        template <typename Expression,
            typename=std::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
#endif
        const marray_slice& operator*=(const Expression& other) const
        {
            return *this = *this * other;
        }

        /**
         * Divide the partially-indexed portion of a tensor or tensor view by the result of an expression
         *
         * @param other The expression by which to divide.
         *
         * @returns *this
         */
#if !MARRAY_DOXYGEN
        template <typename Expression,
            typename=std::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
#endif
        const marray_slice& operator/=(const Expression& other) const
        {
            return *this = *this / other;
        }

        /**
         * Further index the tensor or tensor view.
         *
         * @see marray_base::operator[]()
         *
         * @param i An index, range, [all](@ref MArray::slice::all), or [bcast](@ref MArray::slice::bcast).
         *
         * @returns A partial indexing object.
         */
#if MARRAY_DOXYGEN
        tensor_view_or_reference operator[](index_or_slice i)
#else
        template <int N=DimsLeft>
        std::enable_if_t<N==1 && !sizeof...(Dims), reference>
        operator[](len_type i) const
#endif
        {
            i -= base(NextDim);
            MARRAY_ASSERT(i >= 0 && i < length(NextDim));
            return data_[i * stride(NextDim)];
        }

        /* Inherit docs */
        template <int N=DimsLeft>
        std::enable_if_t<N!=1 || sizeof...(Dims), marray_slice<Type, NDim, NIndexed+1, Dims...>>
        operator[](len_type i) const
        {
            static_assert(DimsLeft, "No more dimensions to index");
            i -= base(NextDim);
            MARRAY_ASSERT(i >= 0 && i < length(NextDim));
            return {*this, i};
        }

        /* Inherit docs */
        template <typename I>
        marray_slice<Type, NDim, NIndexed+1, Dims..., slice_dim>
        operator[](range_t<I> x) const
        {
            static_assert(DimsLeft, "No more dimensions to index");
            x -= base(NextDim);
            MARRAY_ASSERT_RANGE_IN(x, 0, length(NextDim));
            return {*this, x};
        }

        /* Inherit docs */
        marray_slice<Type, NDim, NIndexed+1, Dims..., slice_dim>
        operator[](all_t) const
        {
            static_assert(DimsLeft, "No more dimensions to index");
            return {*this, range(length(NIndexed))};
        }

        /* Inherit docs */
        marray_slice<Type, NDim, NIndexed, Dims..., bcast_dim>
        operator[](bcast_t) const
        {
            return {*this, slice::bcast};
        }

        /**
         * Further index the tensor or tensor view.
         *
         * @see marray_base::operator()()
         *
         * @param args One or more indices, ranges, [all](@ref MArray::slice::all), or
         *             [bcast](@ref MArray::slice::bcast).
         *
         * @returns A partial indexing object.
         */
#if MARRAY_DOXYGEN
        tensor_view_or_reference operator()(index_or_slice... args) const
#else
        template <typename Arg, typename... Args, typename=
            std::enable_if_t<detail::are_indices_or_slices<Arg, Args...>::value>>
        auto operator()(Arg&& arg, Args&&... args) const
#endif
        {
            return (*this)[std::forward<Arg>(arg)](std::forward<Args>(args)...);
        }

        const_pointer cdata() const
        {
            return data();
        }

        pointer data() const
        {
            return data_;
        }

        /**
         * Return a view of the partially-indexed tensor or tensor view.
         *
         * The resulting view or expression leaves any unindexed dimensions intact, i.e. it
         * is as if the remaining dimensions were indexed with `[`[all](@ref MArray::slice::all)`]`.
         *
         * @tparam N  If not specified, the returned view has a fixed number of dimensions equal to the number
         *            of sliced dimensions used to create this indexing object. If [DYNAMIC](@ref MArray::DYNAMIC)
         *            is specified, then the resulting view will have a variable number of dimensons. A specific
         *            numerical value should not be used.
         *
         * @return  An immutable tensor view.
         */
#if MARRAY_DOXYGEN
        template <int N>
        immutable_view
#else
        template <int N=NewNDim>
        marray_view<const Type, N>
#endif
        cview() const
        {
            return view_<const Type,N>(std::make_index_sequence<NSliced>{});
        }

        /**
         * Return a view of the partially-indexed tensor or tensor view.
         *
         * The resulting view or expression leaves any unindexed dimensions intact, i.e. it
         * is as if the remaining dimensions were indexed with `[`[all](@ref MArray::slice::all)`]`.
         *
         * @tparam N  If not specified, the returned view has a fixed number of dimensions equal to the number
         *            of sliced dimensions used to create this indexing object. If [DYNAMIC](@ref MArray::DYNAMIC)
         *            is specified, then the resulting view will have a variable number of dimensons. A specific
         *            numerical value should not be used.
         *
         * @return  An possibly-mutable tensor view. The resulting view is mutable if the value type is not
         *          const-qualified.
         */
#if MARRAY_DOXYGEN
         template <int N>
        possibly_mutable_view
#else
        template <int N=NewNDim>
        marray_view<Type, N>
#endif
        view() const
        {
            return view_<Type,N>(std::make_index_sequence<NSliced>{});
        }

        /**
         * Return a transposed view.
         *
         * This overload is only available for objects which would result in matrix views.
         *
         * @return      A possibly-mutable tensor view. The returned view is mutable if the value type is not
         *              const-qualified.
         */
#if MARRAY_DOXYGEN
        possibly_mutable_view
#else
        template <typename=void, int N=NewNDim, typename=std::enable_if_t<N==2>>
        marray_view<Type, 2>
#endif
        T() const
        {
            return view().T();
        }

        /*
         * These should be protected, but can't since some free helper functions in expression.hpp need to use them.
         */

        template <int Dim>
        auto dim() const -> decltype((std::get<Dim>(dims_)))
        {
            return std::get<Dim>(dims_);
        }

        len_type base(int dim) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < NDim);
            return base_[dim];
        }

        len_type length(int dim) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < NDim);
            return len_[dim];
        }

        stride_type stride(int dim) const
        {
            MARRAY_ASSERT(dim >= 0 && dim < NDim);
            return stride_[dim];
        }
};

}

#endif //MARRAY_MARRAY_SLICE_HPP
