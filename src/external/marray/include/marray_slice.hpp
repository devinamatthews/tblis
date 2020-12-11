#ifndef _MARRAY_MARRAY_SLICE_HPP_
#define _MARRAY_MARRAY_SLICE_HPP_

#include "marray_base.hpp"

namespace MArray
{

namespace detail
{

template <typename It1, typename It2>
void get_slice_dims_helper(It1, It2) {}

template <typename It1, typename It2, typename... Dims>
void get_slice_dims_helper(It1 len, It2 stride,
                           const slice_dim& dim, const Dims&... dims)
{
    *len = dim.len;
    *stride = dim.stride;
    get_slice_dims_helper(++len, ++stride, dims...);
}

template <typename It1, typename It2, typename... Dims>
void get_slice_dims_helper(It1 len, It2 stride,
                           const bcast_dim& dim, const Dims&... dims)
{
    *len = dim.len;
    *stride = 0;
    get_slice_dims_helper(++len, ++stride, dims...);
}

template <typename It1, typename It2, typename... Dims, size_t... I>
void get_slice_dims_helper(It1 len, It2 stride,
                           const std::tuple<Dims...>& dims,
                           detail::integer_sequence<size_t, I...>)
{
    get_slice_dims_helper(len, stride, std::get<I>(dims)...);
}

template <typename It1, typename It2, typename... Dims>
void get_slice_dims(It1 len, It2 stride,
                    const std::tuple<Dims...>& dims)
{
    get_slice_dims_helper(len, stride, dims,
                          detail::static_range<size_t, sizeof...(Dims)>());
}

}

/*
 * Represents a part of an array, where the first NIndexed-1 out of NDim
 * Dimensions have either been indexed into (i.e. a single value
 * specified for that index) or sliced (i.e. a range of values specified).
 * The parameter NSliced specifies how many indices were sliced. The
 * reference may be converted into an array view (of Dimension
 * NDim-NIndexed+1+NSliced) or further indexed, but may not be used to modify
 * data.
 */
template <typename Type, unsigned NDim, unsigned NIndexed, typename... Dims>
class marray_slice
{
    template <typename, unsigned, unsigned, typename...> friend class marray_slice;

    public:
        typedef typename marray_view<Type, NDim>::value_type value_type;
        typedef typename marray_view<Type, NDim>::const_pointer const_pointer;
        typedef typename marray_view<Type, NDim>::pointer pointer;
        typedef typename marray_view<Type, NDim>::const_reference const_reference;
        typedef typename marray_view<Type, NDim>::reference reference;
        typedef detail::marray_iterator<marray_slice> iterator;
        typedef detail::marray_iterator<const marray_slice> const_iterator;
        typedef std::reverse_iterator<iterator> reverse_iterator;
        typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

    protected:
        const std::array<len_type, NDim>& len_;
        const std::array<stride_type, NDim>& stride_;
        pointer data_;
        std::tuple<Dims...> dims_;

        static constexpr unsigned DimsLeft = NDim - NIndexed;
        static constexpr unsigned CurDim = NIndexed-1;
        static constexpr unsigned NextDim = NIndexed;
        static constexpr unsigned NSliced = sizeof...(Dims);
        static constexpr unsigned NewNDim = NSliced + DimsLeft;

    public:
        marray_slice(const marray_slice& other) = default;

        template <typename Array, typename=decltype(std::declval<Array>().lengths())>
        marray_slice(Array&& array, len_type i)
        : len_(array.lengths()), stride_(array.strides()),
          data_(array.data() + i*stride_[CurDim]) {}

        template <typename Array, typename I, typename=decltype(std::declval<Array>().lengths())>
        marray_slice(Array&& array, const range_t<I>& slice)
        : len_(array.lengths()), stride_(array.strides()),
          data_(array.data() + slice.front()*stride_[CurDim]),
          dims_(slice_dim{slice.size(), slice.step()*stride_[CurDim]}) {}

        template <typename Array, typename=decltype(std::declval<Array>().lengths())>
        marray_slice(Array&& array, bcast_t, len_type len)
        : len_(array.lengths()), stride_(array.strides()),
          data_(array.data()), dims_(bcast_dim{len}) {}

        marray_slice(const marray_slice<Type, NDim, NIndexed-1, Dims...>& parent, len_type i)
        : len_(parent.len_), stride_(parent.stride_),
          data_(parent.data_ + i*parent.stride_[CurDim]), dims_(parent.dims_) {}

        template <typename... OldDims, typename I>
        marray_slice(const marray_slice<Type, NDim, NIndexed-1, OldDims...>& parent,
                     const range_t<I>& slice)
        : len_(parent.len_), stride_(parent.stride_),
          data_(parent.data_ + slice.front()*parent.stride_[CurDim]),
          dims_(std::tuple_cat(parent.dims_,
                std::make_tuple(slice_dim{slice.size(), slice.step()*stride_[CurDim]}))) {}

        template <typename... OldDims>
        marray_slice(const marray_slice<Type, NDim, NIndexed, OldDims...>& parent,
                     bcast_t, len_type len)
        : len_(parent.len_), stride_(parent.stride_),
          data_(parent.data_),
          dims_(std::tuple_cat(parent.dims_, std::make_tuple(bcast_dim{len}))) {}

        const marray_slice& operator=(const marray_slice& other) const
        {
            assign_expr(*this, other);
            return *this;
        }

        template <typename Expression,
            typename=detail::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
        const marray_slice& operator=(const Expression& other) const
        {
            assign_expr(*this, other);
            return *this;
        }

        template <typename Expression,
            typename=detail::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
        const marray_slice& operator+=(const Expression& other) const
        {
            *this = *this + other;
            return *this;
        }

        template <typename Expression,
            typename=detail::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
        const marray_slice& operator-=(const Expression& other) const
        {
            *this = *this - other;
            return *this;
        }

        template <typename Expression,
            typename=detail::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
        const marray_slice& operator*=(const Expression& other) const
        {
            *this = *this * other;
            return *this;
        }

        template <typename Expression,
            typename=detail::enable_if_t<is_expression_arg_or_scalar<Expression>::value>>
        const marray_slice& operator/=(const Expression& other) const
        {
            *this = *this / other;
            return *this;
        }

        template <int N=DimsLeft>
        detail::enable_if_t<N==1 && !sizeof...(Dims), reference>
        operator[](len_type i) const
        {
            MARRAY_ASSERT(i >= 0 && i < len_[NDim-1]);
            return data_[i*stride_[NextDim]];
        }

        template <int N=DimsLeft>
        detail::enable_if_t<N!=1 || sizeof...(Dims),
                            marray_slice<Type, NDim, NIndexed+1, Dims...>>
        operator[](len_type i) const
        {
            static_assert(DimsLeft, "No more dimensions to index");
            MARRAY_ASSERT(i >= 0 && i < len_[NextDim]);
            return {*this, i};
        }

        template <typename I>
        marray_slice<Type, NDim, NIndexed+1, Dims..., slice_dim>
        operator[](const range_t<I>& x) const
        {
            static_assert(DimsLeft, "No more dimensions to index");
            MARRAY_ASSERT(x.front() >= 0);
            MARRAY_ASSERT(x.size() >= 0);
            MARRAY_ASSERT(x.front()+x.size() <= len_[NextDim]);
            return {*this, x};
        }

        marray_slice<Type, NDim, NIndexed+1, Dims..., slice_dim>
        operator[](all_t) const
        {
            static_assert(DimsLeft, "No more dimensions to index");
            return {*this, range(len_type(), len_[NIndexed])};
        }

        marray_slice<Type, NDim, NIndexed, Dims..., bcast_dim>
        operator[](bcast_t) const
        {
            return {*this, slice::bcast, len_[NIndexed]};
        }

        template <typename Arg, typename=
            detail::enable_if_t<detail::is_index_or_slice<Arg>::value>>
        auto operator()(Arg&& arg) const ->
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

        friend std::ostream& operator<<(std::ostream& os, const marray_slice& x)
        {
            return os << x.view();
        }

        const_pointer cdata() const
        {
            return data_;
        }

        pointer data() const
        {
            return data_;
        }

        template <unsigned Dim>
        auto dim() const -> decltype((std::get<Dim>(dims_)))
        {
            return std::get<Dim>(dims_);
        }

        len_type base_length(unsigned dim) const
        {
            MARRAY_ASSERT(dim < NDim);
            return len_[dim];
        }

        template <unsigned Dim>
        len_type base_length() const
        {
            static_assert(Dim < NDim, "Dim out of range");
            return len_[Dim];
        }

        stride_type base_stride(unsigned dim) const
        {
            MARRAY_ASSERT(dim < NDim);
            return stride_[dim];
        }

        template <unsigned Dim>
        stride_type base_stride() const
        {
            static_assert(Dim < NDim, "Dim out of range");
            return stride_[Dim];
        }

        marray_view<const Type, NewNDim> cview() const
        {
            return view();
        }

        marray_view<Type, NewNDim> view() const
        {
            std::array<len_type, NewNDim> len;
            std::array<stride_type, NewNDim> stride;

            detail::get_slice_dims(len.begin(), stride.begin(), dims_);
            std::copy_n(len_.begin()+NextDim, DimsLeft, len.begin()+NSliced);
            std::copy_n(stride_.begin()+NextDim, DimsLeft, stride.begin()+NSliced);

            return {len, data_, stride};
        }

        friend marray_view<const Type, NewNDim> cview(const marray_slice& x)
        {
            return x.cview();
        }

        friend marray_view<Type, NewNDim> view(const marray_slice& x)
        {
            return x.view();
        }

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
            return const_iterator{*this, len_[NextDim]};
        }

        const_iterator end() const
        {
            return const_iterator{*this, len_[NextDim]};
        }

        iterator end()
        {
            return iterator{*this, len_[NextDim]};
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
};

}

#endif
