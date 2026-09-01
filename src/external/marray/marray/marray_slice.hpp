#ifndef MARRAY_MARRAY_SLICE_HPP
#define MARRAY_MARRAY_SLICE_HPP

#include <tuple>
#include <utility>

#include "marray_iterator.hpp"
#include "range.hpp"
#include "types.hpp"

#include "fwd/expression_fwd.hpp"
#include "fwd/marray_fwd.hpp"

MARRAY_BEGIN_NAMESPACE

struct bcast_dim
{
};

struct slice_dim
{
    int dim;
    len_type len;
    len_type off;
    stride_type stride;

    slice_dim(int dim, len_type len, len_type off, stride_type stride)
    : dim(dim),
      len(len),
      off(off),
      stride(stride)
    {
    }
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
#if MARRAY_DEBUG
    const_pointer bbox_data_;
    const len_type* bbox_len_;
    const len_type* bbox_off_;
    const stride_type* bbox_stride_;
#endif
    std::tuple<Dims...> dims_;

    static constexpr int DimsLeft = NDim - NIndexed;
    static constexpr int CurDim = NIndexed - 1;
    static constexpr int NextDim = NIndexed;
    static constexpr int NSliced = sizeof...(Dims);
    static constexpr int NewNDim =
        (... || std::is_same_v<Dims, bcast_dim>) ? 0 : NSliced + DimsLeft;

    marray_slice(const marray_slice& other) = default;

    template <typename Array>
    marray_slice(Array&& array, len_type i)
    : data_(array.data() + i * array.stride(CurDim)),
      base_(array.bases().data()),
      len_(array.lengths().data()),
      stride_(array.strides().data()),
#if MARRAY_DEBUG
      bbox_data_(array.bbox_data_),
      bbox_len_(array.bbox_len_.data()),
      bbox_off_(array.bbox_off_.data()),
      bbox_stride_(array.bbox_stride_.data()),
#endif
      dims_()
    {
    }

    template <typename Array, typename I>
    marray_slice(Array&& array, const range_t<I>& slice)
    : data_(array.data() + slice.front() * array.stride(CurDim)),
      base_(array.bases().data()),
      len_(array.lengths().data()),
      stride_(array.strides().data()),
#if MARRAY_DEBUG
      bbox_data_(array.bbox_data_),
      bbox_len_(array.bbox_len_.data()),
      bbox_off_(array.bbox_off_.data()),
      bbox_stride_(array.bbox_stride_.data()),
#endif
      dims_(slice_dim{CurDim,
                      (len_type)slice.size(),
                      (len_type)std::min(slice.front(), slice.back()),
                      (stride_type)slice.step() * array.stride(CurDim)})
    {
    }

    template <typename Array>
    marray_slice(Array&& array, bcast_t)
    : data_(array.data()),
      base_(array.bases().data()),
      len_(array.lengths().data()),
      stride_(array.strides().data()),
#if MARRAY_DEBUG
      bbox_data_(array.bbox_data_),
      bbox_len_(array.bbox_len_.data()),
      bbox_off_(array.bbox_off_.data()),
      bbox_stride_(array.bbox_stride_.data()),
#endif
      dims_(bcast_dim{})
    {
    }

    marray_slice(const marray_slice<Type, NDim, NIndexed - 1, Dims...>& parent,
                 len_type i)
    : data_(parent.data_ + i * parent.stride_[CurDim]),
      base_(parent.base_),
      len_(parent.len_),
      stride_(parent.stride_),
#if MARRAY_DEBUG
      bbox_data_(parent.bbox_data_),
      bbox_len_(parent.bbox_len_),
      bbox_off_(parent.bbox_off_),
      bbox_stride_(parent.bbox_stride_),
#endif
      dims_(parent.dims_)
    {
    }

    template <typename... OldDims, typename I>
    marray_slice(
        const marray_slice<Type, NDim, NIndexed - 1, OldDims...>& parent,
        const range_t<I>& slice)
    : data_(parent.data_ + slice.front() * parent.stride_[CurDim]),
      base_(parent.base_),
      len_(parent.len_),
      stride_(parent.stride_),
#if MARRAY_DEBUG
      bbox_data_(parent.bbox_data_),
      bbox_len_(parent.bbox_len_),
      bbox_off_(parent.bbox_off_),
      bbox_stride_(parent.bbox_stride_),
#endif
      dims_(std::tuple_cat(
          parent.dims_,
          std::make_tuple(
              slice_dim{CurDim,
                        (len_type)slice.size(),
                        (len_type)std::min(slice.front(), slice.back()),
                        (stride_type)slice.step() * parent.stride_[CurDim]})))
    {
    }

    template <typename... OldDims>
    marray_slice(const marray_slice<Type, NDim, NIndexed, OldDims...>& parent,
                 bcast_t)
    : data_(parent.data_),
      base_(parent.base_),
      len_(parent.len_),
      stride_(parent.stride_),
#if MARRAY_DEBUG
      bbox_data_(parent.bbox_data_),
      bbox_len_(parent.bbox_len_),
      bbox_off_(parent.bbox_off_),
      bbox_stride_(parent.bbox_stride_),
#endif
      dims_(std::tuple_cat(parent.dims_, std::make_tuple(bcast_dim{})))
    {
    }

    template <size_t... I>
    void bases_(len_type* bases, std::index_sequence<I...> = {}) const
    {
        ((bases[I] = base_[dim<I>().dim]), ...);

        std::copy_n(base_ + NextDim, DimsLeft, bases + NSliced);
    }

    template <size_t... I>
    void lengths_(len_type* len, std::index_sequence<I...> = {}) const
    {
        ((len[I] = dim<I>().len), ...);

        std::copy_n(len_ + NextDim, DimsLeft, len + NSliced);
    }

    template <size_t... I>
    void strides_(stride_type* stride, std::index_sequence<I...> = {}) const
    {
        ((stride[I] = dim<I>().stride), ...);

        std::copy_n(stride_ + NextDim, DimsLeft, stride + NSliced);
    }

    template <typename T, int N, size_t... I>
    marray_view<T, N> view_(std::index_sequence<I...>) const
    {
        static_assert(NewNDim,
                      "Cannot create a view with broadcasted dimensons");
        static_assert(N == DYNAMIC || N == NewNDim);

        marray_view<T, N> ret;

        if constexpr (N == DYNAMIC)
        {
            ret.base_.resize(NewNDim);
            ret.len_.resize(NewNDim);
            ret.stride_.resize(NewNDim);
        }

        bases_<I...>(ret.base_.data());
        lengths_<I...>(ret.len_.data());
        strides_<I...>(ret.stride_.data());
        ret.data_ = data();

#if MARRAY_DEBUG

        if constexpr (N == DYNAMIC)
        {
            ret.bbox_len_.resize(NewNDim);
            ret.bbox_off_.resize(NewNDim);
            ret.bbox_stride_.resize(NewNDim);
        }

        ((ret.bbox_len_[I] = bbox_len_[dim<I>().dim],
          ret.bbox_stride_[I] = bbox_stride_[dim<I>().dim],
          ret.bbox_off_[I] = bbox_off_[dim<I>().dim]
                           + dim<I>().off
                           * std::abs(dim<I>().stride)
                           / ret.bbox_stride_[I]),
         ...);

        std::copy_n(bbox_len_ + NextDim,
                    DimsLeft,
                    ret.bbox_len_.begin() + NSliced);
        std::copy_n(bbox_off_ + NextDim,
                    DimsLeft,
                    ret.bbox_off_.begin() + NSliced);
        std::copy_n(bbox_stride_ + NextDim,
                    DimsLeft,
                    ret.bbox_stride_.begin() + NSliced);
        ret.bbox_data_ = bbox_data_;

#endif

        return ret;
    }

  public:
    /**
     * Assign the partially-indexed portion of a tensor or tensor view to the
     * result of an expression
     *
     * @param other The expression to use in the assignment.
     *
     * @returns *this
     */
#if !MARRAY_DOXYGEN
    template <typename Expression,
              typename = std::enable_if_t<
                  is_expression_arg_or_scalar<Expression>::value>>
#endif
    const marray_slice& operator=(const Expression& other) const
    {
        assign_expr(view(), other);
        return *this;
    }

    /**
     * Set the tensor elements to those of the given tensor or tensor view.
     *
     * For a tensor ([marray](@ref MArray::marray)), the instance must not be
     * const-qualified. For a tensor view ([marray_view](@ref
     * MArray::marray_view)), the value type must not be const-qualified.
     *
     * @param other     A tensor, tensor view, or partially indexed tensor. The
     *                  dimensions of the tensor must match those of this
     *                  tensor.
     *
     * @return          *this
     */
#if MARRAY_DOXYGEN
    marray_slice& operator=(tensor_or_view other);
#else
    template <typename U, int N, typename D, bool O>
    std::enable_if_t<N == DYNAMIC, marray_slice&>
    operator=(const marray_base<U, N, D, O>& other)
    {
        view() = other;
        return *this;
    }

    /* Inherit docs */
    template <typename U, int N, typename D, bool O>
    std::enable_if_t<N == DYNAMIC, const marray_slice&>
    operator=(const marray_base<U, N, D, O>& other) const
    {
        view() = other;
        return *this;
    }

    /* Inherit docs */
    template <typename U, int N, int I, typename... D>
    std::enable_if_t<NewNDim == marray_slice<U, N, I, D...>::NewNDim,
                     marray_slice&>
    operator=(const marray_slice<U, N, I, D...>& other)
    {
        view() = other.view();
        return *this;
    }

    /* Inherit docs */
    template <typename U, int N, int I, typename... D>
    std::enable_if_t<NewNDim == marray_slice<U, N, I, D...>::NewNDim,
                     const marray_slice&>
    operator=(const marray_slice<U, N, I, D...>& other) const
    {
        view() = other.view();
        return *this;
    }

    /* Inherit docs */
    marray_slice& operator=(const marray_slice& other)
    {
        return operator= <>(other);
    }
#endif

    /**
     * Increment the partially-indexed portion of a tensor or tensor view by the
     * result of an expression
     *
     * @param other The expression by which to increment.
     *
     * @returns *this
     */
#if !MARRAY_DOXYGEN
    template <typename Expression,
              typename = std::enable_if_t<
                  is_expression_arg_or_scalar<Expression>::value>>
#endif
    const marray_slice& operator+=(const Expression& other) const
    {
        return *this = *this + other;
    }

    /**
     * Decrement the partially-indexed portion of a tensor or tensor view by the
     * result of an expression
     *
     * @param other The expression by which to decrement.
     *
     * @returns *this
     */
#if !MARRAY_DOXYGEN
    template <typename Expression,
              typename = std::enable_if_t<
                  is_expression_arg_or_scalar<Expression>::value>>
#endif
    const marray_slice& operator-=(const Expression& other) const
    {
        return *this = *this - other;
    }

    /**
     * Multiply the partially-indexed portion of a tensor or tensor view by the
     * result of an expression
     *
     * @param other The expression by which to multiply.
     *
     * @returns *this
     */
#if !MARRAY_DOXYGEN
    template <typename Expression,
              typename = std::enable_if_t<
                  is_expression_arg_or_scalar<Expression>::value>>
#endif
    const marray_slice& operator*=(const Expression& other) const
    {
        return *this = *this * other;
    }

    /**
     * Divide the partially-indexed portion of a tensor or tensor view by the
     * result of an expression
     *
     * @param other The expression by which to divide.
     *
     * @returns *this
     */
#if !MARRAY_DOXYGEN
    template <typename Expression,
              typename = std::enable_if_t<
                  is_expression_arg_or_scalar<Expression>::value>>
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
     * @param i An index, range, [all](@ref MArray::slice::all), or [bcast](@ref
     *          MArray::slice::bcast).
     *
     * @returns A partial indexing object.
     */
#if MARRAY_DOXYGEN
    tensor_view_or_reference operator[](index_or_slice i)
#else
    template <int N = DimsLeft>
    std::enable_if_t<N == 1 && !sizeof...(Dims), reference>
    operator[](len_type i) const
#endif
    {
        i -= base_[NextDim];
        MARRAY_ASSERT(i >= 0 && i < len_[NextDim]);
        return data_[i * stride_[NextDim]];
    }

    /* Inherit docs */
    template <int N = DimsLeft>
    std::enable_if_t<N == 1 && (NewNDim > 1), marray_view<Type, NewNDim - 1>>
    operator[](len_type i) const
    {
        return operator[]<-1>(i).view();
    }

    /* Inherit docs */
    template <int N = DimsLeft>
    std::enable_if_t<!(N == 1 && (NewNDim > 1 || !sizeof...(Dims))),
                     marray_slice<Type, NDim, NIndexed + 1, Dims...>>
    operator[](len_type i) const
    {
        static_assert(DimsLeft, "No more dimensions to index");
        i -= base_[NextDim];
        MARRAY_ASSERT(i >= 0 && i < len_[NextDim]);
        return {*this, i};
    }

    /* Inherit docs */
    template <typename I, int N = DimsLeft>
    std::enable_if_t<N == 1 && NewNDim, marray_view<Type, NewNDim>>
    operator[](range_t<I> x) const
    {
        return operator[]<I, -1>(x).view();
    }

    /* Inherit docs */
    template <typename I, int N = DimsLeft>
    std::enable_if_t<N != 1 || !NewNDim,
                     marray_slice<Type, NDim, NIndexed + 1, Dims..., slice_dim>>
    operator[](range_t<I> x) const
    {
        static_assert(DimsLeft, "No more dimensions to index");
        x -= base_[NextDim];
        MARRAY_ASSERT_RANGE_IN(x, 0, len_[NextDim]);
        return {*this, x};
    }

    /* Inherit docs */
    template <int N = DimsLeft>
    std::enable_if_t<N == 1 && NewNDim, marray_view<Type, NewNDim>>
    operator[](all_t) const
    {
        return operator[]<-1>(slice::all).view();
    }

    /* Inherit docs */
    template <int N = DimsLeft>
    std::enable_if_t<N != 1 || !NewNDim,
                     marray_slice<Type, NDim, NIndexed + 1, Dims..., slice_dim>>
    operator[](all_t) const
    {
        static_assert(DimsLeft, "No more dimensions to index");
        return {*this, range(len_[NextDim])};
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
     * @param args One or more indices, ranges, [all](@ref MArray::slice::all),
     *             or [bcast](@ref MArray::slice::bcast).
     *
     * @returns A partial indexing object.
     */
#if MARRAY_DOXYGEN
    tensor_view_or_reference operator()(index_or_slice... args) const
#else
    template <typename Arg, typename... Args>
    decltype(auto) operator()(const Arg& arg, const Args&... args) const
#endif
    {
        if constexpr (sizeof...(args) == 0)
            return (*this)[arg];
        else
            return (*this)[arg](args...);
    }

    const_pointer cdata() const { return data(); }

    pointer data() const { return data_; }

    /**
     * Return a view of the partially-indexed tensor or tensor view.
     *
     * The resulting view or expression leaves any unindexed dimensions intact,
     * i.e. it is as if the remaining dimensions were indexed with `[`[all](@ref
     * MArray::slice::all)`]`.
     *
     * @tparam N  If not specified, the returned view has a fixed number of
     * dimensions equal to the number of sliced dimensions used to create this
     * indexing object. If [DYNAMIC](@ref MArray::DYNAMIC) is specified, then
     * the resulting view will have a variable number of dimensons. A specific
     *            numerical value should not be used.
     *
     * @return  An immutable tensor view.
     */
#if MARRAY_DOXYGEN
    template <int N>
    immutable_view
#else
    template <int N = NewNDim>
    marray_view<const Type, N>
#endif
    cview() const
    {
        return view_<const Type, N>(std::make_index_sequence<NSliced>{});
    }

    /**
     * Return a view of the partially-indexed tensor or tensor view.
     *
     * The resulting view or expression leaves any unindexed dimensions intact,
     * i.e. it is as if the remaining dimensions were indexed with `[`[all](@ref
     * MArray::slice::all)`]`.
     *
     * @tparam N  If not specified, the returned view has a fixed number of
     * dimensions equal to the number of sliced dimensions used to create this
     * indexing object. If [DYNAMIC](@ref MArray::DYNAMIC) is specified, then
     * the resulting view will have a variable number of dimensons. A specific
     *            numerical value should not be used.
     *
     * @return  An possibly-mutable tensor view. The resulting view is mutable
     * if the value type is not const-qualified.
     */
#if MARRAY_DOXYGEN
    template <int N>
    possibly_mutable_view
#else
    template <int N = NewNDim>
    marray_view<Type, N>
#endif
    view() const
    {
        return view_<Type, N>(std::make_index_sequence<NSliced>{});
    }

    /**
     * Return a transposed view.
     *
     * This overload is only available for objects which would result in matrix
     * views.
     *
     * @return      A possibly-mutable tensor view. The returned view is mutable
     * if the value type is not const-qualified.
     */
#if MARRAY_DOXYGEN
    possibly_mutable_view
#else
    template <typename = void,
              int N = NewNDim,
              typename = std::enable_if_t<N == 2>>
    marray_view<Type, 2>
#endif
    T() const
    {
        return view().T();
    }

    template <int Dim> auto dim() const -> decltype((std::get<Dim>(dims_)))
    {
        return std::get<Dim>(dims_);
    }

    std::array<len_type, NewNDim> bases() const
    {
        std::array<len_type, NewNDim> base;
        bases_(base.data(), std::make_index_sequence<NSliced>{});
        return base;
    }

    std::array<len_type, NewNDim> lengths() const
    {
        std::array<len_type, NewNDim> len;
        lengths_(len.data(), std::make_index_sequence<NSliced>{});
        return len;
    }

    std::array<stride_type, NewNDim> strides() const
    {
        std::array<stride_type, NewNDim> stride;
        strides_(stride.data(), std::make_index_sequence<NSliced>{});
        return stride;
    }

    len_type base(int dim) const
    {
        MARRAY_ASSERT(dim >= 0 && dim < NewNDim);
        return bases()[dim];
    }

    len_type length(int dim) const
    {
        MARRAY_ASSERT(dim >= 0 && dim < NewNDim);
        return lengths()[dim];
    }

    stride_type stride(int dim) const
    {
        MARRAY_ASSERT(dim >= 0 && dim < NewNDim);
        return strides()[dim];
    }

    len_type base_raw(int dim) const
    {
        MARRAY_ASSERT(dim >= NIndexed && dim < NDim);
        return base_[dim];
    }

    len_type length_raw(int dim) const
    {
        MARRAY_ASSERT(dim >= NIndexed && dim < NDim);
        return len_[dim];
    }

    stride_type stride_raw(int dim) const
    {
        MARRAY_ASSERT(dim >= NIndexed && dim < NDim);
        return stride_[dim];
    }

    constexpr int dimension() const { return NewNDim; }
};

MARRAY_END_NAMESPACE

#endif // MARRAY_MARRAY_SLICE_HPP
