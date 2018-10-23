#ifndef _MARRAY_UTILITY_HPP_
#define _MARRAY_UTILITY_HPP_

#include <type_traits>
#include <array>
#include <vector>
#include <utility>
#include <iterator>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <tuple>
#include <cmath>
#include <cstddef>
#include <string>
#include <functional>
#include <ostream>

#ifdef MARRAY_ENABLE_ASSERTS
#define MARRAY_ASSERT(e) assert(e)
#else
#define MARRAY_ASSERT(e) ((void)0)
#endif

#include "short_vector.hpp"

namespace MArray
{

/*
 * The type all_t specifies a range [0,len_i) for an array
 * dimension i of length len_i (i.e. it selects all of the data along
 * that dimension).
 */
struct all_t { constexpr all_t() {} };
struct bcast_t { constexpr bcast_t() {} };
namespace slice
{
    constexpr all_t all;
    constexpr bcast_t bcast;
}

#ifndef MARRAY_LEN_TYPE
#define MARRAY_LEN_TYPE ptrdiff_t
#endif

typedef MARRAY_LEN_TYPE len_type;

#ifndef MARRAY_STRIDE_TYPE
#define MARRAY_STRIDE_TYPE ptrdiff_t
#endif

typedef MARRAY_STRIDE_TYPE stride_type;

#ifndef MARRAY_OPT_NDIM
#define MARRAY_OPT_NDIM 6
#endif

typedef short_vector<len_type,MARRAY_OPT_NDIM> len_vector;
typedef short_vector<stride_type,MARRAY_OPT_NDIM> stride_vector;
typedef short_vector<unsigned,MARRAY_OPT_NDIM> dim_vector;
typedef short_vector<len_type,MARRAY_OPT_NDIM> index_vector;
typedef short_vector<unsigned,MARRAY_OPT_NDIM> irrep_vector;
template <typename T>
using ptr_vector = short_vector<T*,MARRAY_OPT_NDIM>;

#ifndef MARRAY_DEFAULT_LAYOUT
#define MARRAY_DEFAULT_LAYOUT ROW_MAJOR
#endif

#define MARRAY_PASTE_(x,y) x##y
#define MARRAY_PASTE(x,y) MARRAY_PASTE_(x,y)

#define MARRAY_DEFAULT_DPD_LAYOUT_(type) \
    MARRAY_PASTE(MARRAY_PASTE(type,_),MARRAY_DEFAULT_LAYOUT)

#ifndef MARRAY_DEFAULT_DPD_LAYOUT
#define MARRAY_DEFAULT_DPD_LAYOUT PREFIX
#endif

/*
 * The special value uninitialized is used to construct an array which
 * does not default- or value-initialize its elements (useful for avoiding
 * redundant memory operations for scalar types).
 */
struct uninitialized_t { constexpr uninitialized_t() {} };
constexpr uninitialized_t uninitialized;

/*
 * Specifies the layout of the array data.
 */
struct layout
{
    int type;

    constexpr explicit layout(int type) : type(type) {}

    bool operator==(layout other) const { return type == other.type; }
    bool operator!=(layout other) const { return type != other.type; }
};

struct column_major_layout : layout { constexpr column_major_layout() : layout(0) {} };
constexpr column_major_layout COLUMN_MAJOR;

struct row_major_layout : layout { constexpr row_major_layout() : layout(1) {} };
constexpr row_major_layout ROW_MAJOR;

constexpr decltype(MARRAY_DEFAULT_LAYOUT) DEFAULT;

struct dpd_layout
{
    int type;

    constexpr explicit dpd_layout(int type) : type(type) {}

    dpd_layout(layout layout);

    layout base() const;

    bool operator==(dpd_layout other) const { return type == other.type; }
    bool operator!=(dpd_layout other) const { return type != other.type; }
};

struct balanced_column_major_layout : dpd_layout
{ constexpr balanced_column_major_layout() : dpd_layout(0) {} };
constexpr balanced_column_major_layout BALANCED_COLUMN_MAJOR;

struct balanced_row_major_layout : dpd_layout
{ constexpr balanced_row_major_layout() : dpd_layout(1) {} };
constexpr balanced_row_major_layout BALANCED_ROW_MAJOR;

struct blocked_column_major_layout : dpd_layout
{ constexpr blocked_column_major_layout() : dpd_layout(2) {} };
constexpr blocked_column_major_layout BLOCKED_COLUMN_MAJOR;

struct blocked_row_major_layout : dpd_layout
{ constexpr blocked_row_major_layout() : dpd_layout(3) {} };
constexpr blocked_row_major_layout BLOCKED_ROW_MAJOR;

struct prefix_column_major_layout : dpd_layout
{ constexpr prefix_column_major_layout() : dpd_layout(4) {} };
constexpr prefix_column_major_layout PREFIX_COLUMN_MAJOR;

struct prefix_row_major_layout : dpd_layout
{ constexpr prefix_row_major_layout() : dpd_layout(5) {} };
constexpr prefix_row_major_layout PREFIX_ROW_MAJOR;

constexpr decltype(MARRAY_DEFAULT_DPD_LAYOUT_(BALANCED)) BALANCED;
constexpr decltype(MARRAY_DEFAULT_DPD_LAYOUT_(BLOCKED)) BLOCKED;
constexpr decltype(MARRAY_DEFAULT_DPD_LAYOUT_(PREFIX)) PREFIX;

inline dpd_layout::dpd_layout(layout layout)
: type(layout == DEFAULT   ? MARRAY_DEFAULT_DPD_LAYOUT.type :
       layout == ROW_MAJOR ? MARRAY_PASTE(MARRAY_DEFAULT_DPD_LAYOUT,_ROW_MAJOR).type
                           : MARRAY_PASTE(MARRAY_DEFAULT_DPD_LAYOUT,_COLUMN_MAJOR).type) {}

inline layout dpd_layout::base() const
{
    if (*this == BALANCED_COLUMN_MAJOR ||
        *this == BLOCKED_COLUMN_MAJOR ||
        *this == PREFIX_COLUMN_MAJOR)
    {
        return COLUMN_MAJOR;
    }
    else
    {
        return ROW_MAJOR;
    }
}

template <typename I> class range_t;

namespace detail
{
    template <size_t N>
    std::array<unsigned, N> inverse_permutation(const std::array<unsigned, N>& p)
    {
        std::array<unsigned, N> ip;
        for (unsigned i = 0;i < N;i++) ip[p[i]] = i;
        return ip;
    }

    inline dim_vector inverse_permutation(const dim_vector& p)
    {
        dim_vector ip(p.size());
        for (unsigned i = 0;i < p.size();i++) ip[p[i]] = i;
        return ip;
    }

    template <typename...>
    struct exists {};

    template <typename T>
    using decay_t = typename std::decay<T>::type;

    template <typename T>
    using remove_cv_t = typename std::remove_cv<T>::type;

    template <bool Cond, typename T=void>
    using enable_if_t = typename std::enable_if<Cond,T>::type;

    template <bool Cond, typename T, typename U>
    using conditional_t = typename std::conditional<Cond,T,U>::type;

    template <typename T, typename U=void>
    using enable_if_integral_t = enable_if_t<std::is_integral<T>::value,U>;

    template <typename T, typename U=void>
    using enable_if_not_integral_t = enable_if_t<!std::is_integral<T>::value,U>;

    template <typename T, typename U=void>
    using enable_if_const_t = enable_if_t<std::is_const<T>::value,U>;

    template <typename T, typename U=void>
    using enable_if_not_const_t = enable_if_t<!std::is_const<T>::value,U>;

    template <typename T, typename U, typename V=void>
    using enable_if_assignable_t = enable_if_t<std::is_assignable<T,U>::value,V>;

    template <typename T, typename U, typename V=void>
    using enable_if_convertible_t = enable_if_t<std::is_convertible<T,U>::value,V>;

    template <typename T, typename... Args>
    struct are_convertible;

    template <typename T>
    struct are_convertible<T> : std::true_type {};

    template <typename T, typename Arg, typename... Args>
    struct are_convertible<T, Arg, Args...> :
        conditional_t<std::is_convertible<Arg, T>::value,
                      are_convertible<T, Args...>,
                      std::false_type> {};

    template <typename T, typename... Args>
    struct are_assignable;

    template <typename T>
    struct are_assignable<T> : std::true_type {};

    template <typename T, typename Arg, typename... Args>
    struct are_assignable<T, Arg, Args...> :
        conditional_t<std::is_assignable<T, Arg>::value,
                      are_assignable<T, Args...>,
                      std::false_type> {};

    template <typename T, typename=void>
    struct is_index_or_slice_helper : std::false_type {};

    template <typename T>
    struct is_index_or_slice_helper<T, enable_if_convertible_t<T, int>> : std::true_type {};

    template <typename I>
    struct is_index_or_slice_helper<range_t<I>, enable_if_integral_t<I>> : std::true_type {};

    template <>
    struct is_index_or_slice_helper<all_t> : std::true_type {};

    template <>
    struct is_index_or_slice_helper<bcast_t> : std::true_type {};

    template <typename T>
    struct is_index_or_slice : is_index_or_slice_helper<typename std::decay<T>::type> {};

    template <typename... Args>
    struct are_indices_or_slices;

    template<>
    struct are_indices_or_slices<> : std::true_type {};

    template <typename Arg, typename... Args>
    struct are_indices_or_slices<Arg, Args...> :
        conditional_t<is_index_or_slice<Arg>::value,
                      are_indices_or_slices<Args...>,
                      std::false_type> {};

    template <typename T, typename=void>
    struct is_container : std::false_type {};

    template <typename T>
    struct is_container<T,
        conditional_t<false,
                      exists<typename T::value_type,
                             decltype(std::declval<T>().size()),
                             decltype(std::declval<T>().begin()),
                             decltype(std::declval<T>().end())>,
                      void>>
    : std::true_type {};

    template <typename C, typename T, typename=void>
    struct is_container_of : std::false_type {};

    template <typename C, typename T>
    struct is_container_of<C, T, typename std::enable_if<is_container<C>::value>::type>
    : std::is_assignable<T&, typename C::value_type> {};

    template <typename C, typename T, typename U=void>
    using enable_if_container_of_t = typename std::enable_if<is_container_of<C, T>::value,U>::type;

    template <typename C, typename=void>
    struct is_container_of_containers : std::false_type {};

    template <typename C>
    struct is_container_of_containers<C, enable_if_t<is_container<C>::value>> :
        is_container<typename C::value_type> {};

    template <typename C, typename T, typename=void>
    struct is_container_of_containers_of : std::false_type {};

    template <typename C, typename T>
    struct is_container_of_containers_of<C, T, enable_if_t<is_container<C>::value>> :
        is_container_of<typename C::value_type, T> {};

    template <typename C, typename T, typename U=void>
    using enable_if_container_of_containers_of_t =
        typename std::enable_if<is_container_of_containers_of<C, T>::value,U>::type;

    template <typename T, typename... Ts>
    struct are_containers_helper;

    template <typename T>
    struct are_containers_helper<T> : is_container<T> {};

    template <typename T, typename... Ts>
    struct are_containers_helper
    : std::conditional<is_container<T>::value,
                       are_containers_helper<Ts...>,
                       std::false_type>::type {};

    template <typename... Ts>
    struct are_containers;

    template <>
    struct are_containers<> : std::true_type {};

    template <typename... Ts>
    struct are_containers : are_containers_helper<Ts...> {};

    template <typename T, typename C, typename... Cs>
    struct are_containers_of_helper;

    template <typename T, typename C>
    struct are_containers_of_helper<T, C> : is_container_of<C, T> {};

    template <typename T, typename C, typename... Cs>
    struct are_containers_of_helper
    : std::conditional<is_container_of<C, T>::value,
                       are_containers_of_helper<T, Cs...>,
                       std::false_type>::type {};

    template <typename T, typename... Cs>
    struct are_containers_of;

    template <typename T>
    struct are_containers_of<T> : std::true_type {};

    template <typename T, typename... Cs>
    struct are_containers_of : are_containers_of_helper<T, Cs...> {};

    template <typename Iterator>
    void inc_offsets_helper(unsigned i, Iterator) {}

    template <typename Iterator, typename Offset, typename... Offsets>
    void inc_offsets_helper(unsigned i, Iterator it, Offset& off0,
                            Offsets&... off)
    {
        off0 += (*it)[i];
        inc_offsets_helper(i, ++it, off...);
    }

    template <typename Strides, typename... Offsets>
    void inc_offsets(unsigned i, const Strides& strides, Offsets&... off)
    {
        inc_offsets_helper(i, strides.begin(), off...);
    }

    template <typename Pos, typename Iterator>
    void dec_offsets_helper(unsigned i, const Pos&, Iterator) {}

    template <typename Pos, typename Iterator, typename Offset, typename... Offsets>
    void dec_offsets_helper(unsigned i, const Pos& pos, Iterator it,
                             Offset& off0, Offsets&... off)
    {
        off0 -= pos[i]*(*it)[i];
        dec_offsets_helper(i, pos, ++it, off...);
    }

    template <typename Pos, typename Strides, typename... Offsets>
    void dec_offsets(unsigned i, const Pos& pos, const Strides& strides,
                     Offsets&... off)
    {
        dec_offsets_helper(i, pos, strides.begin(), off...);
    }

    template <typename Pos, typename Iterator>
    void move_offsets_helper(const Pos&, Iterator) {}

    template <typename Pos, typename Iterator, typename Offset, typename... Offsets>
    void move_offsets_helper(const Pos& pos, Iterator it,
                             Offset& off0, Offsets&... off)
    {
        for (unsigned i = 0;i < pos.size();i++) off0 += pos[i]*(*it)[i];
        move_offsets_helper(pos, ++it, off...);
    }

    template <typename Pos, typename Strides, typename... Offsets>
    void move_offsets(const Pos& pos, const Strides& strides,
                      Offsets&... off)
    {
        move_offsets_helper(pos, strides.begin(), off...);
    }

    template <typename Iterator>
    void set_strides_helper(Iterator) {}

    template <typename Iterator, typename Stride, typename... Strides>
    void set_strides_helper(Iterator it, const Stride& stride, const Strides&... strides)
    {
        std::copy(stride.begin(), stride.end(), it->begin());
        set_strides_helper(++it, strides...);
    }

    template <typename Strides_, typename... Strides>
    void set_strides(Strides_& strides_, const Strides&... strides)
    {
        set_strides_helper(strides_.begin(), strides...);
    }

    template <typename T, T... S> struct integer_sequence {};

    template <typename T, typename U, typename V> struct concat_sequences;
    template <typename T, T... S, T... R>
    struct concat_sequences<T, integer_sequence<T, S...>, integer_sequence<T, R...>>
    {
        typedef integer_sequence<T, S..., (R+sizeof...(S))...> type;
    };

    template <typename T, T N, typename=void> struct static_range_helper;

    template <typename T, T N> struct static_range_helper<T, N, enable_if_t<N==0>>
    {
        typedef integer_sequence<T> type;
    };

    template <typename T, T N> struct static_range_helper<T, N, enable_if_t<N==1>>
    {
        typedef integer_sequence<T, 0> type;
    };

    template <typename T, T N> struct static_range_helper<T, N, enable_if_t<(N>1)>>
    {
        typedef typename concat_sequences<T, typename static_range_helper<T, (N+1)/2>::type,
                                             typename static_range_helper<T, N/2>::type>::type type;
    };

    template <typename T, T N>
    using static_range = typename static_range_helper<T, N>::type;
}

}

#endif
