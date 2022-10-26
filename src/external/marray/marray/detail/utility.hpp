#ifndef MARRAY_UTILITY_HPP
#define MARRAY_UTILITY_HPP

#include <cassert>
#include <cstdint>
#include <array>
#include <type_traits>

#ifndef MARRAY_ASSERT
#ifdef MARRAY_ENABLE_ASSERTS
#define MARRAY_ASSERT(e) assert(e)
#else
#define MARRAY_ASSERT(e) ((void)0)
#endif
#endif

#define MARRAY_LIKELY(x) __builtin_expect((x),1)
#define MARRAY_UNLIKELY(x) __builtin_expect((x),0)

#include "../types.hpp"
#include "../fwd/marray_fwd.hpp"

namespace MArray
{

struct all_t;
struct bcast_t;
template <typename I> class range_t;

namespace detail
{

template <typename T, typename U>
T ipow(T a, U b)
{
    T ab = 1;
    while (b --> 0) ab *= a;
    return ab;
}

/**
 * Return q = x/y and r = x%y assuming x >= 0 && y >= 0.
 *
 * @param x         The dividend.
 *
 * @param y         The divisor.
 *
 * @param q         The quotient.
 *
 * @param r         The remainder.
 *
 * @ingroup util
 */
template <typename T>
void divide(T x, T y, T& q, T& r)
{
    using U = std::make_unsigned_t<T>;
    if (sizeof(T) > 4 && MARRAY_LIKELY(((x | y) >> 32) == 0))
    {
        q = uint32_t(x)/uint32_t(y);
        r = uint32_t(x)%uint32_t(y);
    }
    else
    {
        q = U(x)/U(y);
        r = U(x)%U(y);
    }
}

template <typename T, size_t N>
std::array<T, N> inverse_permutation(const std::array<T, N>& p)
{
    std::array<T, N> ip;
    for (T i = T{};i < N;i++) ip[p[i]] = i;
    return ip;
}

template <typename T>
T inverse_permutation(const T& p)
{
    T ip(p.size());
    for (size_t i = 0;i < p.size();i++) ip[p[i]] = i;
    return ip;
}

template <typename...>
struct exists {};

template <typename T, typename... Args>
struct are_convertible;

template <typename T>
struct are_convertible<T> : std::true_type {};

template <typename T, typename Arg, typename... Args>
struct are_convertible<T, Arg, Args...> :
    std::conditional_t<std::is_convertible<Arg, T>::value,
                       are_convertible<T, Args...>,
                       std::false_type> {};

template <typename T, typename... Args>
struct are_assignable;

template <typename T>
struct are_assignable<T> : std::true_type {};

template <typename T, typename Arg, typename... Args>
struct are_assignable<T, Arg, Args...> :
    std::conditional_t<std::is_assignable_v<T, Arg>,
                       are_assignable<T, Args...>,
                       std::false_type> {};

template <typename T, typename=void>
struct is_index_or_slice_helper : std::false_type {};

template <typename T>
struct is_index_or_slice_helper<T, std::enable_if_t<std::is_convertible_v<T,len_type>>> : std::true_type {};

template <typename I>
struct is_index_or_slice_helper<range_t<I>> : std::true_type {};

template <>
struct is_index_or_slice_helper<all_t> : std::true_type {};

template <>
struct is_index_or_slice_helper<bcast_t> : std::true_type {};

template <typename T>
struct is_index_or_slice : is_index_or_slice_helper<std::decay_t<T>> {};

template <typename... Args>
struct are_indices_or_slices;

template<>
struct are_indices_or_slices<> : std::true_type {};

template <typename Arg, typename... Args>
struct are_indices_or_slices<Arg, Args...> :
    std::conditional_t<is_index_or_slice<Arg>::value,
                       are_indices_or_slices<Args...>,
                       std::false_type> {};

template <typename... Args>
struct count_dimensions;

template<>
struct count_dimensions<> : std::integral_constant<int,0> {};

template <typename Arg, typename... Args>
struct count_dimensions<Arg, Args...> :
    std::integral_constant<int,is_index_or_slice<Arg>::value -
                               std::is_same_v<std::decay_t<Arg>,bcast_t> +
                               count_dimensions<Args...>::value> {};

template <typename T, typename=void>
struct is_container : std::false_type {};

template <typename T>
struct is_container<T,
    std::conditional_t<false,
        exists<typename T::value_type,
               decltype(std::declval<T>().size()),
               decltype(std::declval<T>().begin()),
               decltype(std::declval<T>().end())>,
        void>>
: std::true_type {};

template <typename C>
struct container_size : std::integral_constant<int, DYNAMIC> {};

template <typename T, size_t N>
struct container_size<std::array<T,N>> : std::integral_constant<int, N> {};

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
struct is_container_of_containers<C, std::enable_if_t<is_container<C>::value>> :
    is_container<typename C::value_type> {};

template <typename C, typename T, typename=void>
struct is_container_of_containers_of : std::false_type {};

template <typename C, typename T>
struct is_container_of_containers_of<C, T, std::enable_if_t<is_container<C>::value>> :
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
void inc_offsets_helper(int, Iterator) {}

template <typename Iterator, typename Offset, typename... Offsets>
void inc_offsets_helper(int i, Iterator it, Offset& off0,
                        Offsets&... off)
{
    off0 += (*it)[i];
    inc_offsets_helper(i, ++it, off...);
}

template <typename Strides, typename... Offsets>
void inc_offsets(int i, const Strides& strides, Offsets&... off)
{
    inc_offsets_helper(i, strides.begin(), off...);
}

template <typename Pos, typename Iterator>
void dec_offsets_helper(int, const Pos&, Iterator) {}

template <typename Pos, typename Iterator, typename Offset, typename... Offsets>
void dec_offsets_helper(int i, const Pos& pos, Iterator it,
                         Offset& off0, Offsets&... off)
{
    off0 -= pos[i]*(*it)[i];
    dec_offsets_helper(i, pos, ++it, off...);
}

template <typename Pos, typename Strides, typename... Offsets>
void dec_offsets(int i, const Pos& pos, const Strides& strides,
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
    for (size_t i = 0;i < pos.size();i++) off0 += pos[i]*(*it)[i];
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

template <typename T>
struct integral_wrapper
{
    static_assert(std::is_integral_v<T>);

    T value;

    integral_wrapper(  signed     short value) : value(value) {}
    integral_wrapper(unsigned     short value) : value(value) {}
    integral_wrapper(  signed       int value) : value(value) {}
    integral_wrapper(unsigned       int value) : value(value) {}
    integral_wrapper(  signed      long value) : value(value) {}
    integral_wrapper(unsigned      long value) : value(value) {}
    integral_wrapper(  signed long long value) : value(value) {}
    integral_wrapper(unsigned long long value) : value(value) {}

    operator T() const { return value; }
};

using len_type_wrapper = integral_wrapper<len_type>;

template <typename T, typename=void>
struct wrap_if_integral_helper
{
    using type = T;
};

template <typename T>
struct wrap_if_integral_helper<T, std::enable_if_t<std::is_integral_v<T>>>
{
    using type = integral_wrapper<T>;
};

template <typename T>
using wrap_if_integral = typename wrap_if_integral_helper<T>::type;

typedef std::initializer_list<len_type_wrapper> len_type_init;

template <typename Type, int NDim>
struct initializer_type;

template <typename Type>
struct initializer_type<Type, 0>
{
    typedef Type type;
};

template <typename Type>
struct initializer_type<Type, DYNAMIC>
{
    struct private_type {};
    typedef private_type type;
};

template <typename Type, int NDim>
struct initializer_type
{
    typedef std::initializer_list<
        typename initializer_type<Type, NDim-1>::type> type;
};

template <typename Func, typename Arg, typename... Args>
std::enable_if_t<sizeof...(Args) &&
                    std::is_same<decltype(std::declval<Func&&>()(
                        std::declval<Arg&&>(),
                        std::declval<Args&&>()...)),void>::value>
call(Func&& f, Arg&& arg, Args&&... args)
{
    f(std::forward<Arg>(arg), std::forward<Args>(args)...);
}

template <typename Func, typename Arg, typename... Args>
std::enable_if_t<std::is_same<decltype(std::declval<Func&&>()(
                        std::declval<Arg&&>())),void>::value>
call(Func&& f, Arg&& arg, Args&&...)
{
    f(std::forward<Arg>(arg));
}

template <typename T, size_t N>
struct array : std::array<T, N>
{
    array() : std::array<T, N>{} {}

    array(size_t) : std::array<T, N>{} {}

    array(const std::array<T, N>& other) : std::array<T, N>(other) {}

    array& operator=(const std::array<T, N>& other)
    {
        static_cast<std::array<T, N>&>(*this) = other;
        return *this;
    }

    void clear()
    {
        static_cast<std::array<T, N>&>(*this) = {};
    }
};

template <typename T, int NDim>
struct array_type
{
    typedef array<T,NDim> type;
};

template <typename T>
struct array_type<T, DYNAMIC>
{
    typedef short_vector<T,MARRAY_OPT_NDIM> type;
};

template <typename T, int NDim>
using array_type_t = typename array_type<T, NDim>::type;

template <typename T, size_t N, typename C,
    typename=std::enable_if_t<is_container_of<C,T>::value>>
void assign(array<T, N>& lhs, const C& rhs)
{
    std::copy_n(rhs.begin(), N, lhs.begin());
}

template <typename T, size_t M, typename C,
    typename=std::enable_if_t<is_container_of<C,T>::value>>
void assign(short_vector<T, M>& lhs, const C& rhs)
{
    lhs.assign(rhs.begin(), rhs.end());
}

template <int I, typename T, size_t N>
void assign_helper(std::array<T, N>&) {}

template <int I, typename T, size_t N, typename C, typename... Cs>
void assign_helper(std::array<T, N>& lhs, const C& rhs, const Cs&... remaining)
{
    assign(lhs[I], rhs);
    assign_helper<I+1>(lhs, remaining...);
}

template <typename T, size_t N, typename... C,
    typename=std::enable_if_t<is_container<T>::value &&
                              are_containers_of<typename T::value_type,C...>::value &&
                              sizeof...(C) == N>>
void assign(std::array<T, N>& lhs, const C&... rhs)
{
    assign_helper<0>(lhs, rhs...);
}

}
}

#endif //MARRAY_UTILITY_HPP
