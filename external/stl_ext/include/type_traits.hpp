#ifndef _STL_EXT_TYPE_TRAITS_HPP_
#define _STL_EXT_TYPE_TRAITS_HPP_

#include <type_traits>

namespace stl_ext
{

using std::decay;
template <typename T>
using decay_t = typename decay<T>::type;

using std::remove_const;
template <typename T>
using remove_const_t = typename remove_const<T>::type;

using std::remove_volatile;
template <typename T>
using remove_volatile_t = typename remove_volatile<T>::type;

using std::remove_cv;
template <typename T>
using remove_cv_t = typename remove_cv<T>::type;

using std::conditional;
template <bool T, typename U, typename V>
using conditional_t = typename conditional<T,U,V>::type;

using std::enable_if;
template <bool T, typename U=void>
using enable_if_t = typename enable_if<T,U>::type;

template <typename T, typename U, typename=void>
struct common_type;
template <typename T, typename U>
struct common_type<T, U, enable_if_t<std::is_arithmetic<T>::value && std::is_arithmetic<U>::value>>
{
    typedef typename std::common_type<T,U>::type type;
};
template <typename T, typename U>
using common_type_t = typename common_type<T,U>::type;

using std::is_same;
template <typename T, typename U>
constexpr bool is_same_v() { return is_same<T,U>::value; }
template <typename T, typename U, typename V=void>
using enable_if_same = enable_if<is_same<T,U>::value,V>;
template <typename T, typename U, typename V=void>
using enable_if_same_t = typename enable_if_same<T,U,V>::type;
template <typename T, typename U, typename V=void>
using enable_if_not_same = enable_if<!is_same<T,U>::value,V>;
template <typename T, typename U, typename V=void>
using enable_if_not_same_t = typename enable_if_not_same<T,U,V>::type;

using std::is_const;
template <typename T>
constexpr bool is_const_v() { return is_const<T>::value; }
template <typename T, typename U=void>
using enable_if_const = enable_if<is_const<T>::value,U>;
template <typename T, typename U=void>
using enable_if_const_t = typename enable_if_const<T,U>::type;
template <typename T, typename U=void>
using enable_if_non_const = enable_if<!is_const<T>::value,U>;
template <typename T, typename U=void>
using enable_if_non_const_t = typename enable_if_non_const<T,U>::type;

using std::is_integral;
template <typename T>
constexpr bool is_integral_v() { return is_integral<T>::value; }
template <typename T, typename U=void>
using enable_if_integral = enable_if<is_integral<T>::value,U>;
template <typename T, typename U=void>
using enable_if_integral_t = typename enable_if_integral<T,U>::type;
template <typename T, typename U=void>
using enable_if_not_integral = enable_if<!is_integral<T>::value,U>;
template <typename T, typename U=void>
using enable_if_not_integral_t = typename enable_if_not_integral<T,U>::type;

using std::is_floating_point;
template <typename T>
constexpr bool is_floating_point_v() { return is_floating_point<T>::value; }
template <typename T, typename U=void>
using enable_if_floating_point = enable_if<is_floating_point<T>::value,U>;
template <typename T, typename U=void>
using enable_if_floating_point_t = typename enable_if_floating_point<T,U>::type;
template <typename T, typename U=void>
using enable_if_not_floating_point = enable_if<!is_floating_point<T>::value,U>;
template <typename T, typename U=void>
using enable_if_not_floating_point_t = typename enable_if_not_floating_point<T,U>::type;

using std::is_arithmetic;
template <typename T>
constexpr bool is_arithmetic_v() { return is_arithmetic<T>::value; }
template <typename T, typename U=void>
using enable_if_arithmetic = enable_if<is_arithmetic<T>::value,U>;
template <typename T, typename U=void>
using enable_if_arithmetic_t = typename enable_if_arithmetic<T,U>::type;
template <typename T, typename U=void>
using enable_if_not_arithmetic = enable_if<!is_arithmetic<T>::value,U>;
template <typename T, typename U=void>
using enable_if_not_arithmetic_t = typename enable_if_not_arithmetic<T,U>::type;

using std::is_pointer;
template <typename T>
constexpr bool is_pointer_v() { return is_pointer<T>::value; }
template <typename T, typename U=void>
using enable_if_pointer = enable_if<is_pointer<T>::value,U>;
template <typename T, typename U=void>
using enable_if_pointer_t = typename enable_if_pointer<T,U>::type;
template <typename T, typename U=void>
using enable_if_not_pointer = enable_if<!is_pointer<T>::value,U>;
template <typename T, typename U=void>
using enable_if_not_pointer_t = typename enable_if_not_pointer<T,U>::type;

using std::add_pointer;
template <typename T>
using add_pointer_t = typename add_pointer<T>::type;

using std::remove_pointer;
template <typename T>
using remove_pointer_t = typename remove_pointer<T>::type;

using std::is_reference;
template <typename T>
constexpr bool is_reference_v() { return is_reference<T>::value; }
template <typename T, typename U=void>
using enable_if_reference = enable_if<is_reference<T>::value,U>;
template <typename T, typename U=void>
using enable_if_reference_t = typename enable_if_reference<T,U>::type;
template <typename T, typename U=void>
using enable_if_not_reference = enable_if<!is_reference<T>::value,U>;
template <typename T, typename U=void>
using enable_if_not_reference_t = typename enable_if_not_reference<T,U>::type;

using std::add_lvalue_reference;
template <typename T>
using add_lvalue_reference_t = typename add_lvalue_reference<T>::type;

using std::add_rvalue_reference;
template <typename T>
using add_rvalue_reference_t = typename add_rvalue_reference<T>::type;

using std::remove_reference;
template <typename T>
using remove_reference_t = typename remove_reference<T>::type;

using std::is_convertible;
template <typename T, typename U>
constexpr bool is_convertible_v() { return is_convertible<T,U>::value; }
template <typename T, typename U, typename V=void>
using enable_if_convertible = enable_if<is_convertible<T,U>::value,V>;
template <typename T, typename U, typename V=void>
using enable_if_convertible_t = typename enable_if_convertible<T,U,V>::type;
template <typename T, typename U, typename V=void>
using enable_if_not_convertible = enable_if<!is_convertible<T,U>::value,V>;
template <typename T, typename U, typename V=void>
using enable_if_not_convertible_t = typename enable_if_not_convertible<T,U,V>::type;

using std::is_base_of;
template <typename T, typename U>
constexpr bool is_base_of_v() { return is_base_of<T,U>::value; }
template <typename T, typename U, typename V=void>
using enable_if_base_of = enable_if<is_base_of<T,U>::value,V>;
template <typename T, typename U, typename V=void>
using enable_if_base_of_t = typename enable_if_base_of<T,U,V>::type;
template <typename T, typename U, typename V=void>
using enable_if_not_base_of = enable_if<!is_base_of<T,U>::value,V>;
template <typename T, typename U, typename V=void>
using enable_if_not_base_of_t = typename enable_if_not_base_of<T,U,V>::type;

template <typename T> struct is_const_pointer_helper           : std::false_type {};
template <typename T> struct is_const_pointer_helper<const T*> :  std::true_type {};
template <typename T> struct is_const_pointer : is_const_pointer_helper<remove_cv_t<T>> {};
template <typename T>
constexpr bool is_const_pointer_v() { return is_const_pointer<T>::value; }
template <typename T, typename U=void>
using enable_if_const_pointer = enable_if<is_const_pointer<T>::value,U>;
template <typename T, typename U=void>
using enable_if_const_pointer_t = typename enable_if_const_pointer<T,U>::type;

template <typename T> struct is_non_const_pointer :
    std::integral_constant<bool,is_pointer<T>::value &&
                                !is_const_pointer<T>::value> {};
template <typename T>
constexpr bool is_non_const_pointer_v() { return is_const_pointer<T>::value; }
template <typename T, typename U=void>
using enable_if_non_const_pointer = enable_if<is_non_const_pointer<T>::value,U>;
template <typename T, typename U=void>
using enable_if_non_const_pointer_t = typename enable_if_non_const_pointer<T,U>::type;


template <typename...>
struct enable_if_exists { typedef void type; };
template <typename... Args>
using enable_if_exists_t = typename enable_if_exists<Args...>::type;

template <typename T, typename U> using is_similar = is_same<remove_cv_t<T>,remove_cv_t<U>>;
template <typename T, typename U>
constexpr bool is_similar_v() { return is_similar<T,U>::value; }
template <typename T, typename U, typename V=void>
using enable_if_similar = enable_if<is_similar<T,U>::value,V>;
template <typename T, typename U, typename V=void>
using enable_if_similar_t = typename enable_if_similar<T,U,V>::type;
template <typename T, typename U, typename V=void>
using enable_if_not_similar = enable_if<!is_similar<T,U>::value,V>;
template <typename T, typename U, typename V=void>
using enable_if_not_similar_t = typename enable_if_not_similar<T,U,V>::type;

template <typename... Args> struct has_member_helper;
template <typename... Args>
using has_member = stl_ext::conditional_t<false,
    has_member_helper<Args...>,void>;

}

#endif
