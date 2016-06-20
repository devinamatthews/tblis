#include "gtest/gtest.h"

#include "type_traits.hpp"

using namespace stl_ext;

TEST(unit_type_traits, decay_t)
{
    EXPECT_TRUE((is_same<int*,decay_t<int(&)[]>>::value));
}

TEST(unit_type_traits, common_type_t)
{
    EXPECT_TRUE((is_same<long,common_type_t<char,long>>::value));
}

TEST(unit_type_traits, remove_const_t)
{
    EXPECT_TRUE((is_same<int,remove_const_t<const int>>::value));
}

TEST(unit_type_traits, remove_volatile_t)
{
    EXPECT_TRUE((is_same<int,remove_volatile_t<volatile int>>::value));
}

TEST(unit_type_traits, remove_cv_t)
{
    EXPECT_TRUE((is_same<int,remove_cv_t<const volatile int>>::value));
}

TEST(unit_type_traits, conditional_t)
{
    EXPECT_TRUE((is_same<int,conditional_t<true,int,float>>::value));
    EXPECT_TRUE((is_same<float,conditional_t<false,int,float>>::value));
}

template <int I, typename=void> struct foo1;
template <int I> struct foo1<I, enable_if_t<(I>=0)>> { int x[1]; };
template <int I> struct foo1<I, enable_if_t<(I<0)>> { int x[2]; };

TEST(unit_type_traits, enable_if_t)
{
    EXPECT_EQ(sizeof(int), sizeof(foo1<1>));
    EXPECT_EQ(2*sizeof(int), sizeof(foo1<-1>));
}

template <typename T, typename=void> struct foo2;
template <typename T> struct foo2<T, enable_if_same_t<T,int>> { int x[1]; };
template <typename T> struct foo2<T, enable_if_not_same_t<T,int>> { int x[2]; };

TEST(unit_type_traits, enable_if_same_t)
{
    EXPECT_EQ(sizeof(int), sizeof(foo2<int>));
    EXPECT_EQ(2*sizeof(int), sizeof(foo2<float>));
}

template <typename T, typename=void> struct foo3;
template <typename T> struct foo3<T, enable_if_const_t<T>> { int x[1]; };
template <typename T> struct foo3<T, enable_if_non_const_t<T>> { int x[2]; };

TEST(unit_type_traits, enable_if_const_t)
{
    EXPECT_EQ(sizeof(int), sizeof(foo3<const int>));
    EXPECT_EQ(2*sizeof(int), sizeof(foo3<int>));
}

template <typename T, typename=void> struct foo4;
template <typename T> struct foo4<T, enable_if_integral_t<T>> { int x[1]; };
template <typename T> struct foo4<T, enable_if_not_integral_t<T>> { int x[2]; };

TEST(unit_type_traits, enable_if_integral_t)
{
    EXPECT_EQ(sizeof(int), sizeof(foo4<int>));
    EXPECT_EQ(2*sizeof(int), sizeof(foo4<float>));
}

template <typename T, typename=void> struct foo5;
template <typename T> struct foo5<T, enable_if_floating_point_t<T>> { int x[1]; };
template <typename T> struct foo5<T, enable_if_not_floating_point_t<T>> { int x[2]; };

TEST(unit_type_traits, enable_if_floating_point_t)
{
    EXPECT_EQ(sizeof(int), sizeof(foo5<float>));
    EXPECT_EQ(2*sizeof(int), sizeof(foo5<int>));
}

template <typename T, typename=void> struct foo6;
template <typename T> struct foo6<T, enable_if_arithmetic_t<T>> { int x[1]; };
template <typename T> struct foo6<T, enable_if_not_arithmetic_t<T>> { int x[2]; };

TEST(unit_type_traits, enable_if_arithmetic_t)
{
    EXPECT_EQ(sizeof(int), sizeof(foo6<float>));
    EXPECT_EQ(sizeof(int), sizeof(foo6<int>));
    EXPECT_EQ(2*sizeof(int), sizeof(foo6<foo6<int>>));
}

template <typename T, typename=void> struct foo7;
template <typename T> struct foo7<T, enable_if_pointer_t<T>> { int x[1]; };
template <typename T> struct foo7<T, enable_if_not_pointer_t<T>> { int x[2]; };

TEST(unit_type_traits, enable_if_pointer_t)
{
    EXPECT_EQ(sizeof(int), sizeof(foo7<int*>));
    EXPECT_EQ(2*sizeof(int), sizeof(foo7<int>));
}

template <typename T, typename=void> struct foo8;
template <typename T> struct foo8<T, enable_if_reference_t<T>> { int x[1]; };
template <typename T> struct foo8<T, enable_if_not_reference_t<T>> { int x[2]; };

TEST(unit_type_traits, enable_if_reference_t)
{
    EXPECT_EQ(sizeof(int), sizeof(foo8<int&>));
    EXPECT_EQ(2*sizeof(int), sizeof(foo8<int>));
}

template <typename T, typename=void> struct foo9;
template <typename T> struct foo9<T, enable_if_convertible_t<T,int>> { int x[1]; };
template <typename T> struct foo9<T, enable_if_not_convertible_t<T,int>> { int x[2]; };

TEST(unit_type_traits, enable_if_convertible_t)
{
    EXPECT_EQ(sizeof(int), sizeof(foo9<char>));
    EXPECT_EQ(2*sizeof(int), sizeof(foo9<foo9<int>>));
}

struct bar10 {};
struct baz10 : bar10 {};

template <typename T, typename=void> struct foo10;
template <typename T> struct foo10<T, enable_if_base_of_t<bar10,T>> { int x[1]; };
template <typename T> struct foo10<T, enable_if_not_base_of_t<bar10,T>> { int x[2]; };

TEST(unit_type_traits, enable_if_base_of_t)
{
    EXPECT_EQ(sizeof(int), sizeof(foo10<baz10>));
    EXPECT_EQ(2*sizeof(int), sizeof(foo10<foo10<int>>));
}

struct bar11 { typedef int type; };

template <typename T, typename=void> struct foo11 { int x[2]; };
template <typename T> struct foo11<T, enable_if_exists_t<typename T::type>> { int x[1]; };

TEST(unit_type_traits, enable_if_exists_t)
{
    EXPECT_EQ(sizeof(int), sizeof(foo11<bar11>));
    EXPECT_EQ(2*sizeof(int), sizeof(foo11<int>));
}

template <typename T, typename=void> struct foo12;
template <typename T> struct foo12<T, enable_if_similar_t<T,int>> { int x[1]; };
template <typename T> struct foo12<T, enable_if_not_similar_t<T,int>> { int x[2]; };

TEST(unit_type_traits, enable_if_similar_t)
{
    EXPECT_EQ(sizeof(int), sizeof(foo12<int>));
    EXPECT_EQ(sizeof(int), sizeof(foo12<const int>));
    EXPECT_EQ(sizeof(int), sizeof(foo12<volatile int>));
    EXPECT_EQ(2*sizeof(int), sizeof(foo12<float>));
}
