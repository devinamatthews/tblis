#include "gtest/gtest.h"
#include "marray.hpp"
#include "expression.hpp"

#include <typeinfo>
#include <complex>

using namespace std;
using namespace MArray;

template <typename... T> struct types;

typedef types<float,double,std::complex<float>,std::complex<double>> float_types;
typedef types<int8_t,int16_t,int32_t,int64_t> int_types;
typedef types<uint8_t,uint16_t,uint32_t,uint64_t> uint_types;

template <typename T, typename U>
struct concat_types;

template <typename... T, typename... U>
struct concat_types<types<T...>, types<U...>>
{
    typedef types<T..., U...> type;
};

template <typename T, typename U>
struct product_types;

template <typename... U>
struct product_types<types<>, types<U...>>
{
    typedef types<> type;
};

template <typename T, typename... TT, typename... U>
struct product_types<types<T, TT...>, types<U...>>
{
    typedef typename concat_types<types<std::pair<T,U>...>,
        typename product_types<types<TT...>,
                               types<U...>>::type>::type type;
};

typedef typename concat_types<float_types,
    typename concat_types<int_types,uint_types>::type>::type all_types;

template <typename T>
struct to_testing_types;

template <typename... T>
struct to_testing_types<types<T...>>
{
    typedef testing::Types<T...> type;
};

template <typename T, typename U>
using pair_types = typename to_testing_types<
    typename product_types<T, U>::type>::type;

typedef pair_types<all_types, all_types> pair_types_all;
typedef pair_types<float_types, float_types> pair_types_ff;
typedef pair_types<float_types, float_types> pair_types_fi;
typedef pair_types<float_types, float_types> pair_types_fu;
typedef pair_types<float_types, float_types> pair_types_ii;
typedef pair_types<float_types, float_types> pair_types_iu;
typedef pair_types<float_types, float_types> pair_types_uu;

#define VECTOR_INITIALIZER { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15, \
                            16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31, \
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15, \
                            16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31, \
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15, \
                            16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31, \
                             0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15, \
                            16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}

template <class T> struct expression_vector : testing::Test {};

TYPED_TEST_CASE(expression_vector, pair_types_all);

//typedef pair_types<types<double>, types<std::complex<double>>> tmp_types;
//TYPED_TEST_CASE(expression_vector, tmp_types);

/*
 * These are handled outside of the expression framework now.
 *

TYPED_TEST(expression_vector, vector_assign)
{
    typedef typename TypeParam::first_type T;
    typedef typename TypeParam::second_type U;

    T data1[128] = VECTOR_INITIALIZER;
    U data2[128] = {0};

    marray_view<T,1> a({128}, data1);
    marray_view<U,1> b({128}, data2);

    b = a;

    for (int i = 0;i < 128;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(T(i%32), a[i]);
        EXPECT_EQ(U(i%32), b[i]);
    }
}

TYPED_TEST(expression_vector, vector_set)
{
    typedef typename TypeParam::first_type T;
    typedef typename TypeParam::second_type U;

    U data2[128] = {0};

    marray_view<U,1> b({128}, data2);

    b = T(3);

    for (int i = 0;i < 128;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(U(3), b[i]);
    }
}

*/

TYPED_TEST(expression_vector, vector_add)
{
    typedef typename TypeParam::first_type T;
    typedef typename TypeParam::second_type U;
    typedef decltype(std::declval<operators::plus>()(std::declval<T>(),
                                                     std::declval<U>())) V;

    T data1[128] = VECTOR_INITIALIZER;
    U data2[128] = {0};

    marray_view<T,1> a({128}, data1);
    marray_view<U,1> b({128}, data2);

    b = a+a;

    for (int i = 0;i < 128;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(T(i%32), a[i]);
        EXPECT_EQ(U(2*(i%32)), b[i]);
    }

    T data3[128] = VECTOR_INITIALIZER;
    U data4[128] = VECTOR_INITIALIZER;

    marray_view<T,1> c({128}, data3);
    marray_view<U,1> d({128}, data4);
    marray<V,1> e({128}, V());

    e = c+d;

    for (int i = 0;i < 128;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(T(i%32), c[i]);
        EXPECT_EQ(U(i%32), d[i]);
        EXPECT_EQ(V(2*(i%32)), e[i]);
    }
}

TYPED_TEST(expression_vector, vector_sub)
{
    typedef typename TypeParam::first_type T;
    typedef typename TypeParam::second_type U;
    typedef decltype(std::declval<operators::minus>()(std::declval<T>(),
                                                      std::declval<U>())) V;

    T data1[128] = VECTOR_INITIALIZER;
    U data2[128] = {0};

    marray_view<T,1> a({128}, data1);
    marray_view<U,1> b({128}, data2);

    b = a-a;

    for (int i = 0;i < 128;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(T(i%32), a[i]);
        EXPECT_EQ(U(0), b[i]);
    }

    T data3[128] = VECTOR_INITIALIZER;
    U data4[128] = VECTOR_INITIALIZER;

    marray_view<T,1> c({128}, data3);
    marray_view<U,1> d({128}, data4);
    marray<V,1> e({128}, V());

    e = c-d;

    for (int i = 0;i < 128;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(T(i%32), c[i]);
        EXPECT_EQ(U(i%32), d[i]);
        EXPECT_EQ(V(0), e[i]);
    }
}

TYPED_TEST(expression_vector, vector_mul)
{
    typedef typename TypeParam::first_type T;
    typedef typename TypeParam::second_type U;
    typedef decltype(std::declval<operators::multiplies>()(std::declval<T>(),
                                                           std::declval<U>())) V;

    // avoid overflow in floating -> integral since it is undefined and
    // in integral*integral because signed overflow is undefined
    T data1[128] = { 0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7};
    U data2[128] = {0};

    marray_view<T,1> a({128}, data1);
    marray_view<U,1> b({128}, data2);

    b = a*a;

    for (int i = 0;i < 128;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(T(i%8), a[i]);
        EXPECT_EQ(U((i%8)*(i%8)), b[i]);
    }

    T data3[128] = { 0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7};
    U data4[128] = { 0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7, \
                     0, 1, 2, 3, 4, 5, 6, 7};

    marray_view<T,1> c({128}, data3);
    marray_view<U,1> d({128}, data4);
    marray<V,1> e({128}, V());

    e = c*d;

    for (int i = 0;i < 128;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(T(i%8), c[i]);
        EXPECT_EQ(U(i%8), d[i]);
        EXPECT_EQ(V((i%8)*(i%8)), e[i]);
    }
}

TYPED_TEST(expression_vector, vector_div)
{
    typedef typename TypeParam::first_type T;
    typedef typename TypeParam::second_type U;
    typedef decltype(std::declval<operators::divides>()(std::declval<T>(),
                                                        std::declval<U>())) V;

    T data1[128] = VECTOR_INITIALIZER;
    U data2[128] = {0};

    marray_view<T,1> a({128}, data1);
    marray_view<U,1> b({128}, data2);
    a[ 0] = T(1);
    a[32] = T(1);
    a[64] = T(1);
    a[96] = T(1);

    b = a/a;

    for (int i = 0;i < 128;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(T(std::max(1,i%32)), a[i]);
        EXPECT_EQ(U(1), b[i]);
    }

    T data3[128] = VECTOR_INITIALIZER;
    U data4[128] = VECTOR_INITIALIZER;

    marray_view<T,1> c({128}, data3);
    marray_view<U,1> d({128}, data4);
    marray<V,1> e({128}, V());
    c[ 0] = T(1);
    c[32] = T(1);
    c[64] = T(1);
    c[96] = T(1);
    d[ 0] = U(1);
    d[32] = U(1);
    d[64] = U(1);
    d[96] = U(1);

    e = c/d;

    for (int i = 0;i < 128;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(T(std::max(1,i%32)), c[i]);
        EXPECT_EQ(U(std::max(1,i%32)), d[i]);
        EXPECT_EQ(V(1), e[i]);
    }
}

TEST(expression, assign)
{
    using namespace slice;

    double data1[12] = { 1, 2, 3, 4, 5, 6,
                         7, 8, 9,10,11,12};
    double data2[12] = {12,11,10, 9, 8, 7,
                         6, 5, 4, 3, 2, 1};
    double data3[6] = {-1,-1,-1,-1,-1,-1};

    marray<double,3> v1{2, 3, 2};
    marray_view<double,3> v2({2, 3, 2}, data1);
    marray<double,4> v3({4, 3, 2, 2}, 1.0);
    marray<double,3> v4({2, 3, 2}, 4.0);
    marray_view<double,3> v5({2, 3, 2}, data2);
    marray_view<double,2> v6({3, 2}, data3);
    marray<double,2> v7({3, 2}, 5.0);

    v1 = 2.0;
    EXPECT_EQ((array<double,12>{2, 2, 2, 2, 2, 2,
                                2, 2, 2, 2, 2, 2}), *(array<double,12>*)v1.data());

    v1 = 3;
    EXPECT_EQ((array<double,12>{3, 3, 3, 3, 3, 3,
                                3, 3, 3, 3, 3, 3}), *(array<double,12>*)v1.data());

    v1 = v2;
    EXPECT_EQ((array<double,12>{ 1, 2, 3, 4, 5, 6,
                                 7, 8, 9,10,11,12}), *(array<double,12>*)v1.data());

    v1 = v3[range(2)][all][1][all];
    EXPECT_EQ((array<double,12>{ 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1}), *(array<double,12>*)v1.data());

    v1 = v4;
    EXPECT_EQ((array<double,12>{ 4, 4, 4, 4, 4, 4,
                                 4, 4, 4, 4, 4, 4}), *(array<double,12>*)v1.data());

    v2 = 2.0;
    EXPECT_EQ((array<double,12>{2, 2, 2, 2, 2, 2,
                                2, 2, 2, 2, 2, 2}), *(array<double,12>*)v2.data());

    v2 = 3;
    EXPECT_EQ((array<double,12>{3, 3, 3, 3, 3, 3,
                                3, 3, 3, 3, 3, 3}), *(array<double,12>*)v2.data());

    v2 = v5;
    EXPECT_EQ((array<double,12>{12,11,10, 9, 8, 7,
                                 6, 5, 4, 3, 2, 1}), *(array<double,12>*)v2.data());

    v2 = v3[range(2)][all][1][all];
    EXPECT_EQ((array<double,12>{ 1, 1, 1, 1, 1, 1,
                                 1, 1, 1, 1, 1, 1}), *(array<double,12>*)v2.data());

    v2 = v4;
    EXPECT_EQ((array<double,12>{ 4, 4, 4, 4, 4, 4,
                                 4, 4, 4, 4, 4, 4}), *(array<double,12>*)v2.data());

    v1[1][all][all] = 2.0;
    EXPECT_EQ((array<double,12>{4, 4, 4, 4, 4, 4,
                                2, 2, 2, 2, 2, 2}), *(array<double,12>*)v1.data());

    v1[1][all][all] = 3;
    EXPECT_EQ((array<double,12>{4, 4, 4, 4, 4, 4,
                                3, 3, 3, 3, 3, 3}), *(array<double,12>*)v1.data());

    v1[1][all][all] = v6;
    EXPECT_EQ((array<double,12>{ 4, 4, 4, 4, 4, 4,
                                -1,-1,-1,-1,-1,-1}), *(array<double,12>*)v1.data());

    v1[1][all][all] = v3[1][all][1][all];
    EXPECT_EQ((array<double,12>{ 4, 4, 4, 4, 4, 4,
                                 1, 1, 1, 1, 1, 1}), *(array<double,12>*)v1.data());

    v1[1][all][all] = v7;
    EXPECT_EQ((array<double,12>{ 4, 4, 4, 4, 4, 4,
                                 5, 5, 5, 5, 5, 5}), *(array<double,12>*)v1.data());
}

TEST(expression, bcast)
{
    using namespace slice;

    double data[3] = {1, 2, 3};

    marray<double,3> v1{3, 2, 3};
    marray_view<double,1> v2({3}, data);

    v1 = v2;
    EXPECT_EQ((array<double,18>{1, 2, 3, 1, 2, 3,
                                1, 2, 3, 1, 2, 3,
                                1, 2, 3, 1, 2, 3}), *(array<double,18>*)v1.data());

    v1 = 0;
    EXPECT_EQ((array<double,18>{0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0,
                                0, 0, 0, 0, 0, 0}), *(array<double,18>*)v1.data());

    v1 = v2[bcast][bcast];
    EXPECT_EQ((array<double,18>{1, 2, 3, 1, 2, 3,
                                1, 2, 3, 1, 2, 3,
                                1, 2, 3, 1, 2, 3}), *(array<double,18>*)v1.data());

    v1 = v2[all][bcast][bcast];
    EXPECT_EQ((array<double,18>{1, 1, 1, 1, 1, 1,
                                2, 2, 2, 2, 2, 2,
                                3, 3, 3, 3, 3, 3}), *(array<double,18>*)v1.data());
}

TEST(expression, add)
{
    double data1[3] = {1, 2, 3};
    double data2[3] = {3, 2, 1};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);
    marray_view<double,1> v3({3}, data2);

    v1 = v2 + v3;
    EXPECT_EQ((array<double,3>{4, 4, 4}), *(array<double,3>*)v1.data());

    v1 = v2 + 1;
    EXPECT_EQ((array<double,3>{2, 3, 4}), *(array<double,3>*)v1.data());

    v1 = 2.0 + v3;
    EXPECT_EQ((array<double,3>{5, 4, 3}), *(array<double,3>*)v1.data());

    v1 += v2;
    EXPECT_EQ((array<double,3>{6, 6, 6}), *(array<double,3>*)v1.data());

    v1 += 1;
    EXPECT_EQ((array<double,3>{7, 7, 7}), *(array<double,3>*)v1.data());
}

TEST(expression, sub)
{
    double data1[3] = {1, 2, 3};
    double data2[3] = {3, 2, 1};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);
    marray_view<double,1> v3({3}, data2);

    v1 = v2 - v3;
    EXPECT_EQ((array<double,3>{-2, 0, 2}), *(array<double,3>*)v1.data());

    v1 = v2 - 1;
    EXPECT_EQ((array<double,3>{0, 1, 2}), *(array<double,3>*)v1.data());

    v1 = 2.0 - v3;
    EXPECT_EQ((array<double,3>{-1, 0, 1}), *(array<double,3>*)v1.data());

    v1 -= v2;
    EXPECT_EQ((array<double,3>{-2, -2, -2}), *(array<double,3>*)v1.data());

    v1 -= 1;
    EXPECT_EQ((array<double,3>{-3, -3, -3}), *(array<double,3>*)v1.data());
}

TEST(expression, mul)
{
    double data1[3] = {1, 2, 3};
    double data2[3] = {3, 2, 1};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);
    marray_view<double,1> v3({3}, data2);

    v1 = v2 * v3;
    EXPECT_EQ((array<double,3>{3, 4, 3}), *(array<double,3>*)v1.data());

    v1 = v2 * 1;
    EXPECT_EQ((array<double,3>{1, 2, 3}), *(array<double,3>*)v1.data());

    v1 = 2.0 * v3;
    EXPECT_EQ((array<double,3>{6, 4, 2}), *(array<double,3>*)v1.data());

    v1 *= v2;
    EXPECT_EQ((array<double,3>{6, 8, 6}), *(array<double,3>*)v1.data());

    v1 *= 2;
    EXPECT_EQ((array<double,3>{12, 16, 12}), *(array<double,3>*)v1.data());
}

TEST(expression, div)
{
    double data1[3] = {1, 2, 3};
    double data2[3] = {3, 2, 1};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);
    marray_view<double,1> v3({3}, data2);

    v1 = v2 / v3;
    EXPECT_EQ((array<double,3>{1.0/3, 1, 3}), *(array<double,3>*)v1.data());

    v1 = v2 / 1;
    EXPECT_EQ((array<double,3>{1, 2, 3}), *(array<double,3>*)v1.data());

    v1 = 2.0 / v3;
    EXPECT_EQ((array<double,3>{2.0/3, 1, 2}), *(array<double,3>*)v1.data());

    v1 /= v2;
    EXPECT_EQ((array<double,3>{2.0/3, 0.5, 2.0/3}), *(array<double,3>*)v1.data());

    v1 /= 2;
    EXPECT_EQ((array<double,3>{1.0/3, 0.25, 1.0/3}), *(array<double,3>*)v1.data());
}

TEST(expression, pow)
{
    double data1[3] = {1, 2, 3};
    double data2[3] = {3, 2, 1};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);
    marray_view<double,1> v3({3}, data2);

    v1 = pow(v2, v3);
    EXPECT_EQ((array<double,3>{1, 4, 3}), *(array<double,3>*)v1.data());

    v1 = pow(v2, 2);
    EXPECT_EQ((array<double,3>{1, 4, 9}), *(array<double,3>*)v1.data());

    v1 = pow(2.0, v3);
    EXPECT_EQ((array<double,3>{8, 4, 2}), *(array<double,3>*)v1.data());
}

TEST(expression, negate)
{
    double data1[3] = {1, 2, 3};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);

    v1 = -v2;
    EXPECT_EQ((array<double,3>{-1, -2, -3}), *(array<double,3>*)v1.data());
}

TEST(expression, exp)
{
    double data1[3] = {1, 2, 3};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);

    v1 = exp(v2);
    EXPECT_EQ((array<double,3>{exp(1), exp(2), exp(3)}), *(array<double,3>*)v1.data());
}

TEST(expression, sqrt)
{
    double data1[3] = {4, 9, 16};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);

    v1 = sqrt(v2);
    EXPECT_EQ((array<double,3>{2, 3, 4}), *(array<double,3>*)v1.data());
}

TEST(expression, compound)
{
    double data1[3] = {1, 2, 3};
    double data2[3] = {3, 2, 1};
    double data3[3] = {4, 7, 2};

    marray<double,1> v1({3}, 0);
    marray_view<double,1> v2({3}, data1);
    marray_view<double,1> v3({3}, data2);
    marray_view<double,1> v4({3}, data3);

    v1 = (pow(v2, 2) * v3 + 1)/4 + sqrt(v4);
    EXPECT_DOUBLE_EQ(3, v1[0]);
    EXPECT_DOUBLE_EQ(9.0/4 + sqrt(7), v1[1]);
    EXPECT_DOUBLE_EQ(5.0/2 + sqrt(2), v1[2]);
}

TEST(expression, mixed_rank)
{
    double data1[12] = {1, 2, 3, 4, 5, 6,
                        7, 8, 9,10,11,12};
    double data2[6] = {1, 2, 3, 4, 5, 6};
    double data3[3] = {3, 2, 1};

    marray<double,3> v1{2, 2, 3};
    marray_view<double,3> v2({2, 2, 3}, data1);
    marray_view<double,2> v3({2, 3}, data2);
    marray_view<double,1> v4({3}, data3);

    v1 = v2 * 2 + v3 / v4;
    EXPECT_DOUBLE_EQ( 2 + 1.0/3, v1.data()[ 0]);
    EXPECT_DOUBLE_EQ( 4 + 2.0/2, v1.data()[ 1]);
    EXPECT_DOUBLE_EQ( 6 + 3.0/1, v1.data()[ 2]);
    EXPECT_DOUBLE_EQ( 8 + 4.0/3, v1.data()[ 3]);
    EXPECT_DOUBLE_EQ(10 + 5.0/2, v1.data()[ 4]);
    EXPECT_DOUBLE_EQ(12 + 6.0/1, v1.data()[ 5]);
    EXPECT_DOUBLE_EQ(14 + 1.0/3, v1.data()[ 6]);
    EXPECT_DOUBLE_EQ(16 + 2.0/2, v1.data()[ 7]);
    EXPECT_DOUBLE_EQ(18 + 3.0/1, v1.data()[ 8]);
    EXPECT_DOUBLE_EQ(20 + 4.0/3, v1.data()[ 9]);
    EXPECT_DOUBLE_EQ(22 + 5.0/2, v1.data()[10]);
    EXPECT_DOUBLE_EQ(24 + 6.0/1, v1.data()[11]);
}
