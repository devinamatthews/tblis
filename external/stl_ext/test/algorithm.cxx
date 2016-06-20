#include <functional>
#include <vector>
#include <list>

#include "gtest/gtest.h"

#include "algorithm.hpp"

using namespace std;
using namespace stl_ext;

TEST(unit_algorithm, binary_or)
{
    binary_or<logical_not<bool>, logical_not<bool>>
        bor1{logical_not<bool>(), logical_not<bool>()};
    EXPECT_EQ(false, bor1(true));

    binary_or<logical_not<bool>, unary_negate<logical_not<bool>>>
        bor2{logical_not<bool>(), unary_negate<logical_not<bool>>(logical_not<bool>())};
    EXPECT_EQ(true, bor2(true));

    binary_or<unary_negate<logical_not<bool>>, logical_not<bool>>
        bor3{unary_negate<logical_not<bool>>(logical_not<bool>()), logical_not<bool>()};
    EXPECT_EQ(true, bor3(true));

    binary_or<unary_negate<logical_not<bool>>, unary_negate<logical_not<bool>>>
        bor4{unary_negate<logical_not<bool>>(logical_not<bool>()), unary_negate<logical_not<bool>>(logical_not<bool>())};
    EXPECT_EQ(true, bor4(true));
}

TEST(unit_algorithm, binary_and)
{
    binary_and<logical_not<bool>, logical_not<bool>>
        band1{logical_not<bool>(), logical_not<bool>()};
    EXPECT_EQ(false, band1(true));

    binary_and<logical_not<bool>, unary_negate<logical_not<bool>>>
        band2{logical_not<bool>(), unary_negate<logical_not<bool>>(logical_not<bool>())};
    EXPECT_EQ(false, band2(true));

    binary_and<unary_negate<logical_not<bool>>, logical_not<bool>>
        band3{unary_negate<logical_not<bool>>(logical_not<bool>()), logical_not<bool>()};
    EXPECT_EQ(false, band3(true));

    binary_and<unary_negate<logical_not<bool>>, unary_negate<logical_not<bool>>>
        band4{unary_negate<logical_not<bool>>(logical_not<bool>()), unary_negate<logical_not<bool>>(logical_not<bool>())};
    EXPECT_EQ(true, band4(true));
}

TEST(unit_algorithm, or1)
{
    EXPECT_EQ(false, or1(logical_not<bool>(), logical_not<bool>())(true));
    EXPECT_EQ(true, or1(logical_not<bool>(), not1(logical_not<bool>()))(true));
    EXPECT_EQ(true, or1(not1(logical_not<bool>()), logical_not<bool>())(true));
    EXPECT_EQ(true, or1(not1(logical_not<bool>()), not1(logical_not<bool>()))(true));
}

TEST(unit_algorithm, and1)
{
    EXPECT_EQ(false, and1(logical_not<bool>(), logical_not<bool>())(true));
    EXPECT_EQ(false, and1(logical_not<bool>(), not1(logical_not<bool>()))(true));
    EXPECT_EQ(false, and1(not1(logical_not<bool>()), logical_not<bool>())(true));
    EXPECT_EQ(true, and1(not1(logical_not<bool>()), not1(logical_not<bool>()))(true));
}

TEST(unit_algorithm, max)
{
    vector<int> v1 = {1,2,3,7,-2,6,1,4506,2,134,2};
    EXPECT_EQ(4506, max(v1));

    EXPECT_EQ(4506, max(vector<int>{1,2,3,7,-2,6,1,4506,2,134,2}));

    list<int> l1 = {1,2,3,7,-2,6,1,4506,2,134,2};
    EXPECT_EQ(4506, max(l1));

    EXPECT_EQ(4506, max(list<int>{1,2,3,7,-2,6,1,4506,2,134,2}));

    EXPECT_EQ(0, max(vector<int>{}));
}

TEST(unit_algorithm, min)
{
    vector<int> v1 = {1,2,3,7,-2,6,1,4506,2,134,2};
    EXPECT_EQ(-2, min(v1));

    EXPECT_EQ(-2, min(vector<int>{1,2,3,7,-2,6,1,4506,2,134,2}));

    list<int> l1 = {1,2,3,7,-2,6,1,4506,2,134,2};
    EXPECT_EQ(-2, min(l1));

    EXPECT_EQ(-2, min(list<int>{1,2,3,7,-2,6,1,4506,2,134,2}));

    EXPECT_EQ(0, min(vector<int>{}));
}

TEST(unit_algorithm, erase)
{
    vector<int> v = {1,2,3,4,5,6,7,8,9};
    erase(v, [](int x){return !(x%2);});
    EXPECT_EQ(vector<int>({1,3,5,7,9}), v);
    EXPECT_EQ(vector<int>({1,3,7,9}), erase(v, 5));

    vector<int> v2 = erased(v, [](int x){return x < 4;});
    EXPECT_EQ(vector<int>({7,9}), v2);
    EXPECT_EQ(vector<int>({9}), erased(v2, 7));
}

TEST(unit_algorithm, filter)
{
    vector<int> v = {1,2,3,4,5,6,7,8,9};
    filter(v, [](int x){return x%2;});
    EXPECT_EQ(vector<int>({1,3,5,7,9}), v);

    vector<int> v2 = filtered(v, [](int x){return x > 4;});
    EXPECT_EQ(vector<int>({1,3,5,7,9}), v);
    EXPECT_EQ(vector<int>({5,7,9}), v2);
}

TEST(unit_algorithm, apply)
{
    vector<int> v = {0,1,2,3,4};
    auto v2 = apply(v, [](int x){return pow(3.0,x);});
    EXPECT_EQ(vector<double>({1.0,3.0,9.0,27.0,81.0}), v2);
}

TEST(unit_algorithm, sum)
{
    vector<int> v = {0,1,2,3,4};
    EXPECT_EQ(10, sum(v));
    EXPECT_EQ(0, sum(vector<int>()));
}

TEST(unit_algorithm, contains)
{
    vector<int> v = {0,1,2,3,4};
    EXPECT_EQ(true, contains(v, 2));
    EXPECT_EQ(false, contains(v, -10));
}

TEST(unit_algorithm, sort)
{
    vector<int> v = {0,6,2,-1,4};
    EXPECT_EQ(vector<int>({-1,0,2,4,6}), sorted(v));
    EXPECT_EQ(vector<int>({0,6,2,-1,4}), v);
    EXPECT_EQ(vector<int>({-1,0,2,4,6}), sort(v));

    EXPECT_EQ(vector<int>({6,4,2,0,-1}), sorted(v, greater<int>()));
    EXPECT_EQ(vector<int>({-1,0,2,4,6}), v);
    EXPECT_EQ(vector<int>({6,4,2,0,-1}), sort(v, greater<int>()));
}

TEST(unit_algorithm, unique)
{
    vector<int> v = {1,2,3,7,-2,6,1,4506,2,134,2};
    EXPECT_EQ(vector<int>({-2,1,2,3,6,7,134,4506}), uniqued(v));
    EXPECT_EQ(vector<int>({1,2,3,7,-2,6,1,4506,2,134,2}), v);
    EXPECT_EQ(vector<int>({-2,1,2,3,6,7,134,4506}), unique(v));
}

TEST(unit_algorithm, intersect)
{
    vector<int> v1 = {0,1,2,3,4,5,6};
    vector<int> v2 = {1,3,5,7,9,11};
    EXPECT_EQ(vector<int>({1,3,5}), intersection(v1, v2));
    EXPECT_EQ(vector<int>({0,1,2,3,4,5,6}), v1);
    EXPECT_EQ(vector<int>({1,3,5,7,9,11}), v2);
    EXPECT_EQ(vector<int>({1,3,5}), intersect(v1, v2));
    EXPECT_EQ(vector<int>({1,3,5,7,9,11}), v2);
    EXPECT_EQ(vector<int>({1,3,5}), intersect(v2, v1));
}

TEST(unit_algorithm, exclude)
{
    vector<int> v1 = {0,1,2,3,4,5,6};
    vector<int> v2 = {1,3,5,7,9,11};
    EXPECT_EQ(vector<int>({0,2,4,6}), exclusion(v1, v2));
    EXPECT_EQ(vector<int>({7,9,11}), exclusion(v2, v1));
    EXPECT_EQ(vector<int>({0,1,2,3,4,5,6}), v1);
    EXPECT_EQ(vector<int>({1,3,5,7,9,11}), v2);
    EXPECT_EQ(vector<int>({0,2,4,6}), exclude(v1, v2));
    EXPECT_EQ(vector<int>({1,3,5,7,9,11}), v2);
    v1 = {0,1,2,3,4,5,6};
    EXPECT_EQ(vector<int>({7,9,11}), exclude(v2, v1));
}

TEST(unit_algorithm, mutual_exclusion)
{
    vector<int> v1 = {0,1,2,3,4,5,6};
    vector<int> v2 = {1,3,5,7,9,11};
    EXPECT_EQ(vector<int>({0,2,4,6,7,9,11}), mutual_exclusion(v1, v2));
}

TEST(unit_algorithm, mask)
{
    vector<int> v = {0,1,2,3,4,5,6};
    vector<int> m = {0,1,0,1,0,1,0};
    EXPECT_EQ(vector<int>({1,3,5}), masked(v, m));
    EXPECT_EQ(vector<int>({0,1,2,3,4,5,6}), v);
    EXPECT_EQ(vector<int>({1,3,5}), mask(v, m));
}

TEST(unit_algorithm, translate)
{
    vector<int> v = {0,1,2,3,4,5,6};
    vector<int> from = {2,4,6};
    vector<int> to = {-2,-4,-6};
    EXPECT_EQ(vector<int>({0,1,-2,3,-4,5,-6}), translated(v, from, to));
    EXPECT_EQ(vector<int>({0,1,2,3,4,5,6}), v);
    EXPECT_EQ(vector<int>({0,1,-2,3,-4,5,-6}), translate(v, from, to));
}
