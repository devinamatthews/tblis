#include "gtest/gtest.h"

#include "type_traits.hpp"
#include "vector.hpp"

using namespace stl_ext;

TEST(unit_vector, vec)
{
    EXPECT_EQ(vector<int>({1,2,3}), vec(1,2,3));
    EXPECT_EQ(vector<int>({1}), vec(1));
    EXPECT_EQ(vector<int>({1}), vec(1));
    EXPECT_EQ(vector<double>({1.0,2.0}), vec(1.0,2));
    EXPECT_TRUE((is_same<float,decltype(vec(1.0f,2.0f))::value_type>::value));
}

TEST(unit_vector, slice)
{
    vector<int> v{0,1,2,3,4,5,6,7,8,9};
    EXPECT_EQ(vector<int>({4,5,6}), slice(v, 4, 7));
    EXPECT_EQ(vector<int>({4,5,6}), slice(vec(0,1,2,3,4,5,6,7,8,9), 4, 7));
    EXPECT_EQ(vector<int>({7,8,9}), slice(v, 7));
    EXPECT_EQ(vector<int>({7,8,9}), slice(vec(0,1,2,3,4,5,6,7,8,9), 7));
}

TEST(unit_vector, operators)
{
    vector<int> v1{0,1,2};
    vector<int> v2{3,4};
    EXPECT_EQ(vector<int>({0,1,2,3,4}), v1+v2);
    EXPECT_EQ(vector<int>({0,1,2,3,4}), vec(0,1,2)+v2);
    EXPECT_EQ(vector<int>({0,1,2,3,4}), v1+vec(3,4));
    EXPECT_EQ(vector<int>({0,1,2,3,4}), vec(0,1,2)+vec(3,4));
    EXPECT_EQ(vector<int>({0,1,2,3}), v1+3);
    EXPECT_EQ(vector<int>({0,1,2,3}), vec(0,1,2)+3);
    v1 += v2;
    EXPECT_EQ(vector<int>({0,1,2,3,4}), v1);
    v1 += vec(5,6);
    EXPECT_EQ(vector<int>({0,1,2,3,4,5,6}), v1);
    v1 += 7;
    EXPECT_EQ(vector<int>({0,1,2,3,4,5,6,7}), v1);
}
