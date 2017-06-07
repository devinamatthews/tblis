#include "gtest/gtest.h"
#include "utility.hpp"

using namespace std;
using namespace MArray;

TEST(utility, make_vector)
{
    vector<int> v1 = {1, 2, 3, 4};
    auto v2 = make_vector(1, 2, 3, 4);
    EXPECT_EQ(v1, v2);
}

TEST(utility, make_array)
{
    array<int,4> v1 = {1, 2, 3, 4};
    auto v2 = make_array(1, 2, 3, 4);
    EXPECT_EQ(v1, v2);
}

TEST(utility, push_back)
{
    tuple<int,double,float> t1{1, 2.0, 3.0f};
    auto t2 = push_back(t1, 4);
    auto t3 = push_back(tuple<int,double,float>{1, 2.0, 3.0f}, 4);
    EXPECT_EQ(true, (is_same<decltype(t2),tuple<int,double,float,int>>::value));
    EXPECT_EQ(true, (is_same<decltype(t3),tuple<int,double,float,int>>::value));
    EXPECT_EQ(4, get<3>(t2));
    EXPECT_EQ(4, get<3>(t3));
}
