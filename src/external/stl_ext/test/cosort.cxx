#include <vector>

#include "gtest/gtest.h"

#include "cosort.hpp"

using namespace std;
using namespace stl_ext;

TEST(unit_cosort, cosort)
{
    vector<int> k = {4,8,1,6,0,-1,4,4,9,1};
    vector<int> v = {0,1,2,3,4,5,6,7,8,9};

    cosort(k, v);
    EXPECT_EQ(vector<int>({-1,0,1,1,4,4,4,6,8,9}), k);
    EXPECT_EQ(vector<int>({5,4,2,9,0,6,7,3,1,8}), v);
    cosort(k, v, greater<int>());
    EXPECT_EQ(vector<int>({9,8,6,4,4,4,1,1,0,-1}), k);
    EXPECT_EQ(vector<int>({8,1,3,0,6,7,2,9,4,5}), v);
}
