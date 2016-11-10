#include "utility.hpp"

#include "gtest/gtest.h"

using namespace std;
using namespace MArray;

TEST(utility, make_vector)
{
    vector<int> x1 = {1, 2, 3, 4};
    auto x2 = make_vector(1, 2, 3, 4);
    EXPECT_EQ(x1, x2);
}

TEST(utility, make_array)
{
    array<int, 4> x1 = {1, 2, 3, 4};
    auto x2 = make_array(1, 2, 3, 4);
    EXPECT_EQ(x1, x2);
}

TEST(utility, range_iterator_compare)
{
    range_t<int> r(1, 5);
    EXPECT_TRUE(r.begin() == r.end()-4);
    EXPECT_FALSE(r.begin() == r.end());
}

TEST(utility, range_iterator_dereference)
{
    range_t<int> r(1, 5);
    EXPECT_EQ(1, *r.begin());
}

TEST(utility, range_iterator_increment)
{
    range_t<int> r(1, 5);
    auto i = r.begin();
    ++i;
    auto j = i++;
    EXPECT_EQ(3, *i);
    EXPECT_EQ(2, *j);
}

TEST(utility, range_iterator_decrement)
{
    range_t<int> r(1, 5);
    auto i = r.end();
    --i;
    auto j = i--;
    EXPECT_EQ(3, *i);
    EXPECT_EQ(4, *j);
}

TEST(utility, range_iterator_add)
{
    range_t<int> r(1, 7);
    auto i = r.begin();
    EXPECT_EQ(3, *(i+2));
    EXPECT_EQ(6, *(5+i));
    i += 1;
    EXPECT_EQ(2, *i);
}

TEST(utility, range_iterator_sub)
{
    range_t<int> r(1, 5);
    auto i = r.end();
    EXPECT_EQ(3, *(i-2));
    i -= 1;
    EXPECT_EQ(4, *i);
}

TEST(utility, range_iterator_diff)
{
    range_t<int> r(1, 5);
    EXPECT_EQ(4, r.end()-r.begin());
}

TEST(utility, range_iterator_order)
{
    range_t<int> r(1, 5);
    EXPECT_TRUE(r.begin() < r.end());
    EXPECT_TRUE(r.end() > r.begin());
    EXPECT_TRUE(r.begin() <= r.end());
    EXPECT_TRUE(r.end() >= r.begin());
    EXPECT_TRUE(r.begin() <= r.begin());
    EXPECT_TRUE(r.begin() >= r.begin());
}

TEST(utility, range_iterator_index)
{
    range_t<int> r(1, 5);
    EXPECT_EQ(4, r.begin()[3]);
}

TEST(utility, range_iterator_swap)
{
    range_t<int> r(1, 5);
    auto b = r.begin();
    auto e = r.end();
    swap(b, e);
    EXPECT_TRUE(b == r.end());
    EXPECT_TRUE(e == r.begin());
}

TEST(utility, range_size)
{
    range_t<int> r(1, 5);
    EXPECT_EQ(4, r.size());
}

TEST(utility, range_begin_end)
{
    range_t<int> r(1, 5);
    EXPECT_EQ(1, *r.begin());
    EXPECT_EQ(4, *(r.end()-1));
}

TEST(utility, range_front_back)
{
    range_t<int> r(1, 5);
    EXPECT_EQ(1, r.front());
    EXPECT_EQ(4, r.back());
}

TEST(utility, range_index)
{
    range_t<int> r(1, 5);
    assert(r[7] == 8);
}

TEST(utility, range_vector)
{
    range_t<int> r(1, 5);
    vector<int> x1 = {1, 2, 3, 4};
    vector<int> x2 = r;
    EXPECT_EQ(x1, x2);
}

TEST(utility, range_to)
{
    vector<int> x1 = {0, 1, 2, 3};
    vector<int> x2 = range(4);
    EXPECT_EQ(x1, x2);
}

TEST(utility, range_from_to)
{
    vector<int> x1 = {1, 2, 3, 4};
    vector<int> x2 = range(1, 5);
    EXPECT_EQ(x1, x2);
}
