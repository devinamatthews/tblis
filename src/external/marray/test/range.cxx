#include "gtest/gtest.h"
#include "range.hpp"

using namespace std;
using namespace MArray;

TEST(range, iterator)
{
    range_t<int>::iterator it1;
    range_t<int>::iterator it2(0, 2);
    range_t<int>::iterator it3(5, 1);

    EXPECT_TRUE(it1 == it1);
    EXPECT_TRUE(it2 == it2);
    EXPECT_TRUE(it3 == it3);
    EXPECT_TRUE(it1 != it2);
    EXPECT_TRUE(it1 != it3);
    EXPECT_TRUE(it2 != it3);
    EXPECT_EQ(0, *it1);
    EXPECT_EQ(0, *it2);
    EXPECT_EQ(5, *it3);
    ++it1;
    ++it2;
    ++it3;
    EXPECT_EQ(0, *it1);
    EXPECT_EQ(2, *it2);
    EXPECT_EQ(6, *it3);
    EXPECT_EQ(0, *it1++);
    EXPECT_EQ(2, *it2++);
    EXPECT_EQ(6, *it3++);
    EXPECT_EQ(0, *it1);
    EXPECT_EQ(4, *it2);
    EXPECT_EQ(7, *it3);
    --it1;
    --it2;
    --it3;
    EXPECT_EQ(0, *it1);
    EXPECT_EQ(2, *it2);
    EXPECT_EQ(6, *it3);
    EXPECT_EQ(0, *it1--);
    EXPECT_EQ(2, *it2--);
    EXPECT_EQ(6, *it3--);
    EXPECT_EQ(0, *it1);
    EXPECT_EQ(0, *it2);
    EXPECT_EQ(5, *it3);
    EXPECT_EQ(0, *(it1 += 2));
    EXPECT_EQ(4, *(it2 += 2));
    EXPECT_EQ(7, *(it3 += 2));
    EXPECT_EQ(0, *(it1 -= 3));
    EXPECT_EQ(-2, *(it2 -= 3));
    EXPECT_EQ(4, *(it3 -= 3));
    EXPECT_EQ(0, *(it1 + 1));
    EXPECT_EQ(0, *(it2 + 1));
    EXPECT_EQ(5, *(it3 + 1));
    EXPECT_EQ(0, *(1 + it1));
    EXPECT_EQ(0, *(1 + it2));
    EXPECT_EQ(5, *(1 + it3));
    EXPECT_EQ(0, *(it1 - 1));
    EXPECT_EQ(-4, *(it2 - 1));
    EXPECT_EQ(3, *(it3 - 1));
    EXPECT_TRUE(it1 > it2);
    EXPECT_TRUE(it1 >= it1);
    EXPECT_TRUE(it1 <= it1);
    EXPECT_TRUE(it1 <= it3);
    EXPECT_TRUE(it1 < it3);
    EXPECT_FALSE(it1 < it2);
    EXPECT_FALSE(it1 < it1);
    EXPECT_FALSE(it2 >= it1);
    EXPECT_FALSE(it3 <= it2);
    EXPECT_FALSE(it1 > it3);
    EXPECT_EQ(0, it1[4]);
    EXPECT_EQ(6, it2[4]);
    EXPECT_EQ(8, it3[4]);
    swap(it1, it3);
    EXPECT_EQ(4, *it1);
    EXPECT_EQ(-2, *it2);
    EXPECT_EQ(0, *it3);
    EXPECT_EQ(8, it1[4]);
    EXPECT_EQ(6, it2[4]);
    EXPECT_EQ(0, it3[4]);

    range_t<int>::iterator it4(0, 2);
    range_t<int>::iterator it5(4, 2);
    range_t<int>::iterator it6(8, 2);
    EXPECT_EQ(0, it4-it4);
    EXPECT_EQ(-2, it4-it5);
    EXPECT_EQ(-4, it4-it6);
    EXPECT_EQ(0, it5-it5);
    EXPECT_EQ(-2, it5-it6);
    EXPECT_EQ(0, it6-it6);
}

TEST(range, range_t)
{
    range_t<int> r1;
    range_t<int> r2(0, 5, 2);
    range_t<int> r3(r2);
    range_t<int> r4(range_t<int>(1, 3, 1));

    EXPECT_EQ(0, r1.front());
    EXPECT_EQ(0, r1.back());
    EXPECT_EQ(0, r1.step());
    EXPECT_EQ(range_t<int>::iterator(0,0), r1.begin());
    EXPECT_EQ(range_t<int>::iterator(0,0), r1.end());
    EXPECT_EQ(0, r1[3]);

    EXPECT_EQ(0, r2.front());
    EXPECT_EQ(4, r2.back());
    EXPECT_EQ(2, r2.step());
    EXPECT_EQ(3, r2.size());
    EXPECT_EQ(range_t<int>::iterator(0,2), r2.begin());
    EXPECT_EQ(range_t<int>::iterator(6,2), r2.end());
    EXPECT_EQ(6, r2[3]);

    EXPECT_EQ(0, r3.front());
    EXPECT_EQ(4, r3.back());
    EXPECT_EQ(2, r3.step());
    EXPECT_EQ(3, r3.size());
    EXPECT_EQ(range_t<int>::iterator(0,2), r3.begin());
    EXPECT_EQ(range_t<int>::iterator(6,2), r3.end());
    EXPECT_EQ(6, r3[3]);

    EXPECT_EQ(1, r4.front());
    EXPECT_EQ(2, r4.back());
    EXPECT_EQ(1, r4.step());
    EXPECT_EQ(2, r4.size());
    EXPECT_EQ(range_t<int>::iterator(1,1), r4.begin());
    EXPECT_EQ(range_t<int>::iterator(3,1), r4.end());
    EXPECT_EQ(4, r4[3]);

    r1 = r2;
    r2 = range_t<int>(3, 7, 1);

    EXPECT_EQ(0, r1.front());
    EXPECT_EQ(4, r1.back());
    EXPECT_EQ(2, r1.step());
    EXPECT_EQ(3, r1.size());
    EXPECT_EQ(range_t<int>::iterator(0,2), r1.begin());
    EXPECT_EQ(range_t<int>::iterator(6,2), r1.end());
    EXPECT_EQ(6, r1[3]);

    EXPECT_EQ(3, r2.front());
    EXPECT_EQ(6, r2.back());
    EXPECT_EQ(1, r2.step());
    EXPECT_EQ(4, r2.size());
    EXPECT_EQ(range_t<int>::iterator(3,1), r2.begin());
    EXPECT_EQ(range_t<int>::iterator(7,1), r2.end());
    EXPECT_EQ(6, r2[3]);
}

TEST(range, vector)
{
    range_t<int> r1(0, 4, 1);
    range_t<int> r2(2, 7, 2);
    vector<int> v1 = r1;
    vector<int> v2 = r2;
    EXPECT_EQ((vector<int>{0,1,2,3}), v1);
    EXPECT_EQ((vector<int>{2,4,6}), v2);
}

TEST(range, string)
{
    range_t<char> r('A', 'E', 1);
    string s = r;
    EXPECT_EQ("ABCD", s);
}

TEST(range, range)
{
    vector<int> v1 = range(4);
    vector<int> v2 = range(1,4);
    vector<int> v3 = range(2,9,2);
    EXPECT_EQ((vector<int>{0,1,2,3}), v1);
    EXPECT_EQ((vector<int>{1,2,3}), v2);
    EXPECT_EQ((vector<int>{2,4,6,8}), v3);
}
