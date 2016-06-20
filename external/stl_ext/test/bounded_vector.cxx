#include "gtest/gtest.h"

#include "bounded_vector.hpp"

using namespace std;
using namespace stl_ext;

TEST(unit_bounded_vector, constructor)
{
    bounded_vector<int,10> pl1;
    EXPECT_EQ(0, pl1.size());

    bounded_vector<int,10> pl2(4);
    EXPECT_EQ(4, pl2.size());
    EXPECT_EQ(0, pl2.front());

    bounded_vector<int,10> pl3(4, 4);
    EXPECT_EQ(4, pl3.size());
    EXPECT_EQ(4, pl3.front());

    bounded_vector<int,10> pl4(pl3);
    EXPECT_EQ(4, pl4.size());
    EXPECT_EQ(4, pl4.front());

    bounded_vector<int,10> pl5(std::move(pl4));
    EXPECT_EQ(4, pl4.size());
    EXPECT_EQ(4, pl5.size());
    EXPECT_EQ(4, pl5.front());

    bounded_vector<int,10> pl6 = {1, 2, 3, 4};
    EXPECT_EQ(4, pl6.size());
    EXPECT_EQ(1, pl6.front());
    EXPECT_EQ(4, pl6.back());

    bounded_vector<int,10> pl8{1,2,3,4};
    bounded_vector<int,10> pl9(pl8.begin(), pl8.end());
    EXPECT_EQ(4, pl9.size());
    EXPECT_EQ(1, pl9.front());
    EXPECT_EQ(4, pl9.back());
}

TEST(unit_bounded_vector, assign)
{
    bounded_vector<int,10> pl3;
    pl3.assign(4, 4);
    EXPECT_EQ(4, pl3.size());
    EXPECT_EQ(4, pl3.front());

    bounded_vector<int,10> pl4;
    pl4 = pl3;
    EXPECT_EQ(4, pl4.size());
    EXPECT_EQ(4, pl4.front());

    bounded_vector<int,10> pl5;
    pl5 = std::move(pl4);
    EXPECT_EQ(4, pl4.size());
    EXPECT_EQ(4, pl5.size());
    EXPECT_EQ(4, pl5.front());

    bounded_vector<int,10> pl6;
    pl6 = {1, 2, 3, 4};
    EXPECT_EQ(4, pl6.size());
    EXPECT_EQ(1, pl6.front());
    EXPECT_EQ(4, pl6.back());

    bounded_vector<int,10> pl8{1,2,3,4};
    bounded_vector<int,10> pl9;
    pl9.assign(pl8.begin(), pl8.end());
    EXPECT_EQ(4, pl9.size());
    EXPECT_EQ(1, pl9.front());
    EXPECT_EQ(4, pl9.back());
}

TEST(unit_bounded_vector, begin_end)
{
    bounded_vector<int,10> pl{1, 2, 3, 4};
    int *x = pl.data();

    EXPECT_EQ(1, *pl.begin());
    EXPECT_EQ(4, *prev(pl.end()));
    EXPECT_EQ(4, *pl.rbegin());
    EXPECT_EQ(1, *prev(pl.rend()));
    EXPECT_EQ(1, *pl.cbegin());
    EXPECT_EQ(4, *prev(pl.cend()));
    EXPECT_EQ(4, *pl.crbegin());
    EXPECT_EQ(1, *prev(pl.crend()));

    EXPECT_EQ(x  , &*pl.begin());
    EXPECT_EQ(x+3, &*prev(pl.end()));
    EXPECT_EQ(x+3, &*pl.rbegin());
    EXPECT_EQ(x  , &*prev(pl.rend()));
    EXPECT_EQ(x  , &*pl.cbegin());
    EXPECT_EQ(x+3, &*prev(pl.cend()));
    EXPECT_EQ(x+3, &*pl.crbegin());
    EXPECT_EQ(x  , &*prev(pl.crend()));
}

TEST(unit_bounded_vector, front_back)
{
    bounded_vector<int,10> pl{1, 2, 3, 4};

    EXPECT_EQ(1, pl.front());
    EXPECT_EQ(4, pl.back());
}

TEST(unit_bounded_vector, empty)
{
    bounded_vector<int,10> pl;
    EXPECT_TRUE(pl.empty());
    pl = {1, 2, 3, 4};
    EXPECT_FALSE(pl.empty());
}

TEST(unit_bounded_vector, push_pop)
{
    bounded_vector<int,10> sl;
    EXPECT_EQ(0, sl.size());

    sl.push_back(1);
    EXPECT_EQ(1, sl.size());
    EXPECT_EQ(1, sl.back());
    int sp2 = 2;
    sl.push_back(sp2);
    EXPECT_EQ(2, sl.size());
    EXPECT_EQ(2, sl.back());
    sl.push_back(3);
    EXPECT_EQ(3, sl.size());
    EXPECT_EQ(3, sl.back());
    sl.push_back(4);
    EXPECT_EQ(4, sl.size());
    EXPECT_EQ(4, sl.back());
    sl.pop_back();
    EXPECT_EQ(3, sl.size());
    EXPECT_EQ(3, sl.back());
}

TEST(unit_bounded_vector, resize)
{
    bounded_vector<int,10> pl;
    EXPECT_EQ(0, pl.size());
    pl.resize(3);
    EXPECT_EQ(3, pl.size());
    EXPECT_EQ(0, pl.front());
    EXPECT_EQ(0, pl.back());
    pl.resize(4, 4);
    EXPECT_EQ(4, pl.size());
    EXPECT_EQ(0, pl.front());
    EXPECT_EQ(4, pl.back());
    pl.resize(2);
    EXPECT_EQ(2, pl.size());
    EXPECT_EQ(0, pl.front());
    EXPECT_EQ(0, pl.back());
}

TEST(unit_bounded_vector, insert)
{
    bounded_vector<int,10> sl;
    EXPECT_EQ(0, sl.size());
    sl.insert(sl.begin(), 3);
    EXPECT_EQ(1, sl.size());
    EXPECT_EQ(3, sl.front());
    EXPECT_EQ(3, sl.back());
    int sp = 1;
    sl.insert(sl.begin(), sp);
    EXPECT_EQ(2, sl.size());
    EXPECT_EQ(1, sl.front());
    EXPECT_EQ(3, sl.back());
    auto i = sl.insert(sl.end(), 2);
    EXPECT_EQ(3, sl.size());
    EXPECT_EQ(1, sl.front());
    EXPECT_EQ(2, sl.back());
    i = sl.insert(i, 6);
    EXPECT_EQ(4, sl.size());
    EXPECT_EQ(6, *i);

    sl.clear();
    EXPECT_EQ(0, sl.size());
    i = sl.insert(sl.begin(), 2, 2);
    EXPECT_EQ(2, sl.size());
    EXPECT_EQ(2, sl.front());
    EXPECT_EQ(2, sl.back());
    EXPECT_EQ(sl.begin(), i);
    i = sl.insert(sl.end(), {1,2,3});
    EXPECT_EQ(5, sl.size());
    EXPECT_EQ(1, *i);
    EXPECT_EQ(3, sl.back());
    EXPECT_EQ(next(sl.begin(), 2), i);
    i = sl.insert(sl.end(), {7});
    EXPECT_EQ(6, sl.size());
    EXPECT_EQ(7, *i);
    EXPECT_EQ(7, sl.back());
}

TEST(unit_bounded_vector, erase)
{
    bounded_vector<int,10> pl({0,1,2,3,4,5,6,7,8,9});
    EXPECT_EQ(10, pl.size());
    EXPECT_EQ(0, pl.front());
    EXPECT_EQ(9, pl.back());
    pl.erase(pl.begin(), next(pl.begin(), 2));
    EXPECT_EQ(8, pl.size());
    EXPECT_EQ(2, pl.front());
    EXPECT_EQ(9, pl.back());
    pl.erase(prev(pl.end(), 3), pl.end());
    EXPECT_EQ(5, pl.size());
    EXPECT_EQ(2, pl.front());
    EXPECT_EQ(6, pl.back());
    pl.erase(prev(pl.end()));
    EXPECT_EQ(4, pl.size());
    EXPECT_EQ(2, pl.front());
    EXPECT_EQ(5, pl.back());
    pl.erase(pl.begin());
    EXPECT_EQ(3, pl.size());
    EXPECT_EQ(3, pl.front());
    EXPECT_EQ(5, pl.back());
}

TEST(unit_bounded_vector, swap)
{
    bounded_vector<int,10> pl1({0,2,4});
    bounded_vector<int,10> pl2({1,3,5});
    swap(pl1, pl2);
    EXPECT_EQ(1, pl1.front());
    EXPECT_EQ(5, pl1.back());
    EXPECT_EQ(0, pl2.front());
    EXPECT_EQ(4, pl2.back());
    pl1.swap(pl2);
    EXPECT_EQ(0, pl1.front());
    EXPECT_EQ(4, pl1.back());
    EXPECT_EQ(1, pl2.front());
    EXPECT_EQ(5, pl2.back());
}

TEST(unit_bounded_vector, emplace)
{
    bounded_vector<int,10> pl;
    pl.emplace_back(2);
    EXPECT_EQ(1, pl.size());
    EXPECT_EQ(2, pl.front());
    EXPECT_EQ(2, pl.back());
    auto i = pl.emplace(prev(pl.end()), 3);
    EXPECT_EQ(2, pl.size());
    EXPECT_EQ(3, pl.front());
    EXPECT_EQ(2, pl.back());
    EXPECT_EQ(3, *i);
}

TEST(unit_bounded_vector, operators)
{
    bounded_vector<int,10> pl1{1,2,3};
    bounded_vector<int,10> pl2{1,2,4};
    EXPECT_TRUE(pl1 == pl1);
    EXPECT_FALSE(pl1 != pl1);
    EXPECT_TRUE(pl1 != pl2);
    EXPECT_FALSE(pl1 == pl2);
    EXPECT_TRUE(pl1 < pl2);
    EXPECT_TRUE(pl2 > pl1);
    EXPECT_TRUE(pl1 <= pl2);
    EXPECT_TRUE(pl1 <= pl1);
    EXPECT_TRUE(pl2 >= pl1);
    EXPECT_TRUE(pl1 >= pl1);
}

TEST(unit_bounded_vector, indexing)
{
    bounded_vector<int,10> pl{1, 2, 3};
    EXPECT_EQ(1, pl[0]);
    EXPECT_EQ(2, pl[1]);
    EXPECT_EQ(3, pl[2]);
    EXPECT_EQ(1, pl.at(0));
    EXPECT_EQ(2, pl.at(1));
    EXPECT_EQ(3, pl.at(2));
}
