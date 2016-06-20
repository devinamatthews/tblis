#include "gtest/gtest.h"

#include "ptr_vector.hpp"

using namespace std;
using namespace stl_ext;

TEST(unit_ptr_vector, iterator)
{
    typedef ptr_vector<int>::iterator iterator;
    typedef ptr_vector<int>::const_iterator const_iterator;

    int x[] = {1, 2, 3, 4};
    ptr_vector<int> pl = {&x[0], &x[1], &x[2], &x[3]};

    iterator i1 = pl.begin();
    EXPECT_EQ(1, *i1);
    iterator i2(i1);
    EXPECT_EQ(1, *i2);
    const_iterator i3(i1);
    EXPECT_EQ(1, *i3);
    iterator i4 = ++i1;
    EXPECT_EQ(2, *i1);
    EXPECT_EQ(2, *i4);
    i2 = i1;
    EXPECT_EQ(2, *i2);
    i3 = i1;
    EXPECT_EQ(2, *i3);
    i4 = --i1;
    EXPECT_EQ(1, *i1);
    EXPECT_EQ(1, *i4);
    i4 = i1++;
    EXPECT_EQ(2, *i1);
    EXPECT_EQ(1, *i4);
    i4 = i1--;
    EXPECT_EQ(1, *i1);
    EXPECT_EQ(2, *i4);
    swap(i1, i4);
    EXPECT_EQ(2, *i1);
    EXPECT_EQ(1, *i4);
    EXPECT_TRUE(i1 != i4);
    EXPECT_FALSE(i1 == i4);
    i4 = i1;
    EXPECT_FALSE(i1 != i4);
    EXPECT_TRUE(i1 == i4);
    i3 = pl.begin();
    EXPECT_TRUE(i1 != i3);
    EXPECT_FALSE(i1 == i3);
    i3 = i1;
    EXPECT_FALSE(i1 != i3);
    EXPECT_TRUE(i1 == i3);
    EXPECT_EQ(&x[1], i1.operator->());
    EXPECT_EQ(&x[1], *i1.base());

    i1 = pl.begin();
    EXPECT_EQ(pl.end(), i1+4);
    EXPECT_EQ(pl.end(), 4+i1);
    i1 += 4;
    EXPECT_EQ(pl.end(), i1);
    i1 = pl.end();
    EXPECT_EQ(pl.begin(), i1-4);
    EXPECT_EQ(4, i1-pl.begin());
    EXPECT_EQ(4, i1-pl.cbegin());
    i1 -= 4;
    EXPECT_EQ(pl.begin(), i1);
    EXPECT_EQ(1, i1[0]);
    EXPECT_EQ(2, i1[1]);
    EXPECT_EQ(3, i1[2]);
    EXPECT_EQ(4, pl.cbegin()[3]);
    i2 = i1+1;
    EXPECT_TRUE(i1 < i2);
    EXPECT_TRUE(i2 > i1);
    EXPECT_TRUE(i1 <= i2);
    EXPECT_TRUE(i1 <= i1);
    EXPECT_TRUE(i2 >= i1);
    EXPECT_TRUE(i1 >= i1);
    i3 = i1+1;
    EXPECT_TRUE(i1 < i3);
    EXPECT_TRUE(i3 > i1);
    EXPECT_TRUE(i1 <= i3);
    EXPECT_TRUE(i1 <= i1);
    EXPECT_TRUE(i3 >= i1);
    EXPECT_TRUE(i1 >= i1);

    unique_vector<int> ul{1,2,3};
    EXPECT_EQ(1,*(ul.begin().operator->()));
}

TEST(unit_ptr_vector, constructor)
{
    ptr_vector<int> pl1;
    EXPECT_EQ(0, pl1.size());

    ptr_vector<int> pl2(4);
    EXPECT_EQ(4, pl2.size());
    EXPECT_EQ(nullptr, pl2.pfront());

    ptr_vector<int> pl3(4, 4);
    EXPECT_EQ(4, pl3.size());
    EXPECT_EQ(4, pl3.front());

    ptr_vector<int> pl4(pl3);
    EXPECT_EQ(4, pl4.size());
    EXPECT_EQ(4, pl4.front());

    ptr_vector<int> pl5(std::move(pl4));
    EXPECT_EQ(0, pl4.size());
    EXPECT_EQ(4, pl5.size());
    EXPECT_EQ(4, pl5.front());

    ptr_vector<int> pl6 = {1, 2, 3, 4};
    EXPECT_EQ(4, pl6.size());
    EXPECT_EQ(1, pl6.front());
    EXPECT_EQ(4, pl6.back());

    int x[] = {1, 2, 3, 4};
    ptr_vector<int> pl7 = {&x[0], &x[1], &x[2], &x[3]};
    EXPECT_EQ(4, pl7.size());
    EXPECT_EQ(&x[0], pl7.pfront());
    EXPECT_EQ(1, pl7.front());
    EXPECT_EQ(4, pl7.back());

    ptr_vector<int> pl8{1,2,3,4};
    ptr_vector<int> pl9(pl8.begin(), pl8.end());
    EXPECT_EQ(4, pl9.size());
    EXPECT_EQ(1, pl9.front());
    EXPECT_EQ(4, pl9.back());

    ptr_vector<int> pl10(pl8.pbegin(), pl8.pend());
    EXPECT_EQ(4, pl10.size());
    EXPECT_EQ(1, pl10.front());
    EXPECT_EQ(4, pl10.back());
}

TEST(unit_ptr_vector, assign)
{
    ptr_vector<int> pl3;
    pl3.assign(4, 4);
    EXPECT_EQ(4, pl3.size());
    EXPECT_EQ(4, pl3.front());

    ptr_vector<int> pl4;
    pl4 = pl3;
    EXPECT_EQ(4, pl4.size());
    EXPECT_EQ(4, pl4.front());

    ptr_vector<int> pl5;
    pl5 = std::move(pl4);
    EXPECT_EQ(0, pl4.size());
    EXPECT_EQ(4, pl5.size());
    EXPECT_EQ(4, pl5.front());

    ptr_vector<int> pl6;
    pl6 = {1, 2, 3, 4};
    EXPECT_EQ(4, pl6.size());
    EXPECT_EQ(1, pl6.front());
    EXPECT_EQ(4, pl6.back());

    int x[] = {1, 2, 3, 4};
    ptr_vector<int> pl7;
    pl7 = {&x[0], &x[1], &x[2], &x[3]};
    EXPECT_EQ(4, pl7.size());
    EXPECT_EQ(&x[0], pl7.pfront());
    EXPECT_EQ(1, pl7.front());
    EXPECT_EQ(4, pl7.back());

    ptr_vector<int> pl8{1,2,3,4};
    ptr_vector<int> pl9;
    pl9.assign(pl8.begin(), pl8.end());
    EXPECT_EQ(4, pl9.size());
    EXPECT_EQ(1, pl9.front());
    EXPECT_EQ(4, pl9.back());

    ptr_vector<int> pl10;
    pl10.assign(pl8.pbegin(), pl8.pend());
    EXPECT_EQ(4, pl10.size());
    EXPECT_EQ(1, pl10.front());
    EXPECT_EQ(4, pl10.back());
}

TEST(unit_ptr_vector, begin_end)
{
    int x[] = {1, 2, 3, 4};
    ptr_vector<int> pl = {&x[0], &x[1], &x[2], &x[3]};

    EXPECT_EQ(x  , &*pl.begin());
    EXPECT_EQ(x+3, &*prev(pl.end()));
    EXPECT_EQ(x+3, &*pl.rbegin());
    EXPECT_EQ(x  , &*prev(pl.rend()));
    EXPECT_EQ(x  , &*pl.cbegin());
    EXPECT_EQ(x+3, &*prev(pl.cend()));
    EXPECT_EQ(x+3, &*pl.crbegin());
    EXPECT_EQ(x  , &*prev(pl.crend()));

    EXPECT_EQ(x  , *pl.pbegin());
    EXPECT_EQ(x+3, *prev(pl.pend()));
    EXPECT_EQ(x+3, *pl.rpbegin());
    EXPECT_EQ(x  , *prev(pl.rpend()));
    EXPECT_EQ(x  , *pl.cpbegin());
    EXPECT_EQ(x+3, *prev(pl.cpend()));
    EXPECT_EQ(x+3, *pl.crpbegin());
    EXPECT_EQ(x  , *prev(pl.crpend()));
}

TEST(unit_ptr_vector, front_back)
{
    int x[] = {1, 2, 3, 4};
    ptr_vector<int> pl = {&x[0], &x[1], &x[2], &x[3]};

    EXPECT_EQ(1, pl.front());
    EXPECT_EQ(4, pl.back());

    EXPECT_EQ(x  , pl.pfront());
    EXPECT_EQ(x+3, pl.pback());
}

TEST(unit_ptr_vector, empty)
{
    int x[] = {1, 2, 3, 4};
    ptr_vector<int> pl;
    EXPECT_TRUE(pl.empty());
    pl = {&x[0], &x[1], &x[2], &x[3]};
    EXPECT_FALSE(pl.empty());
}

TEST(unit_ptr_vector, push_pop)
{
    shared_vector<int> sl;
    EXPECT_EQ(0, sl.size());

    sl.push_back(1);
    EXPECT_EQ(1, sl.size());
    EXPECT_EQ(1, sl.back());
    shared_ptr<int> sp2(new int (2));
    sl.push_back(sp2);
    EXPECT_EQ(2, sl.size());
    EXPECT_EQ(2, sl.back());
    sl.push_back(shared_ptr<int>(new int(3)));
    EXPECT_EQ(3, sl.size());
    EXPECT_EQ(3, sl.back());
    sl.push_back(new int(4));
    EXPECT_EQ(4, sl.size());
    EXPECT_EQ(4, sl.back());
    sl.pop_back();
    EXPECT_EQ(3, sl.size());
    EXPECT_EQ(3, sl.back());
}

TEST(unit_ptr_vector, resize)
{
    ptr_vector<int> pl;
    EXPECT_EQ(0, pl.size());
    pl.resize(3);
    EXPECT_EQ(3, pl.size());
    EXPECT_EQ(0, pl.pfront());
    EXPECT_EQ(0, pl.pback());
    pl.resize(4, 4);
    EXPECT_EQ(4, pl.size());
    EXPECT_EQ(0, pl.pfront());
    EXPECT_EQ(4, pl.back());
}

TEST(unit_ptr_vector, insert)
{
    shared_vector<int> sl;
    EXPECT_EQ(0, sl.size());
    sl.insert(sl.begin(), 3);
    EXPECT_EQ(1, sl.size());
    EXPECT_EQ(3, sl.front());
    EXPECT_EQ(3, sl.back());
    shared_ptr<int> sp(new int(1));
    sl.insert(sl.begin(), sp);
    EXPECT_EQ(2, sl.size());
    EXPECT_EQ(1, sl.front());
    EXPECT_EQ(3, sl.back());
    auto i = sl.insert(sl.end(), shared_ptr<int>(new int(2)));
    EXPECT_EQ(3, sl.size());
    EXPECT_EQ(1, sl.front());
    EXPECT_EQ(2, sl.back());
    i = sl.insert(i, new int(6));
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
    i = sl.insert(sl.end(), {shared_ptr<int>(new int(7))});
    EXPECT_EQ(6, sl.size());
    EXPECT_EQ(7, *i);
    EXPECT_EQ(7, sl.back());
}

TEST(unit_ptr_vector, erase)
{
    ptr_vector<int> pl({0,1,2,3,4,5,6,7,8,9});
    EXPECT_EQ(10, pl.size());
    EXPECT_EQ(0, pl.front());
    EXPECT_EQ(9, pl.back());
    pl.erase(pl.begin(), next(pl.begin(), 2));
    EXPECT_EQ(8, pl.size());
    EXPECT_EQ(2, pl.front());
    EXPECT_EQ(9, pl.back());
    pl.perase(prev(pl.pend(), 3), pl.pend());
    EXPECT_EQ(5, pl.size());
    EXPECT_EQ(2, pl.front());
    EXPECT_EQ(6, pl.back());
    pl.erase(prev(pl.end()));
    EXPECT_EQ(4, pl.size());
    EXPECT_EQ(2, pl.front());
    EXPECT_EQ(5, pl.back());
    pl.perase(pl.pbegin());
    EXPECT_EQ(3, pl.size());
    EXPECT_EQ(3, pl.front());
    EXPECT_EQ(5, pl.back());
}

TEST(unit_ptr_vector, swap)
{
    ptr_vector<int> pl1({0,2,4});
    ptr_vector<int> pl2({1,3,5});
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

TEST(unit_ptr_vector, emplace)
{
    ptr_vector<int> pl;
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

TEST(unit_ptr_vector, operators)
{
    ptr_vector<int> pl1{1,2,3};
    ptr_vector<int> pl2{1,2,4};
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

TEST(unit_ptr_vector, indexing)
{
    int x[] = {1,2,3};
    ptr_vector<int> pl{&x[0], &x[1], &x[2]};
    EXPECT_EQ(1, pl[0]);
    EXPECT_EQ(2, pl[1]);
    EXPECT_EQ(3, pl[2]);
    EXPECT_EQ(1, pl.at(0));
    EXPECT_EQ(2, pl.at(1));
    EXPECT_EQ(3, pl.at(2));
    EXPECT_EQ(x  , pl.ptr(0));
    EXPECT_EQ(x+1, pl.ptr(1));
    EXPECT_EQ(x+2, pl.ptr(2));
}
