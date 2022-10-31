#include "gtest/gtest.h"

#include "ptr_list.hpp"

using namespace std;
using namespace stl_ext;

TEST(unit_ptr_list, iterator)
{
    typedef ptr_list<int>::iterator iterator;
    typedef ptr_list<int>::const_iterator const_iterator;

    int x[] = {1, 2, 3, 4};
    ptr_list<int> pl = {&x[0], &x[1], &x[2], &x[3]};

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

    unique_list<int> ul{1,2,3};
    EXPECT_EQ(1,*(ul.begin().operator->()));
}

TEST(unit_ptr_list, constructor)
{
    ptr_list<int> pl1;
    EXPECT_EQ(0, pl1.size());

    ptr_list<int> pl2(4);
    EXPECT_EQ(4, pl2.size());
    EXPECT_EQ(nullptr, pl2.pfront());

    ptr_list<int> pl3(4, 4);
    EXPECT_EQ(4, pl3.size());
    EXPECT_EQ(4, pl3.front());

    ptr_list<int> pl4(pl3);
    EXPECT_EQ(4, pl4.size());
    EXPECT_EQ(4, pl4.front());

    ptr_list<int> pl5(std::move(pl4));
    EXPECT_EQ(0, pl4.size());
    EXPECT_EQ(4, pl5.size());
    EXPECT_EQ(4, pl5.front());

    ptr_list<int> pl6 = {1, 2, 3, 4};
    EXPECT_EQ(4, pl6.size());
    EXPECT_EQ(1, pl6.front());
    EXPECT_EQ(4, pl6.back());

    int x[] = {1, 2, 3, 4};
    ptr_list<int> pl7 = {&x[0], &x[1], &x[2], &x[3]};
    EXPECT_EQ(4, pl7.size());
    EXPECT_EQ(&x[0], pl7.pfront());
    EXPECT_EQ(1, pl7.front());
    EXPECT_EQ(4, pl7.back());

    ptr_list<int> pl8{1,2,3,4};
    ptr_list<int> pl9(pl8.begin(), pl8.end());
    EXPECT_EQ(4, pl9.size());
    EXPECT_EQ(1, pl9.front());
    EXPECT_EQ(4, pl9.back());

    ptr_list<int> pl10(pl8.pbegin(), pl8.pend());
    EXPECT_EQ(4, pl10.size());
    EXPECT_EQ(1, pl10.front());
    EXPECT_EQ(4, pl10.back());
}

TEST(unit_ptr_list, assign)
{
    ptr_list<int> pl3;
    pl3.assign(4, 4);
    EXPECT_EQ(4, pl3.size());
    EXPECT_EQ(4, pl3.front());

    ptr_list<int> pl4;
    pl4 = pl3;
    EXPECT_EQ(4, pl4.size());
    EXPECT_EQ(4, pl4.front());

    ptr_list<int> pl5;
    pl5 = std::move(pl4);
    EXPECT_EQ(0, pl4.size());
    EXPECT_EQ(4, pl5.size());
    EXPECT_EQ(4, pl5.front());

    ptr_list<int> pl6;
    pl6 = {1, 2, 3, 4};
    EXPECT_EQ(4, pl6.size());
    EXPECT_EQ(1, pl6.front());
    EXPECT_EQ(4, pl6.back());

    int x[] = {1, 2, 3, 4};
    ptr_list<int> pl7;
    pl7 = {&x[0], &x[1], &x[2], &x[3]};
    EXPECT_EQ(4, pl7.size());
    EXPECT_EQ(&x[0], pl7.pfront());
    EXPECT_EQ(1, pl7.front());
    EXPECT_EQ(4, pl7.back());

    ptr_list<int> pl8{1,2,3,4};
    ptr_list<int> pl9;
    pl9.assign(pl8.begin(), pl8.end());
    EXPECT_EQ(4, pl9.size());
    EXPECT_EQ(1, pl9.front());
    EXPECT_EQ(4, pl9.back());

    ptr_list<int> pl10;
    pl10.assign(pl8.pbegin(), pl8.pend());
    EXPECT_EQ(4, pl10.size());
    EXPECT_EQ(1, pl10.front());
    EXPECT_EQ(4, pl10.back());
}

TEST(unit_ptr_list, begin_end)
{
    int x[] = {1, 2, 3, 4};
    ptr_list<int> pl = {&x[0], &x[1], &x[2], &x[3]};

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

TEST(unit_ptr_list, front_back)
{
    int x[] = {1, 2, 3, 4};
    ptr_list<int> pl = {&x[0], &x[1], &x[2], &x[3]};

    EXPECT_EQ(1, pl.front());
    EXPECT_EQ(4, pl.back());

    EXPECT_EQ(x  , pl.pfront());
    EXPECT_EQ(x+3, pl.pback());
}

TEST(unit_ptr_list, empty)
{
    int x[] = {1, 2, 3, 4};
    ptr_list<int> pl;
    EXPECT_TRUE(pl.empty());
    pl = {&x[0], &x[1], &x[2], &x[3]};
    EXPECT_FALSE(pl.empty());
}

TEST(unit_ptr_list, push_pop)
{
    shared_list<int> sl;
    EXPECT_EQ(0, sl.size());

    sl.push_front(1);
    EXPECT_EQ(1, sl.size());
    EXPECT_EQ(1, sl.front());
    shared_ptr<int> sp1(new int(2));
    sl.push_front(sp1);
    EXPECT_EQ(2, sl.size());
    EXPECT_EQ(2, sl.front());
    sl.push_front(shared_ptr<int>(new int(3)));
    EXPECT_EQ(3, sl.size());
    EXPECT_EQ(3, sl.front());
    sl.push_front(new int(4));
    EXPECT_EQ(4, sl.size());
    EXPECT_EQ(4, sl.front());
    sl.pop_front();
    EXPECT_EQ(3, sl.size());
    EXPECT_EQ(3, sl.front());

    EXPECT_EQ(1, sl.back());
    sl.push_back(1);
    EXPECT_EQ(4, sl.size());
    EXPECT_EQ(1, sl.back());
    shared_ptr<int> sp2(new int (2));
    sl.push_back(sp2);
    EXPECT_EQ(5, sl.size());
    EXPECT_EQ(2, sl.back());
    sl.push_back(shared_ptr<int>(new int(3)));
    EXPECT_EQ(6, sl.size());
    EXPECT_EQ(3, sl.back());
    sl.push_back(new int(4));
    EXPECT_EQ(7, sl.size());
    EXPECT_EQ(4, sl.back());
    sl.pop_back();
    EXPECT_EQ(6, sl.size());
    EXPECT_EQ(3, sl.back());
}

TEST(unit_ptr_list, resize)
{
    ptr_list<int> pl;
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

TEST(unit_ptr_list, insert)
{
    shared_list<int> sl;
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

TEST(unit_ptr_list, erase)
{
    ptr_list<int> pl({0,1,2,3,4,5,6,7,8,9});
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

TEST(unit_ptr_list, swap)
{
    ptr_list<int> pl1({0,2,4});
    ptr_list<int> pl2({1,3,5});
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

TEST(unit_ptr_list, emplace)
{
    ptr_list<int> pl;
    pl.emplace_front(1);
    EXPECT_EQ(1, pl.size());
    EXPECT_EQ(1, pl.front());
    pl.emplace_back(2);
    EXPECT_EQ(2, pl.size());
    EXPECT_EQ(1, pl.front());
    EXPECT_EQ(2, pl.back());
    auto i = pl.emplace(prev(pl.end()), 3);
    EXPECT_EQ(3, pl.size());
    EXPECT_EQ(1, pl.front());
    EXPECT_EQ(2, pl.back());
    EXPECT_EQ(3, *i);
}

TEST(unit_ptr_list, operators)
{
    ptr_list<int> pl1{1,2,3};
    ptr_list<int> pl2{1,2,4};
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

TEST(unit_ptr_list, splice)
{
    ptr_list<int> pl1{1,2,3};
    ptr_list<int> pl2{4,5};
    pl1.splice(next(pl1.begin()), pl2);
    EXPECT_EQ(5, pl1.size());
    EXPECT_EQ(0, pl2.size());
    EXPECT_EQ(1, pl1.front());
    EXPECT_EQ(4, *next(pl1.begin()));
    pl1.splice(pl1.begin(), ptr_list<int>{-1});
    EXPECT_EQ(6, pl1.size());
    EXPECT_EQ(-1, pl1.front());
    ptr_list<int> pl3{7,8,9};
    pl1.splice(pl1.begin(), pl3, prev(pl3.end()));
    EXPECT_EQ(7, pl1.size());
    EXPECT_EQ(2, pl3.size());
    EXPECT_EQ(9, pl1.front());
    EXPECT_EQ(7, pl3.front());
    EXPECT_EQ(8, pl3.back());
    ptr_list<int> pl4{6,9,1};
    pl1.splice(pl1.begin(), pl4, pl4.begin(), prev(pl4.end()));
    EXPECT_EQ(9, pl1.size());
    EXPECT_EQ(1, pl4.size());
    EXPECT_EQ(6, pl1.front());
    EXPECT_EQ(1, pl4.front());
}

TEST(unit_ptr_list, remove)
{
    ptr_list<int> pl{0,1,2,3,4,5,6,7,8,9};
    pl.remove(9);
    EXPECT_EQ(9, pl.size());
    EXPECT_EQ(0, pl.front());
    EXPECT_EQ(8, pl.back());
    pl.remove_if([](int x){return !(x%2);});
    EXPECT_EQ(4, pl.size());
    EXPECT_EQ(1, pl.front());
    EXPECT_EQ(7, pl.back());
}

TEST(unit_ptr_list, unique)
{
    ptr_list<int> pl{0,0,0,0,1,1,3,3,3,3};
    pl.unique();
    EXPECT_EQ(3, pl.size());
    EXPECT_EQ(0, pl.front());
    EXPECT_EQ(3, pl.back());
    pl.unique([](int x, int y){return (x%2) == (y%2);});
    EXPECT_EQ(2, pl.size());
    EXPECT_EQ(0, pl.front());
    EXPECT_EQ(1, pl.back());
}

TEST(unit_ptr_list, merge)
{
    ptr_list<int> pl1{0,5,8};
    ptr_list<int> pl2{1,9};
    pl1.merge(pl2);
    EXPECT_EQ(5, pl1.size());
    EXPECT_EQ(0, pl2.size());
    EXPECT_EQ(0, pl1.front());
    EXPECT_EQ(9, pl1.back());
    pl1.merge(ptr_list<int>{6,7});
    EXPECT_EQ(7, pl1.size());
    EXPECT_EQ(7, *prev(pl1.end(),3));
    pl1.reverse();
    ptr_list<int> pl3{2};
    pl1.merge(pl3, greater<int>());
    EXPECT_EQ(8, pl1.size());
    EXPECT_EQ(0, pl3.size());
    EXPECT_EQ(2, *prev(pl1.end(),3));
    pl1.merge(ptr_list<int>{4,3}, greater<int>());
    EXPECT_EQ(10, pl1.size());
    EXPECT_EQ(3, *prev(pl1.end(),4));
}

TEST(unit_ptr_list, sort)
{
    ptr_list<int> pl{0,5,9,7,3};
    pl.sort();
    EXPECT_EQ(5,pl.size());
    EXPECT_EQ(0,pl.front());
    EXPECT_EQ(9,pl.back());
    pl.sort(greater<int>());
    EXPECT_EQ(5,pl.size());
    EXPECT_EQ(9,pl.front());
    EXPECT_EQ(0,pl.back());
}

TEST(unit_ptr_list, reverse)
{
    ptr_list<int> pl{0,4,5,1};
    pl.reverse();
    EXPECT_EQ(4,pl.size());
    EXPECT_EQ(1,pl.front());
    EXPECT_EQ(0,pl.back());
}
