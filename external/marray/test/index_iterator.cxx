#include "index_iterator.hpp"

#include "gtest/gtest.h"

using namespace std;
using namespace MArray;

TEST(miterator, next)
{
    len_type off1, off2;

    index_iterator<0,1> m1(vector<int>{}, vector<int>{});

    off1 = 0;
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_FALSE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);

    index_iterator<1,2> m2(vector<int>{5}, vector<int>{1}, vector<int>{2});

    off1 = 0;
    off2 = 0;
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(1, off1);
    EXPECT_EQ(2, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(2, off1);
    EXPECT_EQ(4, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(3, off1);
    EXPECT_EQ(6, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(4, off1);
    EXPECT_EQ(8, off2);
    EXPECT_FALSE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(1, off1);
    EXPECT_EQ(2, off2);

    index_iterator<2,1> m3(vector<int>{2,3}, vector<int>{1,2});

    off1 = 0;
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(1, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(2, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(5, off1);
    EXPECT_FALSE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(1, off1);

    index_iterator<1,1> m4(vector<int>{0}, vector<int>{1});

    off1 = 0;
    EXPECT_FALSE(m4.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_FALSE(m4.next(off1));
    EXPECT_EQ(0, off1);
}

TEST(viterator, next)
{
    len_type off1, off2;

    index_iterator<DYNAMIC,1> m1(vector<int>{}, vector<int>{});

    off1 = 0;
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_FALSE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);

    index_iterator<DYNAMIC,2> m2(vector<int>{5}, vector<int>{1}, vector<int>{2});

    off1 = 0;
    off2 = 0;
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(1, off1);
    EXPECT_EQ(2, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(2, off1);
    EXPECT_EQ(4, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(3, off1);
    EXPECT_EQ(6, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(4, off1);
    EXPECT_EQ(8, off2);
    EXPECT_FALSE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(1, off1);
    EXPECT_EQ(2, off2);

    index_iterator<DYNAMIC,1> m3(vector<int>{2,3}, vector<int>{1,2});

    off1 = 0;
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(1, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(2, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(5, off1);
    EXPECT_FALSE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(1, off1);

    index_iterator<DYNAMIC,1> m4(vector<int>{0}, vector<int>{1});

    off1 = 0;
    EXPECT_FALSE(m4.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_FALSE(m4.next(off1));
    EXPECT_EQ(0, off1);
}

TEST(miterator, reset)
{
    len_type off1;

    index_iterator<2,1> m1(vector<int>{2,3}, vector<int>{1,2});

    off1 = 0;
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(1, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(2, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(3, off1);

    off1 = 0;
    m1.reset();
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(1, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(2, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(5, off1);

    index_iterator<1,1> m4(vector<int>{0}, vector<int>{1});

    off1 = 0;
    EXPECT_FALSE(m4.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_FALSE(m4.next(off1));
    EXPECT_EQ(0, off1);
}

TEST(viterator, reset)
{
    len_type off1;

    index_iterator<DYNAMIC,1> m1(vector<int>{2,3}, vector<int>{1,2});

    off1 = 0;
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(1, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(2, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(3, off1);

    off1 = 0;
    m1.reset();
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(1, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(2, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(5, off1);

    index_iterator<DYNAMIC,1> m4(vector<int>{0}, vector<int>{1});

    off1 = 0;
    EXPECT_FALSE(m4.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_FALSE(m4.next(off1));
    EXPECT_EQ(0, off1);
}

TEST(miterator, position)
{
    len_type off1;

    index_iterator<2,1> m1(vector<int>{2,3}, vector<int>{1,2});

    off1 = 0;
    EXPECT_TRUE(m1.position(3, off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(5, off1);

    off1 = 0;
    EXPECT_TRUE(m1.position(vector<int>{1,1}, off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(5, off1);

    EXPECT_EQ((array<len_type,2>{1,2}), m1.position());
}

TEST(viterator, position)
{
    len_type off1;

    index_iterator<DYNAMIC,1> m1(vector<int>{2,3}, vector<int>{1,2});

    off1 = 0;
    EXPECT_TRUE(m1.position(3, off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(5, off1);

    off1 = 0;
    EXPECT_TRUE(m1.position(vector<int>{1,1}, off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(5, off1);

    EXPECT_EQ((len_vector{1,2}), m1.position());
}

TEST(miterator, assign)
{
    len_type off1;

    index_iterator<2,1> m1(vector<int>{2,3}, vector<int>{1,2});
    index_iterator<2,1> m2(vector<int>{2,3}, vector<int>{1,3});

    off1 = 0;
    m1.position(vector<int>{1,1}, off1);
    EXPECT_EQ(3, off1);

    off1 = 0;
    m2.position(vector<int>{1,1}, off1);
    EXPECT_EQ(4, off1);

    m1 = m2;

    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(6, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(7, off1);
}

TEST(viterator, assign)
{
    len_type off1;

    index_iterator<DYNAMIC,1> m1(vector<int>{2,3}, vector<int>{1,2});
    index_iterator<DYNAMIC,1> m2(vector<int>{2,3}, vector<int>{1,3});

    off1 = 0;
    m1.position(vector<int>{1,1}, off1);
    EXPECT_EQ(3, off1);

    off1 = 0;
    m2.position(vector<int>{1,1}, off1);
    EXPECT_EQ(4, off1);

    m1 = m2;

    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(6, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(7, off1);
}

TEST(miterator, swap)
{
    len_type off1, off2;

    index_iterator<2,1> m1(vector<int>{2,3}, vector<int>{1,2});
    index_iterator<2,1> m2(vector<int>{2,3}, vector<int>{1,3});

    off1 = 0;
    m1.position(vector<int>{1,1}, off1);
    EXPECT_EQ(3, off1);

    off2 = 0;
    m2.position(vector<int>{1,1}, off2);
    EXPECT_EQ(4, off2);

    m1.swap(m2);

    EXPECT_TRUE(m1.next(off2));
    EXPECT_EQ(4, off2);
    EXPECT_TRUE(m1.next(off2));
    EXPECT_EQ(6, off2);
    EXPECT_TRUE(m1.next(off2));
    EXPECT_EQ(7, off2);

    EXPECT_TRUE(m2.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m2.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m2.next(off1));
    EXPECT_EQ(5, off1);
}

TEST(viterator, swap)
{
    len_type off1, off2;

    index_iterator<DYNAMIC,1> m1(vector<int>{2,3}, vector<int>{1,2});
    index_iterator<DYNAMIC,1> m2(vector<int>{2,3}, vector<int>{1,3});

    off1 = 0;
    m1.position(vector<int>{1,1}, off1);
    EXPECT_EQ(3, off1);

    off2 = 0;
    m2.position(vector<int>{1,1}, off2);
    EXPECT_EQ(4, off2);

    m1.swap(m2);

    EXPECT_TRUE(m1.next(off2));
    EXPECT_EQ(4, off2);
    EXPECT_TRUE(m1.next(off2));
    EXPECT_EQ(6, off2);
    EXPECT_TRUE(m1.next(off2));
    EXPECT_EQ(7, off2);

    EXPECT_TRUE(m2.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m2.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m2.next(off1));
    EXPECT_EQ(5, off1);
}

TEST(miterator, make_iterator)
{
    len_type off1, off2;

    auto m1 = make_iterator(array<len_type,0>{}, array<stride_type,0>{});

    off1 = 0;
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_FALSE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);

    auto m2 = make_iterator(array<len_type,1>{5}, array<stride_type,1>{1}, array<stride_type,1>{2});

    off1 = 0;
    off2 = 0;
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(1, off1);
    EXPECT_EQ(2, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(2, off1);
    EXPECT_EQ(4, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(3, off1);
    EXPECT_EQ(6, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(4, off1);
    EXPECT_EQ(8, off2);
    EXPECT_FALSE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(1, off1);
    EXPECT_EQ(2, off2);

    auto m3 = make_iterator(array<len_type,2>{2,3}, array<stride_type,2>{1,2});

    off1 = 0;
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(1, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(2, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(5, off1);
    EXPECT_FALSE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(1, off1);
}

TEST(viterator, make_iterator)
{
    len_type off1, off2;

    auto m1 = make_iterator(vector<len_type>{}, vector<stride_type>{});

    off1 = 0;
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_FALSE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);

    auto m2 = make_iterator(vector<len_type>{5}, vector<stride_type>{1}, vector<stride_type>{2});

    off1 = 0;
    off2 = 0;
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(1, off1);
    EXPECT_EQ(2, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(2, off1);
    EXPECT_EQ(4, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(3, off1);
    EXPECT_EQ(6, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(4, off1);
    EXPECT_EQ(8, off2);
    EXPECT_FALSE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(1, off1);
    EXPECT_EQ(2, off2);

    auto m3 = make_iterator(vector<len_type>{2,3}, vector<stride_type>{1,2});

    off1 = 0;
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(1, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(2, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(5, off1);
    EXPECT_FALSE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(1, off1);
}
