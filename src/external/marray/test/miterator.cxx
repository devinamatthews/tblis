#include "miterator.hpp"
#include "utility.hpp"

#include "gtest/gtest.h"

using namespace std;
using namespace MArray;

TEST(miterator, inc_offsets)
{
    array<int,4> strides1 = {1,4,16,64};
    array<int,4> strides2 = {64,16,4,1};
    int off1 = 0, off2 = 0;

    detail::inc_offsets(1, make_array(strides1, strides2), off1, off2);

    EXPECT_EQ(4, off1);
    EXPECT_EQ(16, off2);

    detail::inc_offsets(3, make_array(strides1, strides2), off1, off2);

    EXPECT_EQ(68, off1);
    EXPECT_EQ(17, off2);
}

TEST(miterator, dec_offsets)
{
    array<long,4> strides1 = {1,4,16,64};
    array<long,4> strides2 = {64,16,4,1};
    array<int,4> pos = {1,1,0,2};
    long off1 = 0, off2 = 0;

    detail::dec_offsets(1, pos, make_array(strides1, strides2), off1, off2);

    EXPECT_EQ(-4, off1);
    EXPECT_EQ(-16, off2);

    detail::dec_offsets(3, pos, make_array(strides1, strides2), off1, off2);

    EXPECT_EQ(-132, off1);
    EXPECT_EQ(-18, off2);
}

TEST(miterator, set_offsets)
{
    array<long,4> strides1 = {1,4,16,64};
    array<long,4> strides2 = {64,16,4,1};
    array<int,4> pos = {1,1,0,2};
    long off1 = 0, off2 = 0;

    detail::set_offsets(pos, make_array(strides1, strides2), off1, off2);

    EXPECT_EQ(133, off1);
    EXPECT_EQ(82, off2);
}

TEST(miterator, ctor)
{
    miterator<int, long, 4, 2> it1(make_array(4, 4, 4, 4),
                                   make_array(1, 4, 16, 64),
                                   make_array(64, 16, 4, 1));

    EXPECT_EQ(make_array(4, 4, 4, 4), it1.lengths());
    EXPECT_EQ(make_array(0, 0, 0, 0), it1.position());
    EXPECT_EQ(make_array(1l, 4l, 16l, 64l), it1.strides(0));
    EXPECT_EQ(make_array(64l, 16l, 4l, 1l), it1.strides(1));

    miterator<int, long, 4, 2> it2(it1);

    EXPECT_EQ(make_array(4, 4, 4, 4), it2.lengths());
    EXPECT_EQ(make_array(0, 0, 0, 0), it2.position());
    EXPECT_EQ(make_array(1l, 4l, 16l, 64l), it2.strides(0));
    EXPECT_EQ(make_array(64l, 16l, 4l, 1l), it2.strides(1));

    miterator<int, long, 4, 2> it3(move(it1));

    EXPECT_EQ(make_array(4, 4, 4, 4), it3.lengths());
    EXPECT_EQ(make_array(0, 0, 0, 0), it3.position());
    EXPECT_EQ(make_array(1l, 4l, 16l, 64l), it3.strides(0));
    EXPECT_EQ(make_array(64l, 16l, 4l, 1l), it3.strides(1));
}

TEST(miterator, assign)
{
    miterator<int, long, 4, 2> it1(make_array(4, 4, 4, 4),
                                   make_array(1, 4, 16, 64),
                                   make_array(64, 16, 4, 1));

    int off1 = 0, off2 = 0;
    it1.next(off1, off2);
    it1.next(off1, off2);

    EXPECT_EQ(make_array(4, 4, 4, 4), it1.lengths());
    EXPECT_EQ(make_array(1, 0, 0, 0), it1.position());
    EXPECT_EQ(make_array(1l, 4l, 16l, 64l), it1.strides(0));
    EXPECT_EQ(make_array(64l, 16l, 4l, 1l), it1.strides(1));

    miterator<int, long, 4, 2> it2(make_array(0, 0, 0, 0),
                                   make_array(0, 0, 0, 0),
                                   make_array(0, 0, 0, 0));
    it2 = it1;

    EXPECT_EQ(make_array(4, 4, 4, 4), it2.lengths());
    EXPECT_EQ(make_array(1, 0, 0, 0), it2.position());
    EXPECT_EQ(make_array(1l, 4l, 16l, 64l), it2.strides(0));
    EXPECT_EQ(make_array(64l, 16l, 4l, 1l), it2.strides(1));

    miterator<int, long, 4, 2> it3(make_array(0, 0, 0, 0),
                                   make_array(0, 0, 0, 0),
                                   make_array(0, 0, 0, 0));
    it3 = move(it1);

    EXPECT_EQ(make_array(4, 4, 4, 4), it3.lengths());
    EXPECT_EQ(make_array(1, 0, 0, 0), it3.position());
    EXPECT_EQ(make_array(1l, 4l, 16l, 64l), it3.strides(0));
    EXPECT_EQ(make_array(64l, 16l, 4l, 1l), it3.strides(1));
}

TEST(miterator, reset)
{
    miterator<int, long, 4, 2> it1(make_array(4, 4, 4, 4),
                                   make_array(1, 4, 16, 64),
                                   make_array(64, 16, 4, 1));

    int off1 = 0, off2 = 0;
    it1.next(off1, off2);
    it1.next(off1, off2);

    EXPECT_EQ(make_array(4, 4, 4, 4), it1.lengths());
    EXPECT_EQ(make_array(1, 0, 0, 0), it1.position());
    EXPECT_EQ(make_array(1l, 4l, 16l, 64l), it1.strides(0));
    EXPECT_EQ(make_array(64l, 16l, 4l, 1l), it1.strides(1));

    it1.reset();

    EXPECT_EQ(make_array(4, 4, 4, 4), it1.lengths());
    EXPECT_EQ(make_array(0, 0, 0, 0), it1.position());
    EXPECT_EQ(make_array(1l, 4l, 16l, 64l), it1.strides(0));
    EXPECT_EQ(make_array(64l, 16l, 4l, 1l), it1.strides(1));
}

TEST(miterator, next)
{
    miterator<int, long, 2, 2> it1(make_array(2, 2),
                                   make_array(1, 2),
                                   make_array(2, 1));

    int off1 = 0, off2 = 0;
    bool res;

    res = it1.next(off1, off2);

    EXPECT_EQ(make_array(0, 0), it1.position());
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(res);

    res = it1.next(off1, off2);

    EXPECT_EQ(make_array(1, 0), it1.position());
    EXPECT_EQ(1, off1);
    EXPECT_EQ(2, off2);
    EXPECT_TRUE(res);

    res = it1.next(off1, off2);

    EXPECT_EQ(make_array(0, 1), it1.position());
    EXPECT_EQ(2, off1);
    EXPECT_EQ(1, off2);
    EXPECT_TRUE(res);

    res = it1.next(off1, off2);

    EXPECT_EQ(make_array(1, 1), it1.position());
    EXPECT_EQ(3, off1);
    EXPECT_EQ(3, off2);
    EXPECT_TRUE(res);

    res = it1.next(off1, off2);

    EXPECT_EQ(make_array(0, 0), it1.position());
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_FALSE(res);
}

TEST(miterator, position)
{
    miterator<int, long, 4, 2> it1(make_array(4, 4, 4, 4),
                                   make_array(1, 4, 16, 64),
                                   make_array(64, 16, 4, 1));

    int off1 = 0, off2 = 0;

    it1.position(107, off1, off2);

    EXPECT_EQ(3, it1.position(0));
    EXPECT_EQ(2, it1.position(1));
    EXPECT_EQ(2, it1.position(2));
    EXPECT_EQ(1, it1.position(3));
    EXPECT_EQ(make_array(3, 2, 2, 1), it1.position());
    EXPECT_EQ(3*1+2*4+2*16+1*64, off1);
    EXPECT_EQ(3*64+2*16+2*4+1*1, off2);

    it1.position(make_array(1, 2, 2, 3), off1, off2);

    EXPECT_EQ(1, it1.position(0));
    EXPECT_EQ(2, it1.position(1));
    EXPECT_EQ(2, it1.position(2));
    EXPECT_EQ(3, it1.position(3));
    EXPECT_EQ(make_array(1, 2, 2, 3), it1.position());
    EXPECT_EQ(1*1+2*4+2*16+3*64, off1);
    EXPECT_EQ(1*64+2*16+2*4+3*1, off2);
}

TEST(miterator, dimension)
{
    miterator<int, long, 4, 2> it1(make_array(4, 4, 4, 4),
                                   make_array(1, 4, 16, 64),
                                   make_array(64, 16, 4, 1));

    EXPECT_EQ(4, it1.dimension());
}

TEST(miterator, lengths)
{
    miterator<int, long, 4, 2> it1(make_array(4, 4, 4, 4),
                                   make_array(1, 4, 16, 64),
                                   make_array(64, 16, 4, 1));

    EXPECT_EQ(4, it1.length(0));
    EXPECT_EQ(4, it1.length(1));
    EXPECT_EQ(4, it1.length(2));
    EXPECT_EQ(4, it1.length(3));
    EXPECT_EQ(make_array(4, 4, 4, 4), it1.lengths());
}

TEST(miterator, strides)
{
    miterator<int, long, 4, 2> it1(make_array(4, 4, 4, 4),
                                   make_array(1, 4, 16, 64),
                                   make_array(64, 16, 4, 1));

    EXPECT_EQ(1, it1.stride(0, 0));
    EXPECT_EQ(4, it1.stride(0, 1));
    EXPECT_EQ(16, it1.stride(0, 2));
    EXPECT_EQ(64, it1.stride(0, 3));
    EXPECT_EQ(make_array(1l, 4l, 16l, 64l), it1.strides(0));

    EXPECT_EQ(64, it1.stride(1, 0));
    EXPECT_EQ(16, it1.stride(1, 1));
    EXPECT_EQ(4, it1.stride(1, 2));
    EXPECT_EQ(1, it1.stride(1, 3));
    EXPECT_EQ(make_array(64l, 16l, 4l, 1l), it1.strides(1));
}

TEST(miterator, swap)
{
    miterator<int, long, 4> it1(make_array(4, 4, 4, 4),
                                make_array(1, 4, 16, 64));
    miterator<int, long, 4> it2(make_array(3, 3, 3, 3),
                                make_array(64, 16, 4, 1));

    swap(it1, it2);

    EXPECT_EQ(make_array(3, 3, 3, 3), it1.lengths());
    EXPECT_EQ(make_array(0, 0, 0, 0), it1.position());
    EXPECT_EQ(make_array(64l, 16l, 4l, 1l), it1.strides(0));

    EXPECT_EQ(make_array(4, 4, 4, 4), it2.lengths());
    EXPECT_EQ(make_array(0, 0, 0, 0), it2.position());
    EXPECT_EQ(make_array(1l, 4l, 16l, 64l), it2.strides(0));
}

TEST(miterator, make_iterator)
{
    auto it1 = make_iterator(make_array(4, 4, 4, 4),
                             make_array(1l, 4l, 16l, 64l),
                             make_array(64l, 16l, 4l, 1l));

    EXPECT_TRUE((is_same<miterator<int, long, 4, 2>,decltype(it1)>::value));

    EXPECT_EQ(make_array(4, 4, 4, 4), it1.lengths());
    EXPECT_EQ(make_array(0, 0, 0, 0), it1.position());
    EXPECT_EQ(make_array(1l, 4l, 16l, 64l), it1.strides(0));
    EXPECT_EQ(make_array(64l, 16l, 4l, 1l), it1.strides(1));
}
