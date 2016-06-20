#include "gtest/gtest.h"

#include "global_ptr.hpp"

using namespace stl_ext;

TEST(unit_global_ptr, constructor)
{
    global_ptr<int> gp1;
    EXPECT_EQ(nullptr, gp1.get());

    int *p1 = new int();
    global_ptr<int> gp2(p1);
    EXPECT_EQ(p1, gp2.get());

    global_ptr<int> gp3(gp2);
    EXPECT_EQ(p1, gp3.get());

    global_ptr<int> gp4(std::move(gp3));
    EXPECT_EQ(nullptr, gp3.get());
    EXPECT_EQ(p1, gp4.get());

    int *p2 = new int();
    gp2.set(p2);
    EXPECT_EQ(p2, gp2.get());
    EXPECT_EQ(nullptr, gp3.get());
    EXPECT_EQ(p2, gp4.get());
}

TEST(unit_global_ptr, assignment)
{
    int *p1 = new int();
    global_ptr<int> gp2(p1);
    EXPECT_EQ(p1, gp2.get());

    global_ptr<int> gp3;
    gp3 = gp2;
    EXPECT_EQ(p1, gp3.get());

    global_ptr<int> gp4;
    gp4 = std::move(gp3);
    EXPECT_EQ(nullptr, gp3.get());
    EXPECT_EQ(p1, gp4.get());

    int *p2 = new int();
    gp2.set(p2);
    EXPECT_EQ(p2, gp2.get());
    EXPECT_EQ(nullptr, gp3.get());
    EXPECT_EQ(p2, gp4.get());
}

TEST(unit_global_ptr, swap)
{
    int *p1 = new int();
    global_ptr<int> gp1(p1);
    EXPECT_EQ(p1, gp1.get());

    int *p2 = new int();
    global_ptr<int> gp2(p2);
    EXPECT_EQ(p2, gp2.get());

    swap(gp1, gp2);
    EXPECT_EQ(p2, gp1.get());
    EXPECT_EQ(p1, gp2.get());

    gp1.swap(gp2);
    EXPECT_EQ(p1, gp1.get());
    EXPECT_EQ(p2, gp2.get());
}

TEST(unit_global_ptr, use_count)
{
    int *p1 = new int();
    global_ptr<int> gp1(p1);
    global_ptr<int> gp2(gp1);
    global_ptr<int> gp3(gp1);
    EXPECT_EQ(3, gp1.use_count());
    EXPECT_EQ(3, gp2.use_count());
    EXPECT_EQ(3, gp3.use_count());
}

TEST(unit_global_ptr, unique)
{
    int *p1 = new int();
    global_ptr<int> gp1(p1);
    EXPECT_TRUE(gp1.unique());
    global_ptr<int> gp2(gp1);
    global_ptr<int> gp3(gp1);
    EXPECT_FALSE(gp1.unique());
}

TEST(unit_global_ptr, reset)
{
    int *p1 = new int();
    global_ptr<int> gp1;
    EXPECT_EQ(nullptr, gp1.get());
    gp1.reset(p1);
    global_ptr<int> gp2(gp1);
    EXPECT_EQ(p1, gp1.get());
    EXPECT_EQ(p1, gp2.get());
    gp1.reset();
    EXPECT_EQ(nullptr, gp1.get());
    EXPECT_EQ(p1, gp2.get());
}

TEST(unit_global_ptr, access)
{
    int *p1 = new int();
    global_ptr<int> gp1(p1);
    global_ptr<int> gp2(gp1);
    EXPECT_EQ(p1, gp1.get());
    EXPECT_EQ(p1, gp2.get());
    int *p2 = new int();
    gp1.set(p2);
    EXPECT_EQ(p2, gp1.get());
    EXPECT_EQ(p2, gp2.get());
}

TEST(unit_global_ptr, dereference)
{
    int *p1 = new int(42);
    global_ptr<int> gp1(p1);
    global_ptr<int> gp2(gp1);
    EXPECT_EQ(42, *gp1);
    EXPECT_EQ(42, *gp2);
}

TEST(unit_global_ptr, bool)
{
    int *p1 = new int(42);
    global_ptr<int> gp1;
    EXPECT_FALSE(gp1);
    gp1.set(p1);
    EXPECT_TRUE(gp1);
}
