#include <sstream>

#include "gtest/gtest.h"

#include "iostream.hpp"

#include "vector.hpp"

using namespace std;
using namespace stl_ext;

TEST(unit_iostream, printos)
{
    ostringstream oss;

    //FIXME: positional arguments
    //FIXME: precision/width in argument
    //FIXME: relative precision
    //NOT SUPPORTED: %'d, %Id, %m, %n

    //FIXME: % d
    //FIXME: %.md
    oss << printos("%d %% %4d %04d %-4d %-04d %0-4d %+d %p",
                   4, 34, 34, 34, 34, 34, 3, (void*)666);
    EXPECT_EQ("4 %   34 0034 34   34   34   +3 0x29a", oss.str());
    oss.str("");

    oss << printos("%x %X %#x %#X %u %o %#o %i", 34, 34, 34, 34, 34, 34, 34, 34);
    EXPECT_EQ("22 22 0x22 0X22 34 42 042 34", oss.str());
    oss.str("");

    oss << printos("%c %3c %s %6.4s %-3s", 'a', 'z', "hi", string("yohoho"), "k");
    EXPECT_EQ("a   z hi   yoho k  ", oss.str());
    oss.str("");

    oss << printos("%j", vector<int>{1,2,3});
    EXPECT_EQ("[1, 2, 3]", oss.str());
    oss.str("");

    //FIXME: %a %A %F?
    oss << printos("%f %g %g %G %e %E", 3.14, 3.14, 1e-8, 1e-8, 3.14, 3.14);
    EXPECT_EQ("3.140000 3.14 1e-08 1E-08 3.140000e+00 3.140000E+00", oss.str());
    oss.str("");

    //FIXME: %#g
    oss << printos("%#010.0f %+f %-6g %.3g", 314, 3.14, 3.14, 1.314);
    EXPECT_EQ("000000314. +3.140000 3.14   1.31", oss.str());
    oss.str("");
}

TEST(unit_iostream, fmt_operator)
{
    ostringstream oss;
    oss << "%d %j %f" % fmt(3, vector<int>{1,2}, 3.14);
    EXPECT_EQ("3 [1, 2] 3.140000", oss.str());
}

TEST(unit_iostream, tuple_operator)
{
    ostringstream oss;
    oss << make_tuple(3, "abc", 4.5);
    EXPECT_EQ("{3, abc, 4.5}", oss.str());
}

TEST(unit_iostream, vector_operator)
{
    ostringstream oss;
    oss << vector<int>{1,2,3,4};
    EXPECT_EQ("[1, 2, 3, 4]", oss.str());
}

TEST(unit_iostream, printToAccuracy)
{
    ostringstream oss;
    oss << printToAccuracy(3.141592654, 4e-3);
    EXPECT_EQ("3.142", oss.str());
}
