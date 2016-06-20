#include "gtest/gtest.h"

#include "iostream.hpp"
#include "string.hpp"

using namespace std;
using namespace stl_ext;

TEST(unit_string, str)
{
    EXPECT_EQ("1", str(1));
    EXPECT_EQ("1.45", str(1.45));
    EXPECT_EQ("[1, 2, 3]", str(vector<int>{1,2,3}));
    const char *cs = "hi";
    string s = "hi";
    EXPECT_STREQ("hi", str("hi"));
    EXPECT_STREQ("hi", str(cs));
    EXPECT_EQ(cs, str(cs));
    EXPECT_EQ("hi", str(s));
    EXPECT_EQ(&s, &str(s));
    EXPECT_EQ("1 1.45 [1, 2, 3]", str("%d %g %j", 1, 1.45, vector<int>{1,2,3}));
    EXPECT_EQ("hoople", translated("booths", "bhst", "hlep"));
    EXPECT_EQ("BORK", toupper("Bork"));
    EXPECT_EQ("bork", tolower("Bork"));
}

TEST(unit_string, translate)
{
    EXPECT_EQ("hoople", translated("booths", "bhst", "hlep"));
}

TEST(unit_string, toupper)
{
    EXPECT_EQ("BORK", toupper("Bork"));
}

TEST(unit_string, tolower)
{
    EXPECT_EQ("bork", tolower("Bork"));
}
