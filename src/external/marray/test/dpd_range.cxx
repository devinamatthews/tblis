#include "gtest/gtest.h"
#include "dpd/dpd_range.hpp"

using namespace std;
using namespace MArray;

#define CHECK_DPD_RANGE(r,f,s,d) \
{ \
    std::vector<len_type> front f; front.resize(8); \
    std::vector<len_type> size s; size.resize(8); \
    std::vector<len_type> delta d; delta.resize(8, 1); \
    for (auto i : range(8)) \
    { \
        SCOPED_TRACE(i); \
        EXPECT_EQ(r[i].front(), front[i]); \
        EXPECT_EQ(r[i].size(), size[i]); \
        EXPECT_EQ(r[i].step(), delta[i]); \
    } \
}

TEST(dpd_range, dpd_range)
{
    auto r1 = dpd_range({3, 5, 6, 2});
    CHECK_DPD_RANGE(r1, ({0, 0, 0, 0}), ({3, 5, 6, 2}), ({1, 1, 1, 1}));

    auto r2 = dpd_range({1, 0, 2, 1}, {2, 3, 4, 5});
    CHECK_DPD_RANGE(r2, ({1, 0, 2, 1}), ({1, 3, 2, 4}), ({1, 1, 1, 1}));

    auto r3 = dpd_range({1, 0, 2, 1}, {2, 3, 4, 5}, {1, 1, 2, 4});
    CHECK_DPD_RANGE(r3, ({1, 0, 2, 1}), ({1, 3, 1, 1}), ({1, 1, 2, 4}));

    auto r4 = dpd_range(0, {4})(1, {1, 4})(2, {2, 3})(3, {2, 6, 2});
    CHECK_DPD_RANGE(r4, ({0, 1, 2, 2}), ({4, 3, 1, 2}), ({1, 1, 1, 2}));
}
