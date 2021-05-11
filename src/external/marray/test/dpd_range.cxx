#include "gtest/gtest.h"
#include "dpd_range.hpp"

using namespace std;
using namespace MArray;

#define CHECK_DPD_RANGE(r,f,s,d) \
{ \
    std::vector<len_type> __front f; __front.resize(8); \
    std::vector<len_type> __size s; __size.resize(8); \
    std::vector<len_type> __delta d; __delta.resize(8, 1); \
    for (auto __i : range(8)) \
    { \
        SCOPED_TRACE(__i); \
        EXPECT_EQ(r[__i].front(), __front[__i]); \
        EXPECT_EQ(r[__i].size(), __size[__i]); \
        EXPECT_EQ(r[__i].step(), __delta[__i]); \
    } \
}

TEST(dpd_range, dpd_range)
{
    auto r1 = dpd_range({3, 5, 6, 2});
    CHECK_DPD_RANGE(r1, , ({3, 5, 6, 2}), );

    auto r2 = dpd_range({1, 0, 2, 1}, {2, 3, 4, 5});
    CHECK_DPD_RANGE(r2, ({1, 0, 2, 1}), ({1, 3, 2, 4}), );

    auto r3 = dpd_range({1, 0, 2, 1}, {2, 3, 4, 5}, {1, 1, 2, 4});
    CHECK_DPD_RANGE(r3, ({1, 0, 2, 1}), ({1, 3, 1, 1}), ({1, 1, 2, 4}));

    auto r4 = dpd_range(0, {4})(1, {1, 4})(2, {2, 3})(3, {2, 6, 2});
    CHECK_DPD_RANGE(r4, ({0, 1, 2, 2}), ({4, 3, 1, 2}), ({1, 1, 1, 2}));
}
