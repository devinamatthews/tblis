#include "gtest/gtest.h"
#include "dpd_marray.hpp"

using namespace std;
using namespace MArray;

template <typename T, int... Sizes>
struct arrays_helper;

template <typename T, int Size1, int Size2>
struct arrays_helper<T, Size1, Size2>
{
    typedef array<array<T,Size2>,Size1> type;
};

template <typename T, int Size1, int Size2, int Size3>
struct arrays_helper<T, Size1, Size2, Size3>
{
    typedef array<array<array<T,Size3>,Size2>,Size1> type;
};

template <typename T, int... Sizes>
using arrays = typename arrays_helper<T, Sizes...>::type;

static array<dpd_layout,6> layouts =
{{
    PREFIX_ROW_MAJOR,
    PREFIX_COLUMN_MAJOR,
    BLOCKED_ROW_MAJOR,
    BLOCKED_COLUMN_MAJOR,
    BALANCED_ROW_MAJOR,
    BALANCED_COLUMN_MAJOR,
}};

/*
   3   2   1   3
   1   2   2   4

 168  42  11   3 - 126  22   3   0
 168  42  10   4 - 126  20   4   0

   3   8  24 168 -   0   6   8  72
   1   8  24 168 -   0   2   8  72

                          row               col
     8      11   -     6       3   -     6       3
     8      10   -     6       4   -     2       6
        168                88                88
        168                80                88
*/

static arrays<int,6,4> perms =
    {{{3,2,1,0}, {0,1,2,3}, {3,2,1,0}, {0,1,2,3}, {3,2,1,0}, {0,1,2,3}}};

static arrays<int,8,4> irreps =
    {{{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {1,1,1,0},
      {0,0,0,1}, {1,1,0,1}, {1,0,1,1}, {0,1,1,1}}};
static arrays<len_type,8,4> lengths =
    {{{1,2,1,3}, {3,2,1,3}, {3,2,2,3}, {1,2,2,3},
      {3,2,1,4}, {1,2,1,4}, {1,2,2,4}, {3,2,2,4}}};

static arrays<stride_type,6,8,4> strides =
{{
    {{{42,11, 3, 1}, {42,11, 3, 1}, {42,10, 3, 1}, {42,10, 3, 1},
      {42,10, 4, 1}, {42,10, 4, 1}, {42,11, 4, 1}, {42,11, 4, 1}}},
    {{{ 1, 1, 8,24}, { 1, 3, 8,24}, { 1, 3, 8,24}, { 1, 1, 8,24},
      { 1, 3, 8,24}, { 1, 1, 8,24}, { 1, 1, 8,24}, { 1, 3, 8,24}}},
    {{{ 6, 3, 3, 1}, { 6, 3, 3, 1}, {12, 6, 3, 1}, {12, 6, 3, 1},
      { 8, 4, 4, 1}, { 8, 4, 4, 1}, {16, 8, 4, 1}, {16, 8, 4, 1}}},
    {{{ 1, 1, 2, 2}, { 1, 3, 6, 6}, { 1, 3, 6,12}, { 1, 1, 2, 4},
      { 1, 3, 6, 6}, { 1, 1, 2, 2}, { 1, 1, 2, 4}, { 1, 3, 6,12}}},
    {{{22,11, 3, 1}, {22,11, 3, 1}, {20,10, 3, 1}, {20,10, 3, 1},
      {20,10, 4, 1}, {20,10, 4, 1}, {22,11, 4, 1}, {22,11, 4, 1}}},
    {{{ 1, 1, 8, 8}, { 1, 3, 8, 8}, { 1, 3, 8,16}, { 1, 1, 8,16},
      { 1, 3, 8, 8}, { 1, 1, 8, 8}, { 1, 1, 8,16}, { 1, 3, 8,16}}}
}};

static arrays<stride_type,6,8> offsets =
{{
    {126, 20,  4,152,  0,148,129, 23},
    {  0,  2,  8, 14, 72, 78, 80, 82},
    {162,144, 96,132,  0, 24, 80, 32},
    {  0, 42,108, 22,144, 34,  6, 60},
    { 80+66, 80   ,     0+4,  0+60+4,
       0   ,  0+60, 80+66+3, 80   +3},
    {  0   ,  0   +2, 88   , 88   +6,
      88+48, 88+48+6,  0+24,  0+24+2}
}};

#define CHECK_DPD_MARRAY_VIEW_RESET(v) \
    EXPECT_EQ(nullptr, v.data()); \
    EXPECT_EQ(0u, v.irrep()); \
    EXPECT_EQ(0u, v.num_irreps()); \
    EXPECT_EQ((array<int,4>{}), v.permutation()); \
    EXPECT_EQ((arrays<len_type,4,8>{}), v.lengths());

#define CHECK_DPD_MARRAY_VIEW(v,d,j) \
    SCOPED_TRACE(j); \
    EXPECT_EQ(d, v.data()); \
    EXPECT_EQ(1u, v.irrep()); \
    EXPECT_EQ(2u, v.num_irreps()); \
    EXPECT_EQ(perms[j], v.permutation()); \
    EXPECT_EQ((arrays<len_type,4,8>{{{3, 1}, {2, 2}, {1, 2}, {3, 4}}}), v.lengths()); \
    \
    { \
        auto vs = v(1,0,0,0); \
        EXPECT_EQ(d + offsets[j][0], vs.data()); \
        EXPECT_EQ(lengths[0], vs.lengths()); \
        EXPECT_EQ(strides[j][0], vs.strides()); \
    } \
    \
    { \
        auto vs = v({0,1,0,0}); \
        EXPECT_EQ(d + offsets[j][1], vs.data()); \
        EXPECT_EQ(lengths[1], vs.lengths()); \
        EXPECT_EQ(strides[j][1], vs.strides()); \
    } \
    \
    for (int i = 2;i < 8;i++) \
    { \
        SCOPED_TRACE(i); \
        auto vs = v(irreps[i]); \
        EXPECT_EQ(d + offsets[j][i], vs.data()); \
        EXPECT_EQ(lengths[i], vs.lengths()); \
        EXPECT_EQ(strides[j][i], vs.strides()); \
    }

TEST(dpd_marray_view, constructor)
{
    double tmp;
    double* data = &tmp;

    dpd_marray<double,4> v0(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, layouts[0]);

    dpd_marray_view<double,4> v1;
    CHECK_DPD_MARRAY_VIEW_RESET(v1)

    for (int j = 0;j < 6;j++)
    {
        dpd_marray_view<double,4> v2(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[j]);
        CHECK_DPD_MARRAY_VIEW(v2, data, j)
    }

    dpd_marray_view<double,4> v3(1, 2, arrays<char,4,2>{{{3, 1}, {2, 2}, {1, 2}, {3, 4}}}, data, layouts[0]);
    CHECK_DPD_MARRAY_VIEW(v3, data, 0)

    dpd_marray_view<double,4> v5(v3);
    CHECK_DPD_MARRAY_VIEW(v5, data, 0)

    dpd_marray_view<double,4> v51(v0);
    CHECK_DPD_MARRAY_VIEW(v51, v0.data(), 0)

    dpd_marray_view<double,4> v6(dpd_marray_view<double,4>(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]));
    CHECK_DPD_MARRAY_VIEW(v6, data, 0)

    dpd_marray_view<const double,4> v9;
    CHECK_DPD_MARRAY_VIEW_RESET(v9)

    for (int j = 0;j < 6;j++)
    {
        dpd_marray_view<const double,4> v10(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[j]);
        CHECK_DPD_MARRAY_VIEW(v10, data, j)
    }

    dpd_marray_view<const double,4> v11(1, 2, arrays<char,4,2>{{{3, 1}, {2, 2}, {1, 2}, {3, 4}}}, data, layouts[0]);
    CHECK_DPD_MARRAY_VIEW(v11, data, 0)

    dpd_marray_view<const double,4> v13(v3);
    CHECK_DPD_MARRAY_VIEW(v13, data, 0)

    dpd_marray_view<const double,4> v131(v0);
    CHECK_DPD_MARRAY_VIEW(v131, v0.data(), 0)

    dpd_marray_view<const double,4> v14(v11);
    CHECK_DPD_MARRAY_VIEW(v14, data, 0)

    dpd_marray_view<const double,4> v15(dpd_marray_view<double,4>(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]));
    CHECK_DPD_MARRAY_VIEW(v15, data, 0)

    dpd_marray_view<const double,4> v16(dpd_marray_view<const double,4>(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]));
    CHECK_DPD_MARRAY_VIEW(v16, data, 0)
}

TEST(dpd_marray_view, reset)
{
    double tmp;
    double* data = &tmp;

    dpd_marray_view<double,4> v1;
    dpd_marray_view<const double,4> v2;
    dpd_marray_view<double,4> v3(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]);
    dpd_marray_view<const double,4> v4(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]);
    dpd_marray<double,4> v0(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, layouts[0]);

    CHECK_DPD_MARRAY_VIEW_RESET(v1)

    for (int j = 0;j < 6;j++)
    {
        v1.reset(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[j]);
        CHECK_DPD_MARRAY_VIEW(v1, data, j)
    }

    v1.reset(1, 2, arrays<char,4,2>{{{3, 1}, {2, 2}, {1, 2}, {3, 4}}}, data, layouts[0]);
    CHECK_DPD_MARRAY_VIEW(v1, data, 0)

    v1.reset(v3);
    CHECK_DPD_MARRAY_VIEW(v1, data, 0)

    v1.reset(v0);
    CHECK_DPD_MARRAY_VIEW(v1, v0.data(), 0)

    v1.reset(dpd_marray_view<double,4>(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]));
    CHECK_DPD_MARRAY_VIEW(v1, data, 0)

    v1.reset();
    CHECK_DPD_MARRAY_VIEW_RESET(v1)

    CHECK_DPD_MARRAY_VIEW_RESET(v2)

    for (int j = 0;j < 6;j++)
    {
        v2.reset(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[j]);
        CHECK_DPD_MARRAY_VIEW(v2, data, j)
    }

    v2.reset(1, 2, arrays<char,4,2>{{{3, 1}, {2, 2}, {1, 2}, {3, 4}}}, data, layouts[0]);
    CHECK_DPD_MARRAY_VIEW(v2, data, 0)

    v2.reset(v3);
    CHECK_DPD_MARRAY_VIEW(v2, data, 0)

    v2.reset(v0);
    CHECK_DPD_MARRAY_VIEW(v2, v0.data(), 0)

    v2.reset(v4);
    CHECK_DPD_MARRAY_VIEW(v2, data, 0)

    v2.reset(dpd_marray_view<double,4>(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]));
    CHECK_DPD_MARRAY_VIEW(v2, data, 0)

    v2.reset(dpd_marray_view<const double,4>(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]));
    CHECK_DPD_MARRAY_VIEW(v2, data, 0)

    v2.reset();
    CHECK_DPD_MARRAY_VIEW_RESET(v2)
}

TEST(dpd_marray_view, permute)
{
    double tmp;
    double* data = &tmp;

    int perm_irreps[8] = {1, 0, 2, 3, 4, 5, 7, 6};
    int perm_irreps2[8] = {4, 1, 0, 5, 2, 7, 6, 3};

    arrays<int,6,4> perms2 =
        {{{2,3,1,0}, {1,0,2,3}, {2,3,1,0}, {1,0,2,3}, {2,3,1,0}, {1,0,2,3}}};

    arrays<int,6,4> perms3 =
        {{{1,2,0,3}, {2,1,3,0}, {1,2,0,3}, {2,1,3,0}, {1,2,0,3}, {2,1,3,0}}};

    for (int j = 0;j < 6;j++)
    {
        SCOPED_TRACE(j);

        dpd_marray_view<double,4> v1(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[j]);

        auto v2 = v1.permuted({1, 0, 2, 3});
        EXPECT_EQ(data, v2.data());
        EXPECT_EQ(1u, v2.irrep());
        EXPECT_EQ(2u, v2.num_irreps());
        EXPECT_EQ(perms2[j], v2.permutation());
        EXPECT_EQ((arrays<len_type,4,8>{{{2, 2}, {3, 1}, {1, 2}, {3, 4}}}), v2.lengths());

        for (int i = 0;i < 8;i++)
        {
            SCOPED_TRACE(i);
            std::array<len_type,4> len;
            std::array<stride_type,4> stride;
            for (int k = 0;k < 4;k++)
            {
                len[k] = lengths[i][perms2[1][k]];
                stride[k] = strides[j][i][perms2[1][k]];
            }
            auto vs = v2(irreps[perm_irreps[i]]);
            EXPECT_EQ(data + offsets[j][i], vs.data());
            EXPECT_EQ(len, vs.lengths());
            EXPECT_EQ(stride, vs.strides());
        }

        auto v3 = v1.permuted(std::array<char,4>{1, 0, 2, 3});
        EXPECT_EQ(data, v3.data());
        EXPECT_EQ(1u, v3.irrep());
        EXPECT_EQ(2u, v3.num_irreps());
        EXPECT_EQ(perms2[j], v3.permutation());
        EXPECT_EQ((arrays<len_type,4,8>{{{2, 2}, {3, 1}, {1, 2}, {3, 4}}}), v3.lengths());

        for (int i = 0;i < 8;i++)
        {
            SCOPED_TRACE(i);
            std::array<len_type,4> len;
            std::array<stride_type,4> stride;
            for (int k = 0;k < 4;k++)
            {
                len[k] = lengths[i][perms2[1][k]];
                stride[k] = strides[j][i][perms2[1][k]];
            }
            auto vs = v3(irreps[perm_irreps[i]]);
            EXPECT_EQ(data + offsets[j][i], vs.data());
            EXPECT_EQ(len, vs.lengths());
            EXPECT_EQ(stride, vs.strides());
        }

        v1.permute({1, 0, 2, 3});
        EXPECT_EQ(data, v1.data());
        EXPECT_EQ(1u, v1.irrep());
        EXPECT_EQ(2u, v1.num_irreps());
        EXPECT_EQ(perms2[j], v1.permutation());
        EXPECT_EQ((arrays<len_type,4,8>{{{2, 2}, {3, 1}, {1, 2}, {3, 4}}}), v1.lengths());

        for (int i = 0;i < 8;i++)
        {
            SCOPED_TRACE(i);
            std::array<len_type,4> len;
            std::array<stride_type,4> stride;
            for (int k = 0;k < 4;k++)
            {
                len[k] = lengths[i][perms2[1][k]];
                stride[k] = strides[j][i][perms2[1][k]];
            }
            auto vs = v1(irreps[perm_irreps[i]]);
            EXPECT_EQ(data + offsets[j][i], vs.data());
            EXPECT_EQ(len, vs.lengths());
            EXPECT_EQ(stride, vs.strides());
        }

        v1.permute(std::array<char,4>{2, 0, 3, 1});
        EXPECT_EQ(data, v1.data());
        EXPECT_EQ(1u, v1.irrep());
        EXPECT_EQ(2u, v1.num_irreps());
        EXPECT_EQ(perms3[j], v1.permutation());
        EXPECT_EQ((arrays<len_type,4,8>{{{1, 2}, {2, 2}, {3, 4}, {3, 1}}}), v1.lengths());

        for (int i = 0;i < 8;i++)
        {
            SCOPED_TRACE(i);
            std::array<len_type,4> len;
            std::array<stride_type,4> stride;
            for (int k = 0;k < 4;k++)
            {
                len[k] = lengths[i][perms3[1][k]];
                stride[k] = strides[j][i][perms3[1][k]];
            }
            auto vs = v1(irreps[perm_irreps2[i]]);
            EXPECT_EQ(data + offsets[j][i], vs.data());
            EXPECT_EQ(len, vs.lengths());
            EXPECT_EQ(stride, vs.strides());
        }
    }
}

TEST(dpd_marray_view, transpose)
{
    double tmp;
    double* data = &tmp;

    arrays<int,6,2> perms2 =
        {{{0,1}, {1,0}, {0,1}, {1,0}, {0,1}, {1,0}}};

    for (int j = 0;j < 6;j++)
    {
        SCOPED_TRACE(j);

        dpd_marray_view<double,2> v1(1, 2, {{3, 1}, {3, 4}}, data, layouts[j]);

        auto v2 = v1.transposed();
        EXPECT_EQ(data, v2.data());
        EXPECT_EQ(1u, v2.irrep());
        EXPECT_EQ(2u, v2.num_irreps());
        EXPECT_EQ(perms2[j], v2.permutation());
        EXPECT_EQ((arrays<len_type,2,8>{{{3, 4}, {3, 1}}}), v2.lengths());

        if (j&1)
        {
            auto v21 = v2(1, 0);
            EXPECT_EQ(data + 3, v21.data());
            EXPECT_EQ((std::array<len_type,2>{4, 3}), v21.lengths());
            EXPECT_EQ((std::array<len_type,2>{3, 1}), v21.strides());

            auto v22 = v2(0, 1);
            EXPECT_EQ(data + 0, v22.data());
            EXPECT_EQ((std::array<len_type,2>{3, 1}), v22.lengths());
            EXPECT_EQ((std::array<len_type,2>{1, 1}), v22.strides());
        }
        else
        {
            auto v21 = v2(1, 0);
            EXPECT_EQ(data + 0, v21.data());
            EXPECT_EQ((std::array<len_type,2>{4, 3}), v21.lengths());
            EXPECT_EQ((std::array<len_type,2>{1, 4}), v21.strides());

            auto v22 = v2(0, 1);
            EXPECT_EQ(data + 12, v22.data());
            EXPECT_EQ((std::array<len_type,2>{3, 1}), v22.lengths());
            EXPECT_EQ((std::array<len_type,2>{1, 3}), v22.strides());
        }

        auto v3 = v1.T();
        EXPECT_EQ(data, v3.data());
        EXPECT_EQ(1u, v3.irrep());
        EXPECT_EQ(2u, v3.num_irreps());
        EXPECT_EQ(perms2[j], v3.permutation());
        EXPECT_EQ((arrays<len_type,2,8>{{{3, 4}, {3, 1}}}), v3.lengths());

        if (j&1)
        {
            auto v21 = v3(1, 0);
            EXPECT_EQ(data + 3, v21.data());
            EXPECT_EQ((std::array<len_type,2>{4, 3}), v21.lengths());
            EXPECT_EQ((std::array<len_type,2>{3, 1}), v21.strides());

            auto v22 = v3(0, 1);
            EXPECT_EQ(data + 0, v22.data());
            EXPECT_EQ((std::array<len_type,2>{3, 1}), v22.lengths());
            EXPECT_EQ((std::array<len_type,2>{1, 1}), v22.strides());
        }
        else
        {
            auto v21 = v3(1, 0);
            EXPECT_EQ(data + 0, v21.data());
            EXPECT_EQ((std::array<len_type,2>{4, 3}), v21.lengths());
            EXPECT_EQ((std::array<len_type,2>{1, 4}), v21.strides());

            auto v22 = v3(0, 1);
            EXPECT_EQ(data + 12, v22.data());
            EXPECT_EQ((std::array<len_type,2>{3, 1}), v22.lengths());
            EXPECT_EQ((std::array<len_type,2>{1, 3}), v22.strides());
        }

        v1.transpose();
        EXPECT_EQ(data, v1.data());
        EXPECT_EQ(1u, v1.irrep());
        EXPECT_EQ(2u, v1.num_irreps());
        EXPECT_EQ(perms2[j], v1.permutation());
        EXPECT_EQ((arrays<len_type,2,8>{{{3, 4}, {3, 1}}}), v1.lengths());

        if (j&1)
        {
            auto v21 = v1(1, 0);
            EXPECT_EQ(data + 3, v21.data());
            EXPECT_EQ((std::array<len_type,2>{4, 3}), v21.lengths());
            EXPECT_EQ((std::array<len_type,2>{3, 1}), v21.strides());

            auto v22 = v1(0, 1);
            EXPECT_EQ(data + 0, v22.data());
            EXPECT_EQ((std::array<len_type,2>{3, 1}), v22.lengths());
            EXPECT_EQ((std::array<len_type,2>{1, 1}), v22.strides());
        }
        else
        {
            auto v21 = v1(1, 0);
            EXPECT_EQ(data + 0, v21.data());
            EXPECT_EQ((std::array<len_type,2>{4, 3}), v21.lengths());
            EXPECT_EQ((std::array<len_type,2>{1, 4}), v21.strides());

            auto v22 = v1(0, 1);
            EXPECT_EQ(data + 12, v22.data());
            EXPECT_EQ((std::array<len_type,2>{3, 1}), v22.lengths());
            EXPECT_EQ((std::array<len_type,2>{1, 3}), v22.strides());
        }
    }
}

TEST(dpd_marray_view, block_iteration)
{
    arrays<int,2,2> visited;
    double tmp;
    double* data = &tmp;

    for (int l = 0;l < 6;l++)
    {
        SCOPED_TRACE(l);

        dpd_marray_view<double,3> v1(0, 2, {{2, 3}, {1, 2}, {3, 1}}, data, layouts[l]);
        dpd_marray_view<const double,3> v2(0, 2, {{2, 3}, {1, 2}, {3, 1}}, data, layouts[l]);

        visited = {};
        v1.for_each_block(
        [&](marray_view<double,3>&& v3, int i, int j, int k)
        {
            EXPECT_LT(i, 2u);
            EXPECT_LT(j, 2u);
            EXPECT_LT(k, 2u);
            EXPECT_EQ(i^j^k, 0u);
            auto v4 = v1(i, j, k);
            EXPECT_EQ(v3.data(), v4.data());
            EXPECT_EQ(v3.lengths(), v4.lengths());
            EXPECT_EQ(v3.strides(), v4.strides());
            visited[i][j]++;
        });

        for (int i = 0;i < 2;i++)
        {
            for (int j = 0;j < 2;j++)
            {
                EXPECT_EQ(visited[i][j], 1);
            }
        }

        visited = {};
        v2.for_each_block(
        [&](marray_view<const double,3>&& v3, int i, int j, int k)
        {
            EXPECT_LT(i, 2u);
            EXPECT_LT(j, 2u);
            EXPECT_LT(k, 2u);
            EXPECT_EQ(i^j^k, 0u);
            auto v4 = v2(i, j, k);
            EXPECT_EQ(v3.data(), v4.data());
            EXPECT_EQ(v3.lengths(), v4.lengths());
            EXPECT_EQ(v3.strides(), v4.strides());
            visited[i][j]++;
        });

        for (int i = 0;i < 2;i++)
        {
            for (int j = 0;j < 2;j++)
            {
                EXPECT_EQ(visited[i][j], 1);
            }
        }
    }
}

TEST(dpd_marray_view, element_iteration)
{
    array<int,31> visited;
    double tmp;
    double* data = &tmp;
    arrays<len_type,3,2> len = {{{2, 3}, {1, 2}, {3, 1}}};

    for (int l = 0;l < 6;l++)
    {
        SCOPED_TRACE(l);

        dpd_marray_view<double,3> v1(0, 2, len, data, layouts[l]);
        dpd_marray_view<const double,3> v2(0, 2, len, data, layouts[l]);

        visited = {};
        v1.for_each_element(
        [&](double& v, int i, int j, int k, len_type a, len_type b, len_type c)
        {
            EXPECT_LT(i, 2u);
            EXPECT_LT(j, 2u);
            EXPECT_LT(k, 2u);
            EXPECT_GE(a, 0);
            EXPECT_LT(a, len[0][i]);
            EXPECT_GE(b, 0);
            EXPECT_LT(b, len[1][j]);
            EXPECT_GE(c, 0);
            EXPECT_LT(c, len[2][k]);
            EXPECT_EQ(i^j^k, 0u);
            auto v3 = v1(i, j, k);
            EXPECT_EQ(&v, &v3(a, b, c));
            visited[&v - data]++;
        });

        for (int i = 0;i < 31;i++)
        {
            EXPECT_EQ(visited[i], 1);
        }

        visited = {};
        v2.for_each_element(
        [&](const double& v, int i, int j, int k, len_type a, len_type b, len_type c)
        {
            EXPECT_LT(i, 2u);
            EXPECT_LT(j, 2u);
            EXPECT_LT(k, 2u);
            EXPECT_GE(a, 0);
            EXPECT_LT(a, len[0][i]);
            EXPECT_GE(b, 0);
            EXPECT_LT(b, len[1][j]);
            EXPECT_GE(c, 0);
            EXPECT_LT(c, len[2][k]);
            EXPECT_EQ(i^j^k, 0u);
            auto v4 = v1(i, j, k);
            EXPECT_EQ(&v, &v4(a, b, c));
            visited[&v - data]++;
        });

        for (int i = 0;i < 31;i++)
        {
            EXPECT_EQ(visited[i], 1);
        }
    }
}

TEST(dpd_marray_view, swap)
{
    double tmp1, tmp2;
    double* data1 = &tmp1;
    double* data2 = &tmp2;

    dpd_marray_view<double,3> v1(1, 2, {{2, 3}, {2, 1}, {5, 3}}, data1, PREFIX_ROW_MAJOR);
    dpd_marray_view<double,3> v2(0, 2, {{1, 1}, {6, 3}, {2, 4}}, data2, PREFIX_COLUMN_MAJOR);

    v1.swap(v2);

    EXPECT_EQ(data2, v1.data());
    EXPECT_EQ(0u, v1.irrep());
    EXPECT_EQ(2u, v1.num_irreps());
    EXPECT_EQ((std::array<int,3>{0, 1, 2}), v1.permutation());
    EXPECT_EQ((arrays<len_type,3,8>{{{1, 1}, {6, 3}, {2, 4}}}), v1.lengths());

    EXPECT_EQ(data1, v2.data());
    EXPECT_EQ(1u, v2.irrep());
    EXPECT_EQ(2u, v2.num_irreps());
    EXPECT_EQ((std::array<int,3>{2, 1, 0}), v2.permutation());
    EXPECT_EQ((arrays<len_type,3,8>{{{2, 3}, {2, 1}, {5, 3}}}), v2.lengths());

    swap(v2, v1);

    EXPECT_EQ(data1, v1.data());
    EXPECT_EQ(1u, v1.irrep());
    EXPECT_EQ(2u, v1.num_irreps());
    EXPECT_EQ((std::array<int,3>{2, 1, 0}), v1.permutation());
    EXPECT_EQ((arrays<len_type,3,8>{{{2, 3}, {2, 1}, {5, 3}}}), v1.lengths());

    EXPECT_EQ(data2, v2.data());
    EXPECT_EQ(0u, v2.irrep());
    EXPECT_EQ(2u, v2.num_irreps());
    EXPECT_EQ((std::array<int,3>{0, 1, 2}), v2.permutation());
    EXPECT_EQ((arrays<len_type,3,8>{{{1, 1}, {6, 3}, {2, 4}}}), v2.lengths());
}
