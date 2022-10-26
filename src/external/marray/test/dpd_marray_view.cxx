#include "dpd/dpd_marray.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace MArray;

static std::array<dpd_layout,6> layouts =
{
    PREFIX_ROW_MAJOR,
    PREFIX_COLUMN_MAJOR,
    BLOCKED_ROW_MAJOR,
    BLOCKED_COLUMN_MAJOR,
    BALANCED_ROW_MAJOR,
    BALANCED_COLUMN_MAJOR,
};

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

static dim_vector perms[6] =
    {{3,2,1,0}, {0,1,2,3}, {3,2,1,0}, {0,1,2,3}, {3,2,1,0}, {0,1,2,3}};

static irrep_vector irreps[8] =
    {{1,0,0,0}, {0,1,0,0}, {0,0,1,0}, {1,1,1,0},
     {0,0,0,1}, {1,1,0,1}, {1,0,1,1}, {0,1,1,1}};
static len_vector lengths[8] =
    {{1,2,1,3}, {3,2,1,3}, {3,2,2,3}, {1,2,2,3},
     {3,2,1,4}, {1,2,1,4}, {1,2,2,4}, {3,2,2,4}};

static stride_vector strides[6][8] =
{
    {{42,11, 3, 1}, {42,11, 3, 1}, {42,10, 3, 1}, {42,10, 3, 1},
     {42,10, 4, 1}, {42,10, 4, 1}, {42,11, 4, 1}, {42,11, 4, 1}},
    {{ 1, 1, 8,24}, { 1, 3, 8,24}, { 1, 3, 8,24}, { 1, 1, 8,24},
     { 1, 3, 8,24}, { 1, 1, 8,24}, { 1, 1, 8,24}, { 1, 3, 8,24}},
    {{ 6, 3, 3, 1}, { 6, 3, 3, 1}, {12, 6, 3, 1}, {12, 6, 3, 1},
     { 8, 4, 4, 1}, { 8, 4, 4, 1}, {16, 8, 4, 1}, {16, 8, 4, 1}},
    {{ 1, 1, 2, 2}, { 1, 3, 6, 6}, { 1, 3, 6,12}, { 1, 1, 2, 4},
     { 1, 3, 6, 6}, { 1, 1, 2, 2}, { 1, 1, 2, 4}, { 1, 3, 6,12}},
    {{22,11, 3, 1}, {22,11, 3, 1}, {20,10, 3, 1}, {20,10, 3, 1},
     {20,10, 4, 1}, {20,10, 4, 1}, {22,11, 4, 1}, {22,11, 4, 1}},
    {{ 1, 1, 8, 8}, { 1, 3, 8, 8}, { 1, 3, 8,16}, { 1, 1, 8,16},
     { 1, 3, 8, 8}, { 1, 1, 8, 8}, { 1, 1, 8,16}, { 1, 3, 8,16}}
};

static stride_type offsets[6][8] =
{
     {126, 20,  4,152,  0,148,129, 23},
     {  0,  2,  8, 14, 72, 78, 80, 82},
     {162,144, 96,132,  0, 24, 80, 32},
     {  0, 42,108, 22,144, 34,  6, 60},
     { 80+66, 80   ,     0+4,  0+60+4,
        0   ,  0+60, 80+66+3, 80   +3},
     {  0   ,  0   +2, 88   , 88   +6,
       88+48, 88+48+6,  0+24,  0+24+2}
};

#define CHECK_DPD_MARRAY_VIEW_RESET(v) \
    EXPECT_EQ(nullptr, v.data()); \
    EXPECT_EQ(0u, v.irrep()); \
    EXPECT_EQ(0u, v.num_irreps()); \
    EXPECT_EQ((dim_vector{}), v.permutation()); \
    EXPECT_EQ((matrix<len_type>{}), v.lengths());

#define CHECK_DPD_MARRAY_VIEW(v,d,j) \
    SCOPED_TRACE(j); \
    EXPECT_EQ(d, v.data()); \
    EXPECT_EQ(1u, v.irrep()); \
    EXPECT_EQ(2u, v.num_irreps()); \
    EXPECT_EQ(perms[j], v.permutation()); \
    EXPECT_EQ(matrix<len_type>({{3, 1}, {2, 2}, {1, 2}, {3, 4}}), v.lengths()); \
    \
    { \
        auto vs = v(1,0,0,0); \
        EXPECT_EQ(d + offsets[j][0], vs.data()); \
        EXPECT_EQ(*(std::array<len_type,4>*)lengths[0].data(), vs.lengths()); \
        EXPECT_EQ(*(std::array<stride_type,4>*)strides[j][0].data(), vs.strides()); \
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

    dpd_marray<double> v0(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, layouts[0]);

    dpd_marray_view<double> v1;
    CHECK_DPD_MARRAY_VIEW_RESET(v1)

    dpd_marray_view<double> v2(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]);
    CHECK_DPD_MARRAY_VIEW(v2, data, 0)

    for (int j = 1;j < 6;j++)
    {
        dpd_marray_view<double> v3(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[j]);
        CHECK_DPD_MARRAY_VIEW(v3, data, j)
    }

    dpd_marray_view<double> v5(v2);
    CHECK_DPD_MARRAY_VIEW(v5, data, 0)

    dpd_marray_view<double> v51(v0);
    CHECK_DPD_MARRAY_VIEW(v51, v0.data(), 0)

    dpd_marray_view<double> v6(dpd_marray_view<double>(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]));
    CHECK_DPD_MARRAY_VIEW(v6, data, 0)

    dpd_marray_view<const double> v9;
    CHECK_DPD_MARRAY_VIEW_RESET(v9)

    dpd_marray_view<const double> v10(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]);
    CHECK_DPD_MARRAY_VIEW(v10, data, 0)

    for (int j = 0;j < 6;j++)
    {
        dpd_marray_view<const double> v11(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[j]);
        CHECK_DPD_MARRAY_VIEW(v11, data, j)
    }

    dpd_marray_view<const double> v13(v2);
    CHECK_DPD_MARRAY_VIEW(v13, data, 0)

    dpd_marray_view<const double> v131(v0);
    CHECK_DPD_MARRAY_VIEW(v131, v0.data(), 0)

    dpd_marray_view<const double> v14(v10);
    CHECK_DPD_MARRAY_VIEW(v14, data, 0)

    dpd_marray_view<const double> v15(dpd_marray_view<double>(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]));
    CHECK_DPD_MARRAY_VIEW(v15, data, 0)

    dpd_marray_view<const double> v16(dpd_marray_view<const double>(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]));
    CHECK_DPD_MARRAY_VIEW(v16, data, 0)
}

TEST(dpd_marray_view, reset)
{
    double tmp;
    double* data = &tmp;

    dpd_marray_view<double> v1;
    dpd_marray_view<const double> v2;
    dpd_marray_view<double> v3(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]);
    dpd_marray_view<const double> v4(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]);
    dpd_marray<double> v0(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, layouts[0]);

    CHECK_DPD_MARRAY_VIEW_RESET(v1)

    for (int j = 0;j < 6;j++)
    {
        v1.reset(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[j]);
        CHECK_DPD_MARRAY_VIEW(v1, data, j)
    }

    v1.reset(v3);
    CHECK_DPD_MARRAY_VIEW(v1, data, 0)

    v1.reset(v0);
    CHECK_DPD_MARRAY_VIEW(v1, v0.data(), 0)

    v1.reset(dpd_marray_view<double>(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]));
    CHECK_DPD_MARRAY_VIEW(v1, data, 0)

    v1.reset();
    CHECK_DPD_MARRAY_VIEW_RESET(v1)

    CHECK_DPD_MARRAY_VIEW_RESET(v2)

    for (int j = 0;j < 6;j++)
    {
        v2.reset(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[j]);
        CHECK_DPD_MARRAY_VIEW(v2, data, j)
    }

    v2.reset(v3);
    CHECK_DPD_MARRAY_VIEW(v2, data, 0)

    v2.reset(v0);
    CHECK_DPD_MARRAY_VIEW(v2, v0.data(), 0)

    v2.reset(v4);
    CHECK_DPD_MARRAY_VIEW(v2, data, 0)

    v2.reset(dpd_marray_view<double>(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]));
    CHECK_DPD_MARRAY_VIEW(v2, data, 0)

    v2.reset(dpd_marray_view<const double>(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[0]));
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

    dim_vector perms2[6] =
        {{2,3,1,0}, {1,0,2,3}, {2,3,1,0}, {1,0,2,3}, {2,3,1,0}, {1,0,2,3}};

    dim_vector perms3[6] =
        {{1,2,0,3}, {2,1,3,0}, {1,2,0,3}, {2,1,3,0}, {1,2,0,3}, {2,1,3,0}};

    for (int j = 0;j < 6;j++)
    {
        SCOPED_TRACE(j);

        dpd_marray_view<double> v1(1, 2, {{3, 1}, {2, 2}, {1, 2}, {3, 4}}, data, layouts[j]);

        auto v2 = v1.permuted({1, 0, 2, 3});
        EXPECT_EQ(data, v2.data());
        EXPECT_EQ(1u, v2.irrep());
        EXPECT_EQ(2u, v2.num_irreps());
        EXPECT_EQ(perms2[j], v2.permutation());
        EXPECT_EQ((matrix<len_type>{{2, 2}, {3, 1}, {1, 2}, {3, 4}}), v2.lengths());

        for (int i = 0;i < 8;i++)
        {
            SCOPED_TRACE(i);
            len_vector len(4);
            stride_vector stride(4);
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

        v1.permute({1, 0, 2, 3});
        EXPECT_EQ(data, v1.data());
        EXPECT_EQ(1u, v1.irrep());
        EXPECT_EQ(2u, v1.num_irreps());
        EXPECT_EQ(perms2[j], v1.permutation());
        EXPECT_EQ((matrix<len_type>{{2, 2}, {3, 1}, {1, 2}, {3, 4}}), v1.lengths());

        for (int i = 0;i < 8;i++)
        {
            SCOPED_TRACE(i);
            len_vector len(4);
            stride_vector stride(4);
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

        v1.permute({2, 0, 3, 1});
        EXPECT_EQ(data, v1.data());
        EXPECT_EQ(1u, v1.irrep());
        EXPECT_EQ(2u, v1.num_irreps());
        EXPECT_EQ(perms3[j], v1.permutation());
        EXPECT_EQ((matrix<len_type>{{1, 2}, {2, 2}, {3, 4}, {3, 1}}), v1.lengths());

        for (int i = 0;i < 8;i++)
        {
            SCOPED_TRACE(i);
            len_vector len(4);
            stride_vector stride(4);
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

TEST(dpd_marray_view, block_iteration)
{
    array<array<int,2>,2> visited;
    double tmp;
    double* data = &tmp;

    for (int l = 0;l < 6;l++)
    {
        SCOPED_TRACE(l);

        dpd_marray_view<double> v1(0, 2, {{2, 3}, {1, 2}, {3, 1}}, data, layouts[l]);
        dpd_marray_view<const double> v2(0, 2, {{2, 3}, {1, 2}, {3, 1}}, data, layouts[l]);

        visited = {};
        v1.for_each_block(
        [&](marray_view<double>&& v3, const irrep_vector& irreps)
        {
            EXPECT_EQ(irreps.size(), 3u);
            int i = irreps[0];
            int j = irreps[1];
            int k = irreps[2];
            EXPECT_LT(i, 2u);
            EXPECT_LT(j, 2u);
            EXPECT_LT(k, 2u);
            EXPECT_EQ(i^j^k, 0u);
            auto v4 = v1({i, j, k});
            EXPECT_EQ(v3.data(), v4.data());
            EXPECT_EQ(v3.lengths(), v4.lengths());
            EXPECT_EQ(v3.strides(), v4.strides());
            visited[i][j]++;
        });

        for (len_type i = 0;i < 2;i++)
        {
            for (len_type j = 0;j < 2;j++)
            {
                EXPECT_EQ(visited[i][j], 1);
            }
        }

        visited = {};
        v2.for_each_block(
        [&](marray_view<const double>&& v3, const irrep_vector& irreps)
        {
            EXPECT_EQ(irreps.size(), 3u);
            int i = irreps[0];
            int j = irreps[1];
            int k = irreps[2];
            EXPECT_LT(i, 2u);
            EXPECT_LT(j, 2u);
            EXPECT_LT(k, 2u);
            EXPECT_EQ(i^j^k, 0u);
            auto v4 = v2({i, j, k});
            EXPECT_EQ(v3.data(), v4.data());
            EXPECT_EQ(v3.lengths(), v4.lengths());
            EXPECT_EQ(v3.strides(), v4.strides());
            visited[i][j]++;
        });

        for (len_type i = 0;i < 2;i++)
        {
            for (len_type j = 0;j < 2;j++)
            {
                EXPECT_EQ(visited[i][j], 1);
            }
        }

        visited = {};
        v1.for_each_block<3>(
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

        for (len_type i = 0;i < 2;i++)
        {
            for (len_type j = 0;j < 2;j++)
            {
                EXPECT_EQ(visited[i][j], 1);
            }
        }

        visited = {};
        v2.for_each_block<3>(
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

        for (len_type i = 0;i < 2;i++)
        {
            for (len_type j = 0;j < 2;j++)
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
    array<len_vector,3> len = {{{2, 3}, {1, 2}, {3, 1}}};

    for (int l = 0;l < 6;l++)
    {
        SCOPED_TRACE(l);

        dpd_marray_view<double> v1(0, 2, len, data, layouts[l]);
        dpd_marray_view<const double> v2(0, 2, len, data, layouts[l]);

        visited = {};
        v1.for_each_element(
        [&](double& v, const irrep_vector& irreps, const len_vector& pos)
        {
            EXPECT_EQ(irreps.size(), 3u);
            EXPECT_EQ(pos.size(), 3u);
            int i = irreps[0];
            int j = irreps[1];
            int k = irreps[2];
            len_type a = pos[0];
            len_type b = pos[1];
            len_type c = pos[2];
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
        [&](const double& v, const irrep_vector& irreps, const len_vector& pos)
        {
            EXPECT_EQ(irreps.size(), 3u);
            EXPECT_EQ(pos.size(), 3u);
            int i = irreps[0];
            int j = irreps[1];
            int k = irreps[2];
            len_type a = pos[0];
            len_type b = pos[1];
            len_type c = pos[2];
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

        visited = {};
        v1.for_each_element<3>(
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
        v2.for_each_element<3>(
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

    dpd_marray_view<double> v1(1, 2, {{2, 3}, {2, 1}, {5, 3}}, data1, PREFIX_ROW_MAJOR);
    dpd_marray_view<double> v2(0, 2, {{1, 1}, {6, 3}, {2, 4}}, data2, PREFIX_COLUMN_MAJOR);

    v1.swap(v2);

    EXPECT_EQ(data2, v1.data());
    EXPECT_EQ(0u, v1.irrep());
    EXPECT_EQ(2u, v1.num_irreps());
    EXPECT_EQ((dim_vector{0, 1, 2}), v1.permutation());
    EXPECT_EQ((matrix<len_type>{{1, 1}, {6, 3}, {2, 4}}), v1.lengths());

    EXPECT_EQ(data1, v2.data());
    EXPECT_EQ(1u, v2.irrep());
    EXPECT_EQ(2u, v2.num_irreps());
    EXPECT_EQ((dim_vector{2, 1, 0}), v2.permutation());
    EXPECT_EQ((matrix<len_type>{{2, 3}, {2, 1}, {5, 3}}), v2.lengths());

    swap(v2, v1);

    EXPECT_EQ(data1, v1.data());
    EXPECT_EQ(1u, v1.irrep());
    EXPECT_EQ(2u, v1.num_irreps());
    EXPECT_EQ((dim_vector{2, 1, 0}), v1.permutation());
    EXPECT_EQ((matrix<len_type>{{2, 3}, {2, 1}, {5, 3}}), v1.lengths());

    EXPECT_EQ(data2, v2.data());
    EXPECT_EQ(0u, v2.irrep());
    EXPECT_EQ(2u, v2.num_irreps());
    EXPECT_EQ((dim_vector{0, 1, 2}), v2.permutation());
    EXPECT_EQ((matrix<len_type>{{1, 1}, {6, 3}, {2, 4}}), v2.lengths());
}

TEST(dpd_marray_view, slice)
{
    double tmp;
    double* data = &tmp;

    for (auto k : range(layouts.size()))
    {
        SCOPED_TRACE(k);
        dpd_marray_view<double> v1(1, 2, {{2, 3}, {2, 1}, {5, 3}}, data, layouts[k]);

        auto v2 = v1(dpd_range(1, {2}), slice::all, dpd_index{1, 1});
        EXPECT_EQ(2, v2.dimension());
        EXPECT_EQ(0, v2.irrep());
        EXPECT_EQ(2, v2.num_irreps());
        EXPECT_EQ((matrix<len_type>{{0, 2}, {2, 1}}), v2.lengths());

        v2.for_each_element<2>([&](double& v, int irrepi, int irrepj, len_type i, len_type j)
        {
            SCOPED_TRACE(irrepi);
            SCOPED_TRACE(irrepj);
            SCOPED_TRACE(i);
            SCOPED_TRACE(j);
            EXPECT_EQ(&v1(irrepi,irrepj,1)(i,j,1)-v1.data(), &v-v1.data());
        });

        v1.for_each_element<3>([&](double& v, int irrepi, int irrepj, int irrepk, len_type i, len_type j, len_type k)
        {
            if (irrepi != 1 || i > 1) return;
            if (irrepk != 1 || k != 1) return;
            SCOPED_TRACE(irrepi);
            SCOPED_TRACE(irrepj);
            SCOPED_TRACE(irrepk);
            SCOPED_TRACE(i);
            SCOPED_TRACE(j);
            SCOPED_TRACE(k);
            EXPECT_EQ(&v-v1.data(), &v2(irrepi,irrepj)(i,j)-v1.data());
        });

        auto v3 = v2(dpd_index{1, 1}, dpd_range(0, {1, 2})(1, {1}));
        EXPECT_EQ(1, v3.dimension());
        EXPECT_EQ(1, v3.irrep());
        EXPECT_EQ(2, v3.num_irreps());
        EXPECT_EQ((matrix<len_type>{{1, 1}}), v3.lengths());

        v3.for_each_element<1>([&](double& v, int irrepi, len_type i)
        {
            SCOPED_TRACE(irrepi);
            SCOPED_TRACE(i);
            EXPECT_EQ(&v1(1,irrepi,1)(1,i,1)-v1.data(), &v-v1.data());
        });

        v1.for_each_element<3>([&](double& v, int irrepi, int irrepj, int irrepk, len_type i, len_type j, len_type k)
        {
            if (irrepi != 1 || i != 1) return;
            if (irrepj == 0 && i != 1) return;
            if (irrepk != 1 || k != 1) return;
            SCOPED_TRACE(irrepi);
            SCOPED_TRACE(irrepj);
            SCOPED_TRACE(irrepk);
            SCOPED_TRACE(i);
            SCOPED_TRACE(j);
            SCOPED_TRACE(k);
            EXPECT_EQ(&v-v1.data(), &v3(irrepj)(irrepj == 0 ? j-1 : j)-v1.data());
        });

        auto v4 = v1({dpd_range(1, {2}), dpd_range(0, {1, 2})(1, {1}), dpd_range(0, {2})(1, {2, 3})});
        EXPECT_EQ(3, v4.dimension());
        EXPECT_EQ(1, v4.irrep());
        EXPECT_EQ(2, v4.num_irreps());
        EXPECT_EQ((matrix<len_type>{{0, 2}, {1, 1}, {2, 1}}), v4.lengths());

        v4.for_each_element<3>([&](double& v, int irrepi, int irrepj, int irrepk, len_type i, len_type j, len_type k)
        {
            SCOPED_TRACE(irrepi);
            SCOPED_TRACE(irrepj);
            SCOPED_TRACE(irrepk);
            SCOPED_TRACE(i);
            SCOPED_TRACE(j);
            SCOPED_TRACE(k);
            EXPECT_EQ(&v1(irrepi,irrepj,irrepk)(i, irrepj == 0 ? j+1 : j, irrepk == 1 ? k+2 : k)-v1.data(), &v-v1.data());
        });

        v1.for_each_element<3>([&](double& v, int irrepi, int irrepj, int irrepk, len_type i, len_type j, len_type k)
        {
            if (irrepi != 1 || i > 1) return;
            if (irrepj == 0 && j != 1) return;
            if (k > (irrepk == 0 ? 1 : 2) || k < (irrepk == 0 ? 0 : 2)) return;
            SCOPED_TRACE(irrepi);
            SCOPED_TRACE(irrepj);
            SCOPED_TRACE(irrepk);
            SCOPED_TRACE(i);
            SCOPED_TRACE(j);
            SCOPED_TRACE(k);
            EXPECT_EQ(&v-v1.data(), &v4(irrepi,irrepj,irrepk)(i, irrepj == 0 ? j-1 : j, irrepk == 1 ? k-2 : k)-v1.data());
        });
    }
}
