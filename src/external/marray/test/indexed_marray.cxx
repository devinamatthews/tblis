#include "indexed/indexed_marray.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace MArray;

#define CHECK_INDEXED_VARRAY_RESET(v) \
    EXPECT_EQ(0u, v.dimension()); \
    EXPECT_EQ(0u, v.dense_dimension()); \
    EXPECT_EQ(0u, v.indexed_dimension()); \
    EXPECT_EQ(1u, v.num_indices()); \
    EXPECT_EQ(0u, v.data().size());

#define CHECK_INDEXED_VARRAY(v,value,...) \
    EXPECT_EQ(4u, v.dimension()); \
    EXPECT_EQ(2u, v.dense_dimension()); \
    EXPECT_EQ(2u, v.indexed_dimension()); \
    EXPECT_EQ((len_vector{4, 2, 5, 4}), v.lengths()); \
    EXPECT_EQ((len_vector{4, 2}), v.dense_lengths()); \
    EXPECT_EQ((len_vector{5, 4}), v.indexed_lengths()); \
    EXPECT_EQ((stride_vector __VA_ARGS__), v.dense_strides()); \
    EXPECT_EQ(3u, v.num_indices()); \
    EXPECT_EQ((matrix<len_type>{{0, 0}, {2, 1}, {4, 3}}), v.indices()); \
    EXPECT_EQ(value, v.data(0)[0]);

TEST(indexed_varray, constructor)
{
    indexed_marray<double> v1;
    CHECK_INDEXED_VARRAY_RESET(v1)

    indexed_marray<double> v2({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}});
    CHECK_INDEXED_VARRAY(v2, 0.0, {2, 1})

    indexed_marray<double> v3(vector<char>{4, 2, 5, 4}, vector<array<char,2>>{{0, 0}, {2, 1}, {4, 3}});
    CHECK_INDEXED_VARRAY(v3, 0.0, {2, 1})

    indexed_marray<double> v21({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, 1.0);
    CHECK_INDEXED_VARRAY(v21, 1.0, {2, 1})

    indexed_marray<double> v31(vector<char>{4, 2, 5, 4}, vector<array<char,2>>{{0, 0}, {2, 1}, {4, 3}}, 1.0);
    CHECK_INDEXED_VARRAY(v31, 1.0, {2, 1})

    indexed_marray<double> v4({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, 1.0, COLUMN_MAJOR);
    CHECK_INDEXED_VARRAY(v4, 1.0, {1, 4})

    indexed_marray<double> v41({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, COLUMN_MAJOR);
    CHECK_INDEXED_VARRAY(v41, 0.0, {1, 4})

    indexed_marray<double> v5(v21.view());
    CHECK_INDEXED_VARRAY(v5, 1.0, {2, 1})

    indexed_marray<double> v52(v21.cview());
    CHECK_INDEXED_VARRAY(v52, 1.0, {2, 1})

    indexed_marray<double> v51(v21);
    CHECK_INDEXED_VARRAY(v51, 1.0, {2, 1})

    indexed_marray<double> v6(indexed_marray<double>({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}));
    CHECK_INDEXED_VARRAY(v6, 0.0, {2, 1})

    indexed_marray<double> v61(indexed_marray<double>({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, COLUMN_MAJOR));
    CHECK_INDEXED_VARRAY(v61, 0.0, {1, 4})

    indexed_marray<double> v62(indexed_marray<double>({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}), COLUMN_MAJOR);
    CHECK_INDEXED_VARRAY(v62, 0.0, {1, 4})
}

TEST(indexed_varray, reset)
{
    indexed_marray<double> v2({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, 1.0);

    indexed_marray<double> v1;
    CHECK_INDEXED_VARRAY_RESET(v1)

    v1.reset({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}});
    CHECK_INDEXED_VARRAY(v1, 0.0, {2, 1})

    v1.reset(vector<char>{4, 2, 5, 4}, vector<array<char,2>>{{0, 0}, {2, 1}, {4, 3}});
    CHECK_INDEXED_VARRAY(v1, 0.0, {2, 1})

    v1.reset({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, 1.0);
    CHECK_INDEXED_VARRAY(v1, 1.0, {2, 1})

    v1.reset(vector<char>{4, 2, 5, 4}, vector<array<char,2>>{{0, 0}, {2, 1}, {4, 3}}, 1.0);
    CHECK_INDEXED_VARRAY(v1, 1.0, {2, 1})

    v1.reset({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, 1.0, COLUMN_MAJOR);
    CHECK_INDEXED_VARRAY(v1, 1.0, {1, 4})

    v1.reset({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, COLUMN_MAJOR);
    CHECK_INDEXED_VARRAY(v1, 0.0, {1, 4})

    v1.reset(v2.view());
    CHECK_INDEXED_VARRAY(v1, 1.0, {2, 1})

    v1.reset(v2.cview());
    CHECK_INDEXED_VARRAY(v1, 1.0, {2, 1})

    v1.reset(v2);
    CHECK_INDEXED_VARRAY(v1, 1.0, {2, 1})

    v1.reset(indexed_marray<double>({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}));
    CHECK_INDEXED_VARRAY(v1, 0.0, {2, 1})

    v1.reset(indexed_marray<double>({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, COLUMN_MAJOR));
    CHECK_INDEXED_VARRAY(v1, 0.0, {1, 4})

    v1.reset(indexed_marray<double>({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}), COLUMN_MAJOR);
    CHECK_INDEXED_VARRAY(v1, 0.0, {1, 4})

    v1.reset();
    CHECK_INDEXED_VARRAY_RESET(v1)
}

TEST(indexed_varray, view)
{
    indexed_marray<double> v1({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}}, 1.0);

    auto v2 = v1.cview();
    CHECK_INDEXED_VARRAY(v2, 1.0, {2, 1});

    auto v3 = v1.view();
    CHECK_INDEXED_VARRAY(v3, 1.0, {2, 1});

    auto v4 = const_cast<const indexed_marray<double>&>(v1).view();
    CHECK_INDEXED_VARRAY(v4, 1.0, {2, 1});
}

TEST(indexed_varray, access)
{
    indexed_marray<double> v1({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}});

    auto v2 = v1[0];
    EXPECT_EQ((len_vector{4, 2}), v2.lengths());
    EXPECT_EQ((stride_vector{2, 1}), v2.strides());
    EXPECT_EQ(v1.data(0), v2.data());

    auto v3 = const_cast<const indexed_marray<double>&>(v1)[2];
    EXPECT_EQ((len_vector{4, 2}), v3.lengths());
    EXPECT_EQ((stride_vector{2, 1}), v3.strides());
    EXPECT_EQ(v1.data(2), v3.data());
}

TEST(indexed_varray, index_iteration)
{
    int indices[3][2] = {{0, 0}, {2, 1}, {4, 3}};
    array<int,3> visited;

    indexed_marray<double> v1({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}});
    const indexed_marray<double> v2(v1);

    visited = {};
    v1.for_each_index(
    [&](const marray_view<double>& v, const index_vector& idx)
    {
        EXPECT_EQ(idx.size(), 2u);
        len_type i = idx[0];
        len_type j = idx[1];
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (i == indices[m][0] && j == indices[m][1])
            {
                EXPECT_EQ(v1.data(m), v.data());
                found = true;
                visited[m]++;
            }
        }
        EXPECT_TRUE(found);
        EXPECT_EQ((len_vector{4, 2}), v.lengths());
        EXPECT_EQ((stride_vector{2, 1}), v.strides());
    });

    for (len_type i = 0;i < 3;i++)
    {
        EXPECT_EQ(visited[i], 1);
    }

    visited = {};
    v2.for_each_index(
    [&](const marray_view<const double>& v, const index_vector& idx)
    {
        EXPECT_EQ(idx.size(), 2u);
        len_type i = idx[0];
        len_type j = idx[1];
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (i == indices[m][0] && j == indices[m][1])
            {
                EXPECT_EQ(v2.data(m), v.data());
                found = true;
                visited[m]++;
            }
        }
        EXPECT_TRUE(found);
        EXPECT_EQ((len_vector{4, 2}), v.lengths());
        EXPECT_EQ((stride_vector{2, 1}), v.strides());
    });

    for (len_type i = 0;i < 3;i++)
    {
        EXPECT_EQ(visited[i], 1);
    }

    visited = {};
    v1.for_each_index<2,2>(
    [&](const marray_view<double,2>& v, len_type i, len_type j)
    {
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (i == indices[m][0] && j == indices[m][1])
            {
                EXPECT_EQ(v1.data(m), v.data());
                found = true;
                visited[m]++;
            }
        }
        EXPECT_TRUE(found);
        EXPECT_EQ((array<len_type,2>{4, 2}), v.lengths());
        EXPECT_EQ((array<stride_type,2>{2, 1}), v.strides());
    });

    for (len_type i = 0;i < 3;i++)
    {
        EXPECT_EQ(visited[i], 1);
    }

    visited = {};
    v2.for_each_index<2,2>(
    [&](const marray_view<const double,2>& v, len_type i, len_type j)
    {
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (i == indices[m][0] && j == indices[m][1])
            {
                EXPECT_EQ(v2.data(m), v.data());
                found = true;
                visited[m]++;
            }
        }
        EXPECT_TRUE(found);
        EXPECT_EQ((array<len_type,2>{4, 2}), v.lengths());
        EXPECT_EQ((array<stride_type,2>{2, 1}), v.strides());
    });

    for (len_type i = 0;i < 3;i++)
    {
        EXPECT_EQ(visited[i], 1);
    }
}

TEST(indexed_varray, element_iteration)
{
    int indices[3][2] = {{0, 0}, {2, 1}, {4, 3}};
    array<array<array<int,3>,2>,4> visited;

    indexed_marray<double> v1({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}});
    const indexed_marray<double> v2(v1);

    visited = {};
    v1.for_each_element(
    [&](double& v, const len_vector& idx)
    {
        EXPECT_EQ(idx.size(), 4u);
        len_type i = idx[0];
        len_type j = idx[1];
        len_type k = idx[2];
        len_type l = idx[3];
        EXPECT_GE(i, 0);
        EXPECT_LT(i, 4);
        EXPECT_GE(j, 0);
        EXPECT_LT(j, 2);
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (k == indices[m][0] && l == indices[m][1])
            {
                EXPECT_EQ(&v1[m](i, j), &v);
                found = true;
                visited[i][j][m]++;
            }
        }
        EXPECT_TRUE(found);
    });

    for (len_type i = 0;i < 4;i++)
    {
        for (len_type j = 0;j < 2;j++)
        {
            for (len_type k = 0;k < 3;k++)
            {
                EXPECT_EQ(visited[i][j][k], 1);
            }
        }
    }

    visited = {};
    v2.for_each_element(
    [&](const double& v, const len_vector& idx)
    {
        EXPECT_EQ(idx.size(), 4u);
        len_type i = idx[0];
        len_type j = idx[1];
        len_type k = idx[2];
        len_type l = idx[3];
        EXPECT_GE(i, 0);
        EXPECT_LT(i, 4);
        EXPECT_GE(j, 0);
        EXPECT_LT(j, 2);
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (k == indices[m][0] && l == indices[m][1])
            {
                EXPECT_EQ(&v2[m](i, j), &v);
                found = true;
                visited[i][j][m]++;
            }
        }
        EXPECT_TRUE(found);
    });

    for (len_type i = 0;i < 4;i++)
    {
        for (len_type j = 0;j < 2;j++)
        {
            for (len_type k = 0;k < 3;k++)
            {
                EXPECT_EQ(visited[i][j][k], 1);
            }
        }
    }

    visited = {};
    v1.for_each_element<2,2>(
    [&](double& v, len_type i, len_type j, len_type k, len_type l)
    {
        EXPECT_GE(i, 0);
        EXPECT_LT(i, 4);
        EXPECT_GE(j, 0);
        EXPECT_LT(j, 2);
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (k == indices[m][0] && l == indices[m][1])
            {
                EXPECT_EQ(&v1[m](i, j), &v);
                found = true;
                visited[i][j][m]++;
            }
        }
        EXPECT_TRUE(found);
    });

    for (len_type i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        for (len_type j = 0;j < 2;j++)
        {
            SCOPED_TRACE(j);
            for (len_type k = 0;k < 3;k++)
            {
                SCOPED_TRACE(k);
                EXPECT_EQ(visited[i][j][k], 1);
            }
        }
    }

    visited = {};
    v2.for_each_element<2,2>(
    [&](const double& v, len_type i, len_type j, len_type k, len_type l)
    {
        EXPECT_GE(i, 0);
        EXPECT_LT(i, 4);
        EXPECT_GE(j, 0);
        EXPECT_LT(j, 2);
        bool found = false;
        for (int m = 0;m < 3;m++)
        {
            if (k == indices[m][0] && l == indices[m][1])
            {
                EXPECT_EQ(&v2[m](i, j), &v);
                found = true;
                visited[i][j][m]++;
            }
        }
        EXPECT_TRUE(found);
    });

    for (len_type i = 0;i < 4;i++)
    {
        for (len_type j = 0;j < 2;j++)
        {
            for (len_type k = 0;k < 3;k++)
            {
                EXPECT_EQ(visited[i][j][k], 1);
            }
        }
    }
}

TEST(indexed_varray, swap)
{
    indexed_marray<double> v1({4, 2, 5, 4}, {{0, 0}, {2, 1}, {4, 3}});
    indexed_marray<double> v2({4, 5, 7}, {{0, 4}, {2, 2}, {4, 1}, {1, 1}});

    auto data1 = v1.data();
    auto data2 = v2.data();

    v1.swap(v2);

    EXPECT_EQ(4u, v2.dimension());
    EXPECT_EQ(2u, v2.dense_dimension());
    EXPECT_EQ(2u, v2.indexed_dimension());
    EXPECT_EQ((len_vector{4, 2, 5, 4}), v2.lengths());
    EXPECT_EQ((len_vector{4, 2}), v2.dense_lengths());
    EXPECT_EQ((len_vector{5, 4}), v2.indexed_lengths());
    EXPECT_EQ((stride_vector{2, 1}), v2.dense_strides());
    EXPECT_EQ(3u, v2.num_indices());
    EXPECT_EQ((matrix<len_type>{{0, 0}, {2, 1}, {4, 3}}), v2.indices());
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ(1u, v1.dense_dimension());
    EXPECT_EQ(2u, v1.indexed_dimension());
    EXPECT_EQ((len_vector{4, 5, 7}), v1.lengths());
    EXPECT_EQ((len_vector{4}), v1.dense_lengths());
    EXPECT_EQ((len_vector{5, 7}), v1.indexed_lengths());
    EXPECT_EQ((stride_vector{1}), v1.dense_strides());
    EXPECT_EQ(4u, v1.num_indices());
    EXPECT_EQ((matrix<len_type>{{0, 4}, {2, 2}, {4, 1}, {1, 1}}), v1.indices());

    swap(v2, v1);

    EXPECT_EQ(4u, v1.dimension());
    EXPECT_EQ(2u, v1.dense_dimension());
    EXPECT_EQ(2u, v1.indexed_dimension());
    EXPECT_EQ((len_vector{4, 2, 5, 4}), v1.lengths());
    EXPECT_EQ((len_vector{4, 2}), v1.dense_lengths());
    EXPECT_EQ((len_vector{5, 4}), v1.indexed_lengths());
    EXPECT_EQ((stride_vector{2, 1}), v1.dense_strides());
    EXPECT_EQ(3u, v1.num_indices());
    EXPECT_EQ((matrix<len_type>{{0, 0}, {2, 1}, {4, 3}}), v1.indices());
    EXPECT_EQ(3u, v2.dimension());
    EXPECT_EQ(1u, v2.dense_dimension());
    EXPECT_EQ(2u, v2.indexed_dimension());
    EXPECT_EQ((len_vector{4, 5, 7}), v2.lengths());
    EXPECT_EQ((len_vector{4}), v2.dense_lengths());
    EXPECT_EQ((len_vector{5, 7}), v2.indexed_lengths());
    EXPECT_EQ((stride_vector{1}), v2.dense_strides());
    EXPECT_EQ(4u, v2.num_indices());
    EXPECT_EQ((matrix<len_type>{{0, 4}, {2, 2}, {4, 1}, {1, 1}}), v2.indices());
}
