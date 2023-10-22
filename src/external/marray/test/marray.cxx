#include "marray.hpp"

#include "rotate.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace MArray;

TEST(marray, constructor)
{
    double data[40];

    marray_view<double,3> v0({4, 2, 5}, data);
    marray_view<const double,3> v01({4, 2, 5}, data);
    marray_view<int,3> v02({4, 2, 5}, (int*)data);
    marray<int,3> v03{4, 2, 5};
    marray_view<double,4> v04({3, 4, 2, 5}, data);

    marray<double,2> v1;
    EXPECT_EQ(nullptr, v1.data());
    EXPECT_EQ((array<len_type,2>{0, 0}), v1.lengths());
    EXPECT_EQ((array<stride_type,2>{0, 0}), v1.strides());

    marray<double,3> v2{4, 2, 5};
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v2.strides());
    EXPECT_EQ(0, v2.data()[0]);

    marray<double,3> v21{4, 2, 5};
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v21.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v21.strides());
    EXPECT_EQ(0, v21.data()[0]);

    marray<double,3> v3(array<char,3>{4, 2, 5});
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v3.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v3.strides());
    EXPECT_EQ(0, v3.data()[0]);

    marray<double,3> v31({4, 2, 5}, 1.0);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v31.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v31.strides());
    EXPECT_EQ(1, v31.data()[0]);

    marray<double,3> v32(array<char,3>{4, 2, 5}, 1.0);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v32.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v32.strides());
    EXPECT_EQ(1, v32.data()[0]);

    marray<double,3> v4({4, 2, 5}, 1.0, COLUMN_MAJOR);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v4.lengths());
    EXPECT_EQ((array<stride_type,3>{1, 4, 8}), v4.strides());
    EXPECT_EQ(1, v4.data()[0]);

    marray<double,3> v41({4, 2, 5}, COLUMN_MAJOR);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v41.lengths());
    EXPECT_EQ((array<stride_type,3>{1, 4, 8}), v41.strides());
    EXPECT_EQ(0, v41.data()[0]);

    marray<double,3> v5(v0);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v5.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v5.strides());

    marray<double,3> v52(v01);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v52.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v52.strides());

    marray<double,3> v53(v02);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v53.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v53.strides());

    marray<double,3> v54(v03);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v54.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v54.strides());

    marray<double,3> v55(v04[0]);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v55.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v55.strides());

    marray<double,3> v51(v31);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v51.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v51.strides());
    EXPECT_EQ(1, v51.data()[0]);

    marray<double,3> v6(marray<double,3>{4, 2, 5});
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v6.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v6.strides());

    marray<double,1> v7(std::vector<double>{1,2,3});
    EXPECT_EQ((array<len_type,1>{3}), v7.lengths());
    EXPECT_EQ((array<stride_type,1>{1}), v7.strides());
    EXPECT_EQ(1, v7.data()[0]);
}

TEST(varray, constructor)
{
    double data[40];

    marray_view<double> v0({4, 2, 5}, data);
    marray_view<const double> v01({4, 2, 5}, data);
    marray_view<int> v02({4, 2, 5}, (int*)data);
    marray<int> v03{4, 2, 5};

    marray<double> v1;
    EXPECT_EQ(0u, v1.dimension());
    EXPECT_EQ(nullptr, v1.data());

    marray<double> v21{4, 2, 5};
    EXPECT_EQ(3u, v21.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v21.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v21.strides());
    EXPECT_EQ(0, v21.data()[0]);

    marray<double> v2{4, 2, 5};
    EXPECT_EQ(3u, v2.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v2.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v2.strides());
    EXPECT_EQ(0, v2.data()[0]);

    /*
     * The semantics for this have changed, now a 1-D copy of
     * the vector is created.
     *
    marray<double> v3(vector<char>{4, 2, 5});
    EXPECT_EQ(3u, v3.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v3.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v3.strides());
    EXPECT_EQ(0, v3.data()[0]);
     */

    marray<double> v31({4, 2, 5}, 1.0);
    EXPECT_EQ(3u, v31.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v31.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v31.strides());
    EXPECT_EQ(1, v31.data()[0]);

    marray<double> v32(vector<char>{4, 2, 5}, 1.0);
    EXPECT_EQ(3u, v32.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v32.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v32.strides());
    EXPECT_EQ(1, v32.data()[0]);

    marray<double> v4({4, 2, 5}, 1.0, COLUMN_MAJOR);
    EXPECT_EQ(3u, v4.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v4.lengths());
    EXPECT_EQ((stride_vector{1, 4, 8}), v4.strides());
    EXPECT_EQ(1, v4.data()[0]);

    marray<double> v41({4, 2, 5}, COLUMN_MAJOR);
    EXPECT_EQ(3u, v41.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v41.lengths());
    EXPECT_EQ((stride_vector{1, 4, 8}), v41.strides());
    EXPECT_EQ(0, v41.data()[0]);

    marray<double> v5(v0);
    EXPECT_EQ(3u, v5.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v5.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v5.strides());

    marray<double> v52(v01);
    EXPECT_EQ(3u, v52.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v52.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v52.strides());

    marray<double> v53(v02);
    EXPECT_EQ(3u, v53.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v53.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v53.strides());

    marray<double> v54(v03);
    EXPECT_EQ(3u, v54.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v54.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v54.strides());

    marray<double> v51(v31);
    EXPECT_EQ(3u, v51.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v51.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v51.strides());
    EXPECT_EQ(1, v51.data()[0]);

    marray<double> v6(marray<double>{4, 2, 5});
    EXPECT_EQ(3u, v6.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v6.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v6.strides());

    marray<double> v7(std::vector<double>{1,2,3});
    EXPECT_EQ(1u, v7.dimension());
    EXPECT_EQ((len_vector{3}), v7.lengths());
    EXPECT_EQ((stride_vector{1}), v7.strides());
    EXPECT_EQ(1, v7.data()[0]);
}

TEST(marray, reset)
{
    double data[40];

    marray_view<double,3> v0({4, 2, 5}, data);
    marray_view<const double,3> v01({4, 2, 5}, data);
    marray_view<int,3> v02({4, 2, 5}, (int*)data);
    marray<int,3> v03{4, 2, 5};
    marray_view<double,4> v04({3, 4, 2, 5}, data);

    marray<double,3> v1;
    marray<double,3> v2({4, 2, 5}, 1.0);

    v1.reset({4, 2, 5});
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());
    EXPECT_EQ(0, v1.data()[0]);

    v1.reset(array<char,3>{4, 2, 5});
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());
    EXPECT_EQ(0, v1.data()[0]);

    v1.reset({4, 2, 5}, 1.0);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());
    EXPECT_EQ(1, v1.data()[0]);

    v1.reset(array<char,3>{4, 2, 5}, 1.0);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());
    EXPECT_EQ(1, v1.data()[0]);

    v1.reset({4, 2, 5}, 1.0, COLUMN_MAJOR);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{1, 4, 8}), v1.strides());
    EXPECT_EQ(1, v1.data()[0]);

    v1.reset({4, 2, 5}, COLUMN_MAJOR);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{1, 4, 8}), v1.strides());
    EXPECT_EQ(0, v1.data()[0]);

    v1.reset(v0);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());

    v1.reset(v01);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());

    v1.reset(v02);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());

    v1.reset(v03);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());

    v1.reset(v04[0]);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());

    v1.reset(v2);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());
    EXPECT_EQ(1, v1.data()[0]);

    v1.reset(marray<double,3>{4, 2, 5});
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());

    v1.reset();
    EXPECT_EQ(nullptr, v1.data());
    EXPECT_EQ((array<len_type,3>{0, 0, 0}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{0, 0, 0}), v1.strides());
}

TEST(varray, reset)
{
    double data[40];

    marray_view<double> v0({4, 2, 5}, data);
    marray_view<const double> v01({4, 2, 5}, data);
    marray_view<int> v02({4, 2, 5}, (int*)data);
    marray<int> v03{4, 2, 5};

    marray<double> v1;
    marray<double> v2({4, 2, 5}, 1.0);

    v1.reset({4, 2, 5});
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());
    EXPECT_EQ(0, v1.data()[0]);

    /*
     * The semantics for this have changed, now a 1-D copy of
     * the vector is created.
     *
    v1.reset(vector<char>{4, 2, 5});
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());
    EXPECT_EQ(0, v1.data()[0]);
     */

    v1.reset({4, 2, 5}, 1.0);
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());
    EXPECT_EQ(1, v1.data()[0]);

    v1.reset(vector<char>{4, 2, 5}, 1.0);
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());
    EXPECT_EQ(1, v1.data()[0]);

    v1.reset({4, 2, 5}, 1.0, COLUMN_MAJOR);
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{1, 4, 8}), v1.strides());
    EXPECT_EQ(1, v1.data()[0]);

    v1.reset({4, 2, 5}, COLUMN_MAJOR);
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{1, 4, 8}), v1.strides());
    EXPECT_EQ(0, v1.data()[0]);

    v1.reset(v0);
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());

    v1.reset(v01);
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());

    v1.reset(v02);
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());

    v1.reset(v03);
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());

    v1.reset(v2);
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());
    EXPECT_EQ(1, v1.data()[0]);

    v1.reset(marray<double>{4, 2, 5});
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());

    v1.reset();
    EXPECT_EQ(0u, v1.dimension());
    EXPECT_EQ(nullptr, v1.data());
}

TEST(marray, initialize)
{
    marray<double,3> v1({3, 2, 3}, ROW_MAJOR);
    marray<double,3> v2({3, 2, 3}, COLUMN_MAJOR);

    v1 = {{{ 0, 1, 2},
           { 3, 4, 5}},
          {{ 6, 7, 8},
           { 9,10,11}},
          {{12,13,14},
           {15,16,17}}};

    EXPECT_EQ((std::array<double,18>{ 0, 1, 2, 3, 4, 5,
                                      6, 7, 8, 9,10,11,
                                     12,13,14,15,16,17}),
              *(const std::array<double,18>*)v1.data());

    v2 = {{{ 0, 1, 2},
           { 3, 4, 5}},
          {{ 6, 7, 8},
           { 9,10,11}},
          {{12,13,14},
           {15,16,17}}};

    EXPECT_EQ((std::array<double,18>{ 0, 6,12, 3, 9,15,
                                      1, 7,13, 4,10,16,
                                      2, 8,14, 5,11,17}),
              *(const std::array<double,18>*)v2.data());

    marray<double,3> v4({{{ 0, 1, 2},
                          { 3, 4, 5}},
                         {{ 6, 7, 8},
                          { 9,10,11}},
                         {{12,13,14},
                          {15,16,17}}});

    EXPECT_EQ((std::array<double,18>{ 0, 1, 2, 3, 4, 5,
                                      6, 7, 8, 9,10,11,
                                     12,13,14,15,16,17}),
              *(const std::array<double,18>*)v4.data());

    marray<double,3> v5 = {{{ 0, 1, 2},
                            { 3, 4, 5}},
                           {{ 6, 7, 8},
                            { 9,10,11}},
                           {{12,13,14},
                            {15,16,17}}};

    EXPECT_EQ((std::array<double,18>{ 0, 1, 2, 3, 4, 5,
                                      6, 7, 8, 9,10,11,
                                     12,13,14,15,16,17}),
              *(const std::array<double,18>*)v5.data());
}

TEST(varray, assign)
{
    double data1[6] = {0, 1, 2,
                      3, 4, 5};
    int data2[6] = {0, 1, 2,
                    3, 4, 5};

    marray<double> v1{2, 3};

    v1 = marray_view<double>({2, 3}, data1);
    EXPECT_EQ((len_vector{2, 3}), v1.lengths());
    EXPECT_EQ((stride_vector{3, 1}), v1.strides());
    EXPECT_EQ((array<double,6>{0, 1, 2,
                               3, 4, 5}), *(array<double,6>*)v1.data());

    v1 = 1.0;
    EXPECT_EQ((len_vector{2, 3}), v1.lengths());
    EXPECT_EQ((stride_vector{3, 1}), v1.strides());
    EXPECT_EQ((array<double,6>{1, 1, 1,
                               1, 1, 1}), *(array<double,6>*)v1.data());

    v1 = marray_view<int>({2, 3}, data2);
    EXPECT_EQ((len_vector{2, 3}), v1.lengths());
    EXPECT_EQ((stride_vector{3, 1}), v1.strides());
    EXPECT_EQ((array<double,6>{0, 1, 2,
                               3, 4, 5}), *(array<double,6>*)v1.data());

    v1 = 1;
    EXPECT_EQ((len_vector{2, 3}), v1.lengths());
    EXPECT_EQ((stride_vector{3, 1}), v1.strides());
    EXPECT_EQ((array<double,6>{1, 1, 1,
                               1, 1, 1}), *(array<double,6>*)v1.data());

    v1 = marray_view<const double>({2, 3}, data1);
    EXPECT_EQ((len_vector{2, 3}), v1.lengths());
    EXPECT_EQ((stride_vector{3, 1}), v1.strides());
    EXPECT_EQ((array<double,6>{0, 1, 2,
                               3, 4, 5}), *(array<double,6>*)v1.data());
}

TEST(marray, resize)
{
    double data[6] = {0, 1, 2,
                      3, 4, 5};

    marray<double,2> v1(marray_view<const double,2>({2, 3}, data));

    v1.resize({2, 2});
    EXPECT_EQ((array<len_type,2>{2, 2}), v1.lengths());
    EXPECT_EQ((array<stride_type,2>{2, 1}), v1.strides());
    EXPECT_EQ((array<double,4>{0, 1,
                               3, 4}), *(array<double,4>*)v1.data());

    v1.resize(array<char,2>{3, 4}, 1);
    EXPECT_EQ((array<len_type,2>{3, 4}), v1.lengths());
    EXPECT_EQ((array<stride_type,2>{4, 1}), v1.strides());
    EXPECT_EQ((array<double,12>{0, 1, 1, 1,
                                3, 4, 1, 1,
                                1, 1, 1, 1}), *(array<double,12>*)v1.data());
}

TEST(varray, resize)
{
    double data[6] = {0, 1, 2,
                      3, 4, 5};

    marray<double> v1(marray_view<const double>({2, 3}, data));

    v1.resize({2, 2});
    EXPECT_EQ((len_vector{2, 2}), v1.lengths());
    EXPECT_EQ((stride_vector{2, 1}), v1.strides());
    EXPECT_EQ((array<double,4>{0, 1,
                               3, 4}), *(array<double,4>*)v1.data());

    v1.resize(vector<char>{3, 4}, 1);
    EXPECT_EQ((len_vector{3, 4}), v1.lengths());
    EXPECT_EQ((stride_vector{4, 1}), v1.strides());
    EXPECT_EQ((array<double,12>{0, 1, 1, 1,
                                3, 4, 1, 1,
                                1, 1, 1, 1}), *(array<double,12>*)v1.data());
}

TEST(marray, push_pop)
{
    double data1[6] = {0, 1, 2,
                       3, 4, 5};
    double data2[3] = {6, 7, 8};
    double data3[3] = {-1, -1, -1};

    marray<double,2> v1(marray_view<const double,2>({2, 3}, data1));
    marray_view<const double,1> v2({3}, data2);
    marray_view<const double,1> v3({3}, data3);

    v1.push_back(0, v2);
    EXPECT_EQ((array<len_type,2>{3, 3}), v1.lengths());
    EXPECT_EQ((array<stride_type,2>{3, 1}), v1.strides());
    EXPECT_EQ((array<double,9>{0, 1, 2,
                               3, 4, 5,
                               6, 7, 8}), *(array<double,9>*)v1.data());

    v1.push_back(1, v3);
    EXPECT_EQ((array<len_type,2>{3, 4}), v1.lengths());
    EXPECT_EQ((array<stride_type,2>{4, 1}), v1.strides());
    EXPECT_EQ((array<double,12>{0, 1, 2, -1,
                                3, 4, 5, -1,
                                6, 7, 8, -1}), *(array<double,12>*)v1.data());

    v1.pop_back(0);
    EXPECT_EQ((array<len_type,2>{2, 4}), v1.lengths());
    EXPECT_EQ((array<stride_type,2>{4, 1}), v1.strides());
    EXPECT_EQ((array<double,8>{0, 1, 2, -1,
                               3, 4, 5, -1}), *(array<double,8>*)v1.data());

    v1.pop_back(1);
    EXPECT_EQ((array<len_type,2>{2, 3}), v1.lengths());
    EXPECT_EQ((array<stride_type,2>{3, 1}), v1.strides());
    EXPECT_EQ((array<double,6>{0, 1, 2,
                               3, 4, 5}), *(array<double,6>*)v1.data());

    marray<double,1> v4(marray_view<const double,1>({6}, data1));

    v4.push_back(6);

    EXPECT_EQ((array<len_type,1>{7}), v4.lengths());
    EXPECT_EQ((array<stride_type,1>{1}), v4.strides());
    EXPECT_EQ((array<double,7>{0, 1, 2, 3, 4, 5, 6}), *(array<double,7>*)v4.data());

    v4.pop_back();
    v4.pop_back();

    EXPECT_EQ((array<len_type,1>{5}), v4.lengths());
    EXPECT_EQ((array<stride_type,1>{1}), v4.strides());
    EXPECT_EQ((array<double,5>{0, 1, 2, 3, 4}), *(array<double,5>*)v4.data());
}

TEST(varray, push_pop)
{
    double data1[6] = {0, 1, 2,
                       3, 4, 5};
    double data2[3] = {6, 7, 8};
    double data3[3] = {-1, -1, -1};

    marray<double> v1(marray_view<const double>({2, 3}, data1));
    marray_view<const double> v2({3}, data2);
    marray_view<const double> v3({3}, data3);

    v1.push_back(0, v2);
    EXPECT_EQ((len_vector{3, 3}), v1.lengths());
    EXPECT_EQ((stride_vector{3, 1}), v1.strides());
    EXPECT_EQ((array<double,9>{0, 1, 2,
                               3, 4, 5,
                               6, 7, 8}), *(array<double,9>*)v1.data());

    v1.push_back(1, v3);
    EXPECT_EQ((len_vector{3, 4}), v1.lengths());
    EXPECT_EQ((stride_vector{4, 1}), v1.strides());
    EXPECT_EQ((array<double,12>{0, 1, 2, -1,
                                3, 4, 5, -1,
                                6, 7, 8, -1}), *(array<double,12>*)v1.data());

    v1.pop_back(0);
    EXPECT_EQ((len_vector{2, 4}), v1.lengths());
    EXPECT_EQ((stride_vector{4, 1}), v1.strides());
    EXPECT_EQ((array<double,8>{0, 1, 2, -1,
                               3, 4, 5, -1}), *(array<double,8>*)v1.data());

    v1.pop_back(1);
    EXPECT_EQ((len_vector{2, 3}), v1.lengths());
    EXPECT_EQ((stride_vector{3, 1}), v1.strides());
    EXPECT_EQ((array<double,6>{0, 1, 2,
                               3, 4, 5}), *(array<double,6>*)v1.data());
}

TEST(marray, view)
{
    marray<double,3> v1{4, 2, 5};

    auto v2 = v1.cview();
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v2.strides());
    EXPECT_EQ(v1.data(), v2.data());

    auto v3 = v1.view();
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v3.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v3.strides());
    EXPECT_EQ(v1.data(), v3.data());

    auto v4 = const_cast<const marray<double,3>&>(v1).view();
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v4.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v4.strides());
    EXPECT_EQ(v1.data(), v4.data());
}

TEST(varray, view)
{
    marray<double> v1{4, 2, 5};

    auto v2 = v1.cview();
    EXPECT_EQ((len_vector{4, 2, 5}), v2.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v2.strides());
    EXPECT_EQ(v1.data(), v2.data());

    auto v3 = v1.view();
    EXPECT_EQ((len_vector{4, 2, 5}), v3.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v3.strides());
    EXPECT_EQ(v1.data(), v3.data());

    auto v4 = const_cast<const marray<double>&>(v1).view();
    EXPECT_EQ((len_vector{4, 2, 5}), v4.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v4.strides());
    EXPECT_EQ(v1.data(), v4.data());
}

TEST(marray, permuted)
{
    marray<double,3> v1{4, 2, 5};

    auto v2 = v1.permuted({1, 0, 2});
    EXPECT_EQ((array<len_type,3>{2, 4, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{5, 10, 1}), v2.strides());
    EXPECT_EQ(v1.data(), v2.data());

    auto v3 = v1.permuted(array<char,3>{2, 0, 1});
    EXPECT_EQ((array<len_type,3>{5, 4, 2}), v3.lengths());
    EXPECT_EQ((array<stride_type,3>{1, 10, 5}), v3.strides());
    EXPECT_EQ(v1.data(), v3.data());

    auto v4 = const_cast<const marray<double,3>&>(v1).permuted(1, 0, 2);
    EXPECT_EQ((array<len_type,3>{2, 4, 5}), v4.lengths());
    EXPECT_EQ((array<stride_type,3>{5, 10, 1}), v4.strides());
    EXPECT_EQ(v1.data(), v4.data());

    auto v5 = v1.permuted(1, 0, 2);
    EXPECT_EQ((array<len_type,3>{2, 4, 5}), v5.lengths());
    EXPECT_EQ((array<stride_type,3>{5, 10, 1}), v5.strides());
    EXPECT_EQ(v1.data(), v5.data());
}

TEST(varray, permuted)
{
    marray<double> v1{4, 2, 5};

    auto v2 = v1.permuted({1, 0, 2});
    EXPECT_EQ((len_vector{2, 4, 5}), v2.lengths());
    EXPECT_EQ((stride_vector{5, 10, 1}), v2.strides());
    EXPECT_EQ(v1.data(), v2.data());

    auto v3 = v1.permuted(vector<char>{2, 0, 1});
    EXPECT_EQ((len_vector{5, 4, 2}), v3.lengths());
    EXPECT_EQ((stride_vector{1, 10, 5}), v3.strides());
    EXPECT_EQ(v1.data(), v3.data());

    auto v4 = const_cast<const marray<double>&>(v1).permuted(1, 0, 2);
    EXPECT_EQ((len_vector{2, 4, 5}), v4.lengths());
    EXPECT_EQ((stride_vector{5, 10, 1}), v4.strides());
    EXPECT_EQ(v1.data(), v4.data());

    auto v5 = v1.permuted(1, 0, 2);
    EXPECT_EQ((len_vector{2, 4, 5}), v5.lengths());
    EXPECT_EQ((stride_vector{5, 10, 1}), v5.strides());
    EXPECT_EQ(v1.data(), v5.data());
}

TEST(marray, transposed)
{
    marray<double,2> v1{4, 8};

    auto v2 = v1.transposed();
    EXPECT_EQ((array<len_type,2>{8, 4}), v2.lengths());
    EXPECT_EQ((array<stride_type,2>{1, 8}), v2.strides());
    EXPECT_EQ(v1.data(), v2.data());

    auto v3 = v1.T();
    EXPECT_EQ((array<len_type,2>{8, 4}), v3.lengths());
    EXPECT_EQ((array<stride_type,2>{1, 8}), v3.strides());
    EXPECT_EQ(v1.data(), v3.data());

    auto v4 = const_cast<const marray<double,2>&>(v1).T();
    EXPECT_EQ((array<len_type,2>{8, 4}), v4.lengths());
    EXPECT_EQ((array<stride_type,2>{1, 8}), v4.strides());
    EXPECT_EQ(v1.data(), v4.data());
}

TEST(marray, lowered)
{
    marray<double,3> v1{4, 2, 5};

    auto v2 = v1.lowered<3>({1, 2});
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v2.strides());
    EXPECT_EQ(v1.data(), v2.data());

    auto v3 = v1.lowered<2>(array<char,1>{1});
    EXPECT_EQ((array<len_type,2>{4, 10}), v3.lengths());
    EXPECT_EQ((array<stride_type,2>{10, 1}), v3.strides());
    EXPECT_EQ(v1.data(), v3.data());

    auto v4 = v1.lowered<1>({});
    EXPECT_EQ((array<len_type,1>{40}), v4.lengths());
    EXPECT_EQ((array<stride_type,1>{1}), v4.strides());
    EXPECT_EQ(v1.data(), v4.data());

    auto v5 = const_cast<const marray<double,3>&>(v1).lowered(1, 2);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v5.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v5.strides());
    EXPECT_EQ(v1.data(), v2.data());
}

TEST(varray, lowered)
{
    marray<double> v1{4, 2, 5};

    auto v2 = v1.lowered({1, 2});
    EXPECT_EQ((len_vector{4, 2, 5}), v2.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v2.strides());
    EXPECT_EQ(v1.data(), v2.data());

    auto v3 = v1.lowered(vector<char>{1});
    EXPECT_EQ((len_vector{4, 10}), v3.lengths());
    EXPECT_EQ((stride_vector{10, 1}), v3.strides());
    EXPECT_EQ(v1.data(), v3.data());

    auto v3b = v1.lowered(1);
    EXPECT_EQ((array<len_type,2>{4, 10}), v3b.lengths());
    EXPECT_EQ((array<stride_type,2>{10, 1}), v3b.strides());
    EXPECT_EQ(v1.data(), v3b.data());

    auto v4 = v1.lowered();
    EXPECT_EQ((array<len_type,1>{40}), v4.lengths());
    EXPECT_EQ((array<stride_type,1>{1}), v4.strides());
    EXPECT_EQ(v1.data(), v4.data());

    auto v5 = const_cast<const marray<double>&>(v1).lowered(1, 2);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v5.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v5.strides());
    EXPECT_EQ(v1.data(), v2.data());
}

TEST(marray, rotate)
{
    array<double,12> data = { 0, 1, 2,
                              3, 4, 5,
                              6, 7, 8,
                              9,10,11};

    marray<double,2> v1(marray_view<const double,2>({4, 3}, data.data()));

    rotate(v1, 1, 1);
    EXPECT_EQ((array<double,12>{ 1, 2, 0,
                                 4, 5, 3,
                                 7, 8, 6,
                                10,11, 9}), *(array<double,12>*)v1.data());

    rotate(v1, 0, -1);
    EXPECT_EQ((array<double,12>{10,11, 9,
                                 1, 2, 0,
                                 4, 5, 3,
                                 7, 8, 6}), *(array<double,12>*)v1.data());

    rotate(v1, {4,3});
    EXPECT_EQ((array<double,12>{10,11, 9,
                                 1, 2, 0,
                                 4, 5, 3,
                                 7, 8, 6}), *(array<double,12>*)v1.data());

    rotate(v1, array<char,2>{1,1});
    EXPECT_EQ((array<double,12>{ 2, 0, 1,
                                 5, 3, 4,
                                 8, 6, 7,
                                11, 9,10}), *(array<double,12>*)v1.data());
}

TEST(varray, rotate)
{
    array<double,12> data = { 0, 1, 2,
                              3, 4, 5,
                              6, 7, 8,
                              9,10,11};

    marray<double> v1(marray_view<const double>({4, 3}, data.data()));

    rotate(v1, 1, 1);
    EXPECT_EQ((array<double,12>{ 1, 2, 0,
                                 4, 5, 3,
                                 7, 8, 6,
                                10,11, 9}), *(array<double,12>*)v1.data());

    rotate(v1, 0, -1);
    EXPECT_EQ((array<double,12>{10,11, 9,
                                 1, 2, 0,
                                 4, 5, 3,
                                 7, 8, 6}), *(array<double,12>*)v1.data());

    rotate(v1, {4,3});
    EXPECT_EQ((array<double,12>{10,11, 9,
                                 1, 2, 0,
                                 4, 5, 3,
                                 7, 8, 6}), *(array<double,12>*)v1.data());

    rotate(v1, vector<char>{1,1});
    EXPECT_EQ((array<double,12>{ 2, 0, 1,
                                 5, 3, 4,
                                 8, 6, 7,
                                11, 9,10}), *(array<double,12>*)v1.data());
}

TEST(marray, front_back)
{
    marray<double,1> v1({8}, 0);

    EXPECT_EQ(v1.data(), &v1.cfront());
    EXPECT_EQ(v1.data(), &v1.front());
    EXPECT_EQ(v1.data(), &(const_cast<const marray<double,1>&>(v1).front()));
    EXPECT_EQ(v1.data()+7, &v1.cback());
    EXPECT_EQ(v1.data()+7, &v1.back());
    EXPECT_EQ(v1.data()+7, &(const_cast<const marray<double,1>&>(v1).back()));

    marray<double,3> v2{4, 2, 5};

    auto v3 = v2.cfront(0);
    EXPECT_EQ((array<len_type,2>{2, 5}), v3.lengths());
    EXPECT_EQ((array<stride_type,2>{5, 1}), v3.strides());
    EXPECT_EQ(v2.data(), v3.data());

    auto v4 = v2.front(1);
    EXPECT_EQ((array<len_type,2>{4, 5}), v4.lengths());
    EXPECT_EQ((array<stride_type,2>{10, 1}), v4.strides());
    EXPECT_EQ(v2.data(), v4.data());

    auto v5 = const_cast<const marray<double,3>&>(v2).front(1);
    EXPECT_EQ((array<len_type,2>{4, 5}), v5.lengths());
    EXPECT_EQ((array<stride_type,2>{10, 1}), v5.strides());
    EXPECT_EQ(v2.data(), v5.data());

    auto v6 = v2.cback(0);
    EXPECT_EQ((array<len_type,2>{2, 5}), v6.lengths());
    EXPECT_EQ((array<stride_type,2>{5, 1}), v6.strides());
    EXPECT_EQ(v2.data() + 30, v6.data());

    auto v7 = v2.back(1);
    EXPECT_EQ((array<len_type,2>{4, 5}), v7.lengths());
    EXPECT_EQ((array<stride_type,2>{10, 1}), v7.strides());
    EXPECT_EQ(v2.data() + 5, v7.data());

    auto v8 = const_cast<const marray<double,3>&>(v2).back(1);
    EXPECT_EQ((array<len_type,2>{4, 5}), v8.lengths());
    EXPECT_EQ((array<stride_type,2>{10, 1}), v8.strides());
    EXPECT_EQ(v2.data() + 5, v8.data());
}

TEST(varray, front_back)
{
    marray<double> v2{4, 2, 5};

    auto v3 = v2.cfront(0);
    EXPECT_EQ((len_vector{2, 5}), v3.lengths());
    EXPECT_EQ((stride_vector{5, 1}), v3.strides());
    EXPECT_EQ(v2.data(), v3.data());

    auto v4 = v2.front(1);
    EXPECT_EQ((len_vector{4, 5}), v4.lengths());
    EXPECT_EQ((stride_vector{10, 1}), v4.strides());
    EXPECT_EQ(v2.data(), v4.data());

    auto v5 = const_cast<const marray<double>&>(v2).front(1);
    EXPECT_EQ((len_vector{4, 5}), v5.lengths());
    EXPECT_EQ((stride_vector{10, 1}), v5.strides());
    EXPECT_EQ(v2.data(), v5.data());

    auto v6 = v2.cback(0);
    EXPECT_EQ((len_vector{2, 5}), v6.lengths());
    EXPECT_EQ((stride_vector{5, 1}), v6.strides());
    EXPECT_EQ(v2.data() + 30, v6.data());

    auto v7 = v2.back(1);
    EXPECT_EQ((len_vector{4, 5}), v7.lengths());
    EXPECT_EQ((stride_vector{10, 1}), v7.strides());
    EXPECT_EQ(v2.data() + 5, v7.data());

    auto v8 = const_cast<const marray<double>&>(v2).back(1);
    EXPECT_EQ((len_vector{4, 5}), v8.lengths());
    EXPECT_EQ((stride_vector{10, 1}), v8.strides());
    EXPECT_EQ(v2.data() + 5, v8.data());
}

TEST(marray, access)
{
    array<double,12> data = { 0, 1, 2,
                              3, 4, 5,
                              6, 7, 8,
                              9,10,11};

    marray<double,2> v1(marray_view<const double,2>({4, 3}, data.data()));

    EXPECT_EQ( 0, v1(0, 0));
    EXPECT_EQ( 5, v1(1, 2));
    EXPECT_EQ(10, v1(3, 1));
    EXPECT_EQ(10, (const_cast<const marray<double,2>&>(v1)(3, 1)));

    EXPECT_EQ( 0, v1[0][0]);
    EXPECT_EQ( 5, v1[1][2]);
    EXPECT_EQ(10, v1[3][1]);
    EXPECT_EQ(10, (const_cast<const marray<double,2>&>(v1)[3][1]));

    auto v2 = view(v1(slice::all, range(2)));
    EXPECT_EQ((array<len_type,2>{4, 2}), v2.lengths());
    EXPECT_EQ((array<stride_type,2>{3, 1}), v2.strides());
    EXPECT_EQ(v1.data(), v2.data());

    auto v3 = view(v1(range(0, 4, 2), 1));
    EXPECT_EQ((array<len_type,1>{2}), v3.lengths());
    EXPECT_EQ((array<stride_type,1>{6}), v3.strides());
    EXPECT_EQ(v1.data() + 1, v3.data());

    auto v4 = view(v1[slice::all][range(2)]);
    EXPECT_EQ((array<len_type,2>{4, 2}), v4.lengths());
    EXPECT_EQ((array<stride_type,2>{3, 1}), v4.strides());
    EXPECT_EQ(v1.data(), v4.data());

    auto v5 = view(v1[range(1,3)]);
    EXPECT_EQ((array<len_type,2>{2,3}), v5.lengths());
    EXPECT_EQ((array<stride_type,2>{3,1}), v5.strides());
    EXPECT_EQ(v1.data() + 3, v5.data());

    auto v6 = view(v1[2]);
    EXPECT_EQ((array<len_type,1>{3}), v6.lengths());
    EXPECT_EQ((array<stride_type,1>{1}), v6.strides());
    EXPECT_EQ(v1.data() + 6, v6.data());

    auto v7 = view(const_cast<const marray<double,2>&>(v1)(slice::all, range(2)));
    EXPECT_EQ((array<len_type,2>{4, 2}), v7.lengths());
    EXPECT_EQ((array<stride_type,2>{3, 1}), v7.strides());
    EXPECT_EQ(v1.data(), v7.data());

    auto v8 = view(const_cast<const marray<double,2>&>(v1)(range(0, 4, 2), 1));
    EXPECT_EQ((array<len_type,1>{2}), v8.lengths());
    EXPECT_EQ((array<stride_type,1>{6}), v8.strides());
    EXPECT_EQ(v1.data() + 1, v8.data());

    auto v9 = view(const_cast<const marray<double,2>&>(v1)[slice::all][range(2)]);
    EXPECT_EQ((array<len_type,2>{4, 2}), v9.lengths());
    EXPECT_EQ((array<stride_type,2>{3, 1}), v9.strides());
    EXPECT_EQ(v1.data(), v9.data());

    auto v10 = view(const_cast<const marray<double,2>&>(v1)[range(1,3)]);
    EXPECT_EQ((array<len_type,2>{2,3}), v10.lengths());
    EXPECT_EQ((array<stride_type,2>{3,1}), v10.strides());
    EXPECT_EQ(v1.data() + 3, v10.data());

    auto v11 = view(const_cast<const marray<double,2>&>(v1)[2]);
    EXPECT_EQ((array<len_type,1>{3}), v11.lengths());
    EXPECT_EQ((array<stride_type,1>{1}), v11.strides());
    EXPECT_EQ(v1.data() + 6, v11.data());
}

TEST(varray, access)
{
    array<double,12> data = { 0, 1, 2,
                              3, 4, 5,
                              6, 7, 8,
                              9,10,11};

    marray<double> v1(marray_view<const double>({4, 3}, data.data()));

    EXPECT_EQ( 0, v1(0, 0));
    EXPECT_EQ( 5, v1(1, 2));
    EXPECT_EQ(10, v1(3, 1));
    EXPECT_EQ(10, const_cast<const marray<double>&>(v1)(3, 1));

    auto v2 = view<DYNAMIC>(v1(slice::all, range(2)));
    EXPECT_EQ((len_vector{4, 2}), v2.lengths());
    EXPECT_EQ((stride_vector{3, 1}), v2.strides());
    EXPECT_EQ(v1.data(), v2.data());

    auto v3 = view<DYNAMIC>(v1(range(0, 4, 2), 1));
    EXPECT_EQ((len_vector{2}), v3.lengths());
    EXPECT_EQ((stride_vector{6}), v3.strides());
    EXPECT_EQ(v1.data() + 1, v3.data());

    auto v4 = view<DYNAMIC>(const_cast<const marray<double>&>(v1)(slice::all, range(2)));
    EXPECT_EQ((len_vector{4, 2}), v4.lengths());
    EXPECT_EQ((stride_vector{3, 1}), v4.strides());
    EXPECT_EQ(v1.data(), v4.data());

    auto v5 = view<DYNAMIC>(const_cast<const marray<double>&>(v1)(range(0, 4, 2), 1));
    EXPECT_EQ((len_vector{2}), v5.lengths());
    EXPECT_EQ((stride_vector{6}), v5.strides());
    EXPECT_EQ(v1.data() + 1, v5.data());
}

TEST(marray, iteration)
{
    array<array<int,3>,4> visited;
    array<array<double,3>,4> data = {{{ 0, 1, 2},
                                      { 3, 4, 5},
                                      { 6, 7, 8},
                                      { 9,10,11}}};

    marray<double,2> v1 = {{{ 0, 1, 2},
                            { 3, 4, 5},
                            { 6, 7, 8},
                            { 9,10,11}}};
    const marray<double,2> v2(v1);

    visited = {};
    v1.for_each_element(
    [&](double& v, len_type i, len_type j)
    {
        EXPECT_GE(i, 0);
        EXPECT_LT(i, 4);
        EXPECT_GE(j, 0);
        EXPECT_LT(j, 3);
        EXPECT_EQ(v, data[i][j]);
        visited[i][j]++;
    });

    for (len_type i = 0;i < 4;i++)
    {
        for (len_type j = 0;j < 3;j++)
        {
            EXPECT_EQ(visited[i][j], 1);
        }
    }

    visited = {};
    v2.for_each_element(
    [&](const double& v, len_type i, len_type j)
    {
        EXPECT_GE(i, 0);
        EXPECT_LT(i, 4);
        EXPECT_GE(j, 0);
        EXPECT_LT(j, 3);
        EXPECT_EQ(v, data[i][j]);
        visited[i][j]++;
    });

    for (len_type i = 0;i < 4;i++)
    {
        for (len_type j = 0;j < 3;j++)
        {
            EXPECT_EQ(visited[i][j], 1);
        }
    }
}

TEST(varray, iteration)
{
    array<array<int,3>,4> visited;
    array<array<double,3>,4> data = {{{ 0, 1, 2},
                                      { 3, 4, 5},
                                      { 6, 7, 8},
                                      { 9,10,11}}};

    marray<double> v1{4, 3};
    copy_n(&data[0][0], 12, v1.data());
    const marray<double> v2(v1);

    visited = {};
    v1.for_each_element(
    [&](double& v, const len_vector& pos)
    {
        EXPECT_EQ(pos.size(), 2u);
        len_type i = pos[0];
        len_type j = pos[1];
        EXPECT_GE(i, 0);
        EXPECT_LT(i, 4);
        EXPECT_GE(j, 0);
        EXPECT_LT(j, 3);
        EXPECT_EQ(v, data[i][j]);
        visited[i][j]++;
    });

    for (len_type i = 0;i < 4;i++)
    {
        for (len_type j = 0;j < 3;j++)
        {
            EXPECT_EQ(visited[i][j], 1);
        }
    }

    visited = {};
    v2.for_each_element(
    [&](const double& v, const len_vector& pos)
    {
        EXPECT_EQ(pos.size(), 2u);
        len_type i = pos[0];
        len_type j = pos[1];
        EXPECT_GE(i, 0);
        EXPECT_LT(i, 4);
        EXPECT_GE(j, 0);
        EXPECT_LT(j, 3);
        EXPECT_EQ(v, data[i][j]);
        visited[i][j]++;
    });

    for (len_type i = 0;i < 4;i++)
    {
        for (len_type j = 0;j < 3;j++)
        {
            EXPECT_EQ(visited[i][j], 1);
        }
    }

    visited = {};
    v1.for_each_element<2>(
    [&](double& v, len_type i, len_type j)
    {
        EXPECT_GE(i, 0);
        EXPECT_LT(i, 4);
        EXPECT_GE(j, 0);
        EXPECT_LT(j, 3);
        EXPECT_EQ(v, data[i][j]);
        visited[i][j]++;
    });

    for (len_type i = 0;i < 4;i++)
    {
        for (len_type j = 0;j < 3;j++)
        {
            EXPECT_EQ(visited[i][j], 1);
        }
    }

    visited = {};
    v2.for_each_element<2>(
    [&](const double& v, len_type i, len_type j)
    {
        EXPECT_GE(i, 0);
        EXPECT_LT(i, 4);
        EXPECT_GE(j, 0);
        EXPECT_LT(j, 3);
        EXPECT_EQ(v, data[i][j]);
        visited[i][j]++;
    });

    for (len_type i = 0;i < 4;i++)
    {
        for (len_type j = 0;j < 3;j++)
        {
            EXPECT_EQ(visited[i][j], 1);
        }
    }
}

TEST(marray, swap)
{
    marray<double,3> v1{4, 2, 5};
    marray<double,3> v2{3, 8, 3};

    auto data1 = v1.data();
    auto data2 = v2.data();

    v1.swap(v2);

    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v2.strides());
    EXPECT_EQ(data1, v2.data());
    EXPECT_EQ((array<len_type,3>{3, 8, 3}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{24, 3, 1}), v1.strides());
    EXPECT_EQ(data2, v1.data());

    swap(v2, v1);

    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());
    EXPECT_EQ(data1, v1.data());
    EXPECT_EQ((array<len_type,3>{3, 8, 3}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{24, 3, 1}), v2.strides());
    EXPECT_EQ(data2, v2.data());
}

TEST(varray, swap)
{
    marray<double> v1{4, 2, 5};
    marray<double> v2{3, 8};

    auto data1 = v1.data();
    auto data2 = v2.data();

    v1.swap(v2);

    EXPECT_EQ((len_vector{4, 2, 5}), v2.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v2.strides());
    EXPECT_EQ(data1, v2.data());
    EXPECT_EQ((len_vector{3, 8}), v1.lengths());
    EXPECT_EQ((stride_vector{8, 1}), v1.strides());
    EXPECT_EQ(data2, v1.data());

    swap(v2, v1);

    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());
    EXPECT_EQ(data1, v1.data());
    EXPECT_EQ((len_vector{3, 8}), v2.lengths());
    EXPECT_EQ((stride_vector{8, 1}), v2.strides());
    EXPECT_EQ(data2, v2.data());
}

TEST(marray, print)
{
    marray<double,3> v1 =
    {
     {
      {0, 1, 2},
      {3, 4, 5}
     },
     {
      {6, 7, 8},
      {9, 10, 11}
     }
    };

    std::string expected =
R"XXX({
 {
  {0, 1, 2},
  {3, 4, 5}
 },
 {
  {6, 7, 8},
  {9, 10, 11}
 }
})XXX";

    std::ostringstream oss;
    oss << v1;

    EXPECT_EQ(oss.str(), expected);
}

TEST(varray, print)
{
    double data[2][2][3] =
    {
     {
      {0, 1, 2},
      {3, 4, 5}
     },
     {
      {6, 7, 8},
      {9, 10, 11}
     }
    };

    marray_view<double> v0({2,2,3}, (double*)data, ROW_MAJOR);
    marray<double> v1(v0, ROW_MAJOR);

    string expected =
R"XXX({
 {
  {0, 1, 2},
  {3, 4, 5}
 },
 {
  {6, 7, 8},
  {9, 10, 11}
 }
})XXX";

    ostringstream oss;
    oss << v1;

    EXPECT_EQ(oss.str(), expected);
}
