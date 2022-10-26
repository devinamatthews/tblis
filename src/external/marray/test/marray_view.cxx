#include "marray.hpp"
#include "rotate.hpp"
#include "gtest/gtest.h"

using namespace std;
using namespace MArray;
using namespace MArray::slice;

TEST(marray_view, constructor)
{
    double tmp;
    double* data = &tmp;

    marray<double,3> v0{4, 2, 5};
    marray<double,4> v01{3, 4, 2, 5};

    marray_view<double,2> v1;
    EXPECT_EQ(nullptr, v1.data());
    EXPECT_EQ((array<len_type,2>{0, 0}), v1.lengths());
    EXPECT_EQ((array<stride_type,2>{0, 0}), v1.strides());

    marray_view<double,3> v2({4, 2, 5}, data);
    EXPECT_EQ(data, v2.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v2.strides());

    marray_view<double,3> v3(array<char,3>{4, 2, 5}, data);
    EXPECT_EQ(data, v3.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v3.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v3.strides());

    marray_view<double,3> v4({4, 2, 5}, data, COLUMN_MAJOR);
    EXPECT_EQ(data, v4.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v4.lengths());
    EXPECT_EQ((array<stride_type,3>{1, 4, 8}), v4.strides());

    marray_view<double,3> v5(v2);
    EXPECT_EQ(data, v5.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v5.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v5.strides());

    marray_view<double,3> v51(v0);
    EXPECT_EQ(v0.data(), v51.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v51.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v51.strides());

    marray_view<double,3> v52(v01[0]);
    EXPECT_EQ(v01.data(), v52.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v52.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v52.strides());

    marray_view<double,3> v6(marray_view<double,3>({4, 2, 5}, data));
    EXPECT_EQ(data, v6.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v6.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v6.strides());

    marray_view<double,3> v7({4, 2, 5}, data, {3, 8, 24});
    EXPECT_EQ(data, v7.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v7.lengths());
    EXPECT_EQ((array<stride_type,3>{3, 8, 24}), v7.strides());

    marray_view<double,3> v8(array<char,3>{4, 2, 5}, data, array<char,3>{3, 8, 24});
    EXPECT_EQ(data, v8.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v8.lengths());
    EXPECT_EQ((array<stride_type,3>{3, 8, 24}), v8.strides());

    marray_view<const double,2> v9;
    EXPECT_EQ(nullptr, v9.data());
    EXPECT_EQ((array<len_type,2>{0, 0}), v9.lengths());
    EXPECT_EQ((array<stride_type,2>{0, 0}), v9.strides());

    marray_view<const double,3> v10({4, 2, 5}, data);
    EXPECT_EQ(data, v10.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v10.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v10.strides());

    marray_view<const double,3> v11(array<char,3>{4, 2, 5}, data);
    EXPECT_EQ(data, v11.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v11.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v11.strides());

    marray_view<const double,3> v12({4, 2, 5}, data, COLUMN_MAJOR);
    EXPECT_EQ(data, v12.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v12.lengths());
    EXPECT_EQ((array<stride_type,3>{1, 4, 8}), v12.strides());

    marray_view<const double,3> v13(v2);
    EXPECT_EQ(data, v13.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v13.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v13.strides());

    marray_view<const double,3> v131(v0);
    EXPECT_EQ(v0.data(), v131.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v131.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v131.strides());

    marray_view<const double,3> v132(v01[0]);
    EXPECT_EQ(v01.data(), v132.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v132.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v132.strides());

    marray_view<const double,3> v14(v10);
    EXPECT_EQ(data, v14.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v14.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v14.strides());

    marray_view<const double,3> v15(marray_view<double,3>({4, 2, 5}, data));
    EXPECT_EQ(data, v15.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v15.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v15.strides());

    marray_view<const double,3> v16(marray_view<const double,3>({4, 2, 5}, data));
    EXPECT_EQ(data, v16.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v16.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v16.strides());

    marray_view<const double,3> v17({4, 2, 5}, data, {3, 8, 24});
    EXPECT_EQ(data, v17.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v17.lengths());
    EXPECT_EQ((array<stride_type,3>{3, 8, 24}), v17.strides());

    marray_view<const double,3> v18(array<char,3>{4, 2, 5}, data, array<char,3>{3, 8, 24});
    EXPECT_EQ(data, v18.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v18.lengths());
    EXPECT_EQ((array<stride_type,3>{3, 8, 24}), v18.strides());
}

TEST(varray_view, constructor)
{
    double tmp;
    double* data = &tmp;

    marray<double> v0{4, 2, 5};

    marray_view<double> v1;
    EXPECT_EQ(0u, v1.dimension());
    EXPECT_EQ(nullptr, v1.data());

    marray_view<double> v2({4, 2, 5}, data);
    EXPECT_EQ(3u, v2.dimension());
    EXPECT_EQ(data, v2.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v2.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v2.strides());

    marray_view<double> v3(vector<char>{4, 2, 5}, data);
    EXPECT_EQ(3u, v3.dimension());
    EXPECT_EQ(data, v3.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v3.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v3.strides());

    marray_view<double> v4({4, 2, 5}, data, COLUMN_MAJOR);
    EXPECT_EQ(3u, v4.dimension());
    EXPECT_EQ(data, v4.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v4.lengths());
    EXPECT_EQ((stride_vector{1, 4, 8}), v4.strides());

    marray_view<double> v5(v2);
    EXPECT_EQ(3u, v5.dimension());
    EXPECT_EQ(data, v5.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v5.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v5.strides());

    marray_view<double> v51(v0);
    EXPECT_EQ(3u, v51.dimension());
    EXPECT_EQ(v0.data(), v51.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v51.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v51.strides());

    marray_view<double> v6(marray_view<double>({4, 2, 5}, data));
    EXPECT_EQ(3u, v6.dimension());
    EXPECT_EQ(data, v6.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v6.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v6.strides());

    marray_view<double> v7({4, 2, 5}, data, {3, 8, 24});
    EXPECT_EQ(3u, v7.dimension());
    EXPECT_EQ(data, v7.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v7.lengths());
    EXPECT_EQ((stride_vector{3, 8, 24}), v7.strides());

    marray_view<double> v8(vector<char>{4, 2, 5}, data, vector<char>{3, 8, 24});
    EXPECT_EQ(3u, v8.dimension());
    EXPECT_EQ(data, v8.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v8.lengths());
    EXPECT_EQ((stride_vector{3, 8, 24}), v8.strides());

    marray_view<const double> v9;
    EXPECT_EQ(0u, v9.dimension());
    EXPECT_EQ(nullptr, v9.data());

    marray_view<const double> v10({4, 2, 5}, data);
    EXPECT_EQ(3u, v10.dimension());
    EXPECT_EQ(data, v10.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v10.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v10.strides());

    marray_view<const double> v11(vector<char>{4, 2, 5}, data);
    EXPECT_EQ(3u, v11.dimension());
    EXPECT_EQ(data, v11.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v11.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v11.strides());

    marray_view<const double> v12({4, 2, 5}, data, COLUMN_MAJOR);
    EXPECT_EQ(3u, v12.dimension());
    EXPECT_EQ(data, v12.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v12.lengths());
    EXPECT_EQ((stride_vector{1, 4, 8}), v12.strides());

    marray_view<const double> v13(v2);
    EXPECT_EQ(3u, v13.dimension());
    EXPECT_EQ(data, v13.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v13.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v13.strides());

    marray_view<const double> v131(v0);
    EXPECT_EQ(3u, v131.dimension());
    EXPECT_EQ(v0.data(), v131.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v131.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v131.strides());

    marray_view<const double> v14(v10);
    EXPECT_EQ(3u, v14.dimension());
    EXPECT_EQ(data, v14.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v14.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v14.strides());

    marray_view<const double> v15(marray_view<double>({4, 2, 5}, data));
    EXPECT_EQ(3u, v15.dimension());
    EXPECT_EQ(data, v15.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v15.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v15.strides());

    marray_view<const double> v16(marray_view<const double>({4, 2, 5}, data));
    EXPECT_EQ(3u, v16.dimension());
    EXPECT_EQ(data, v16.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v16.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v16.strides());

    marray_view<const double> v17({4, 2, 5}, data, {3, 8, 24});
    EXPECT_EQ(3u, v17.dimension());
    EXPECT_EQ(data, v17.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v17.lengths());
    EXPECT_EQ((stride_vector{3, 8, 24}), v17.strides());

    marray_view<const double> v18(vector<char>{4, 2, 5}, data, vector<char>{3, 8, 24});
    EXPECT_EQ(3u, v18.dimension());
    EXPECT_EQ(data, v18.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v18.lengths());
    EXPECT_EQ((stride_vector{3, 8, 24}), v18.strides());
}

TEST(marray_view, reset)
{
    double tmp;
    double* data = &tmp;

    marray_view<double,3> v1;
    marray_view<const double,3> v2;
    marray_view<double,3> v3({4, 2, 5}, data);
    marray_view<const double,3> v4({4, 2, 5}, data);
    marray<double,3> v0{4, 2, 5};
    marray<double,4> v01{3, 4, 2, 5};

    v1.reset({4, 2, 5}, data);
    EXPECT_EQ(data, v1.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());

    v1.reset(array<char,3>{4, 2, 5}, data);
    EXPECT_EQ(data, v1.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());

    v1.reset({4, 2, 5}, data, COLUMN_MAJOR);
    EXPECT_EQ(data, v1.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{1, 4, 8}), v1.strides());

    v1.reset(v3);
    EXPECT_EQ(data, v1.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());

    v1.reset(v0);
    EXPECT_EQ(v0.data(), v1.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());

    v1.reset(v01[0]);
    EXPECT_EQ(v01.data(), v1.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());

    v1.reset(marray_view<double,3>({4, 2, 5}, data));
    EXPECT_EQ(data, v1.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v1.strides());

    v1.reset({4, 2, 5}, data, {3, 8, 24});
    EXPECT_EQ(data, v1.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{3, 8, 24}), v1.strides());

    v1.reset(array<char,3>{4, 2, 5}, data, array<char,3>{3, 8, 24});
    EXPECT_EQ(data, v1.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{3, 8, 24}), v1.strides());

    v1.reset();
    EXPECT_EQ(nullptr, v1.data());
    EXPECT_EQ((array<len_type,3>{0, 0, 0}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{0, 0, 0}), v1.strides());

    v2.reset({4, 2, 5}, data);
    EXPECT_EQ(data, v2.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v2.strides());

    v2.reset(array<char,3>{4, 2, 5}, data);
    EXPECT_EQ(data, v2.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v2.strides());

    v2.reset({4, 2, 5}, data, COLUMN_MAJOR);
    EXPECT_EQ(data, v2.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{1, 4, 8}), v2.strides());

    v2.reset(v3);
    EXPECT_EQ(data, v2.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v2.strides());

    v2.reset(v0);
    EXPECT_EQ(v0.data(), v2.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v2.strides());

    v2.reset(v01[0]);
    EXPECT_EQ(v01.data(), v2.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v2.strides());

    v2.reset(v4);
    EXPECT_EQ(data, v2.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v2.strides());

    v2.reset(marray_view<double,3>({4, 2, 5}, data));
    EXPECT_EQ(data, v2.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v2.strides());

    v2.reset(marray_view<const double,3>({4, 2, 5}, data));
    EXPECT_EQ(data, v2.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), v2.strides());

    v2.reset({4, 2, 5}, data, {3, 8, 24});
    EXPECT_EQ(data, v2.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{3, 8, 24}), v2.strides());

    v2.reset(array<char,3>{4, 2, 5}, data, array<char,3>{3, 8, 24});
    EXPECT_EQ(data, v2.data());
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{3, 8, 24}), v2.strides());

    v2.reset();
    EXPECT_EQ(nullptr, v1.data());
    EXPECT_EQ((array<len_type,3>{0, 0, 0}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{0, 0, 0}), v2.strides());
}

TEST(varray_view, reset)
{
    double tmp;
    double* data = &tmp;

    marray_view<double> v1;
    marray_view<double> v2({4, 2, 5}, data);
    marray<double> v0{4, 2, 5};

    v1.reset({4, 2, 5}, data);
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ(data, v1.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());

    v1.reset(vector<char>{4, 2, 5}, data);
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ(data, v1.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());

    v1.reset({4, 2, 5}, data, COLUMN_MAJOR);
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ(data, v1.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{1, 4, 8}), v1.strides());

    v1.reset(v2);
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ(data, v1.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());

    v1.reset(v0);
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ(v0.data(), v1.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());

    v1.reset(marray_view<double>({4, 2, 5}, data));
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ(data, v1.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());

    v1.reset({4, 2, 5}, data, {3, 8, 24});
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ(data, v1.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{3, 8, 24}), v1.strides());

    v1.reset(vector<char>{4, 2, 5}, data, vector<char>{3, 8, 24});
    EXPECT_EQ(3u, v1.dimension());
    EXPECT_EQ(data, v1.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{3, 8, 24}), v1.strides());

    v1.reset();
    EXPECT_EQ(0u, v1.dimension());
    EXPECT_EQ(nullptr, v1.data());

    marray_view<const double> v9;
    marray_view<const double> v10({4, 2, 5}, data);

    EXPECT_EQ(0u, v9.dimension());
    EXPECT_EQ(nullptr, v9.data());

    v9.reset({4, 2, 5}, data);
    EXPECT_EQ(3u, v9.dimension());
    EXPECT_EQ(data, v9.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v9.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v9.strides());

    v9.reset(vector<char>{4, 2, 5}, data);
    EXPECT_EQ(3u, v9.dimension());
    EXPECT_EQ(data, v9.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v9.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v9.strides());

    v9.reset({4, 2, 5}, data, COLUMN_MAJOR);
    EXPECT_EQ(3u, v9.dimension());
    EXPECT_EQ(data, v9.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v9.lengths());
    EXPECT_EQ((stride_vector{1, 4, 8}), v9.strides());

    v9.reset(v2);
    EXPECT_EQ(3u, v9.dimension());
    EXPECT_EQ(data, v9.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v9.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v9.strides());

    v9.reset(v0);
    EXPECT_EQ(3u, v9.dimension());
    EXPECT_EQ(v0.data(), v9.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v9.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v9.strides());

    v9.reset(v10);
    EXPECT_EQ(3u, v9.dimension());
    EXPECT_EQ(data, v9.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v9.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v9.strides());

    v9.reset(marray_view<double>({4, 2, 5}, data));
    EXPECT_EQ(3u, v9.dimension());
    EXPECT_EQ(data, v9.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v9.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v9.strides());

    v9.reset(marray_view<const double>({4, 2, 5}, data));
    EXPECT_EQ(3u, v9.dimension());
    EXPECT_EQ(data, v9.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v9.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v9.strides());

    v9.reset({4, 2, 5}, data, {3, 8, 24});
    EXPECT_EQ(3u, v9.dimension());
    EXPECT_EQ(data, v9.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v9.lengths());
    EXPECT_EQ((stride_vector{3, 8, 24}), v9.strides());

    v9.reset(vector<char>{4, 2, 5}, data, vector<char>{3, 8, 24});
    EXPECT_EQ(3u, v9.dimension());
    EXPECT_EQ(data, v9.data());
    EXPECT_EQ((len_vector{4, 2, 5}), v9.lengths());
    EXPECT_EQ((stride_vector{3, 8, 24}), v9.strides());

    v9.reset();
    EXPECT_EQ(0u, v9.dimension());
    EXPECT_EQ(nullptr, v9.data());
}

TEST(marray_view, initialize)
{
    std::array<double,18> data = {};

    marray_view<double,3> v1({3, 2, 3}, data.data(), ROW_MAJOR);
    marray_view<double,3> v2({3, 2, 3}, data.data(), COLUMN_MAJOR);

    v1 = {{{ 0, 1, 2},
           { 3, 4, 5}},
          {{ 6, 7, 8},
           { 9,10,11}},
          {{12,13,14},
           {15,16,17}}};

    EXPECT_EQ((std::array<double,18>{ 0, 1, 2, 3, 4, 5,
                                        6, 7, 8, 9,10,11,
                                       12,13,14,15,16,17}), data);

    v2 = {{{ 0, 1, 2},
           { 3, 4, 5}},
          {{ 6, 7, 8},
           { 9,10,11}},
          {{12,13,14},
           {15,16,17}}};

    EXPECT_EQ((std::array<double,18>{ 0, 6,12, 3, 9,15,
                                      1, 7,13, 4,10,16,
                                      2, 8,14, 5,11,17}), data);
}

TEST(marray_view, assign)
{
    double data1[6] = {0, 1, 2,
                      3, 4, 5};
    int data2[6] = {0, 1, 2,
                    3, 4, 5};

    marray<double> v0{2, 3};
    marray_view<double> v1(v0);

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

TEST(marray_view, shift)
{
    double tmp;
    double* data = &tmp;

    marray_view<double,3> v1_({4, 6, 5}, data);
    marray_view<double,3> v1 = v1_[range(2)][range(3,6)][range(2)];

    auto v2 = v1.shifted(0, 2);
    EXPECT_EQ(v1.lengths(), v2.lengths());
    EXPECT_EQ(v1.strides(), v2.strides());
    EXPECT_EQ(v1.data() + 2*v1.stride(0), v2.data());

    auto v3 = v1.shifted(1, -3);
    EXPECT_EQ(v1.lengths(), v3.lengths());
    EXPECT_EQ(v1.strides(), v3.strides());
    EXPECT_EQ(v1.data() - 3*v1.stride(1), v3.data());

    auto v4 = v1.shifted_down(2);
    EXPECT_EQ(v1.lengths(), v4.lengths());
    EXPECT_EQ(v1.strides(), v4.strides());
    EXPECT_EQ(v1.data() + v1.length(2)*v1.stride(2), v4.data());

    auto v5 = v2.shifted_up(0);
    EXPECT_EQ(v1.lengths(), v5.lengths());
    EXPECT_EQ(v1.strides(), v5.strides());
    EXPECT_EQ(v2.data() - v1.length(0)*v1.stride(0), v5.data());

    auto ref = v1.data();
    v1.shift(0, 2);
    EXPECT_EQ(ref + 2*v1.stride(0), v1.data());

    ref = v1.data();
    v1.shift(1, -3);
    EXPECT_EQ(ref - 3*v1.stride(1), v1.data());

    ref = v1.data();
    v1.shift_down(2);
    EXPECT_EQ(ref + v1.length(2)*v1.stride(2), v1.data());

    ref = v1.data();
    v1.shift_up(0);
    EXPECT_EQ(ref - v1.length(0)*v1.stride(0), v1.data());
}

TEST(varray_view, shift)
{
    double tmp;
    double* data = &tmp;

    marray_view<double> v1_({4, 6, 5}, data);
    marray_view<double> v1 = v1_(range(2),range(3,6),range(2));

    auto v2 = v1.shifted(0, 2);
    EXPECT_EQ(v1.lengths(), v2.lengths());
    EXPECT_EQ(v1.strides(), v2.strides());
    EXPECT_EQ(v1.data() + 2*v1.stride(0), v2.data());

    auto v3 = v1.shifted(1, -3);
    EXPECT_EQ(v1.lengths(), v3.lengths());
    EXPECT_EQ(v1.strides(), v3.strides());
    EXPECT_EQ(v1.data() - 3*v1.stride(1), v3.data());

    auto v4 = v1.shifted_down(2);
    EXPECT_EQ(v1.lengths(), v4.lengths());
    EXPECT_EQ(v1.strides(), v4.strides());
    EXPECT_EQ(v1.data() + v1.length(2)*v1.stride(2), v4.data());

    auto v5 = v2.shifted_up(0);
    EXPECT_EQ(v1.lengths(), v5.lengths());
    EXPECT_EQ(v1.strides(), v5.strides());
    EXPECT_EQ(v2.data() - v1.length(0)*v1.stride(0), v5.data());

    auto ref = v1.data();
    v1.shift(0, 2);
    EXPECT_EQ(ref + 2*v1.stride(0), v1.data());

    ref = v1.data();
    v1.shift(1, -3);
    EXPECT_EQ(ref - 3*v1.stride(1), v1.data());

    ref = v1.data();
    v1.shift_down(2);
    EXPECT_EQ(ref + v1.length(2)*v1.stride(2), v1.data());

    ref = v1.data();
    v1.shift_up(0);
    EXPECT_EQ(ref - v1.length(0)*v1.stride(0), v1.data());
}

TEST(marray_view, permute)
{
    double tmp;
    double* data = &tmp;

    marray_view<double,3> v1({4, 2, 5}, data);

    auto v2 = v1.permuted({1, 0, 2});
    EXPECT_EQ((array<len_type,3>{2, 4, 5}), v2.lengths());
    EXPECT_EQ((array<stride_type,3>{5, 10, 1}), v2.strides());
    EXPECT_EQ(v1.data(), v2.data());

    auto v3 = v1.permuted(array<char,3>{2, 0, 1});
    EXPECT_EQ((array<len_type,3>{5, 4, 2}), v3.lengths());
    EXPECT_EQ((array<stride_type,3>{1, 10, 5}), v3.strides());
    EXPECT_EQ(v1.data(), v3.data());

    v1.permute({1, 0, 2});
    EXPECT_EQ((array<len_type,3>{2, 4, 5}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{5, 10, 1}), v1.strides());
    EXPECT_EQ(data, v1.data());

    v1.permute(array<char,3>{2, 0, 1});
    EXPECT_EQ((array<len_type,3>{5, 2, 4}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{1, 5, 10}), v1.strides());
    EXPECT_EQ(data, v1.data());

    v1.permute(0, 2, 1);
    EXPECT_EQ((array<len_type,3>{5, 4, 2}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{1, 10, 5}), v1.strides());
    EXPECT_EQ(data, v1.data());
}

TEST(varray_view, permute)
{
    double tmp;
    double* data = &tmp;

    marray_view<double> v1({4, 2, 5}, data);

    auto v2 = v1.permuted({1, 0, 2});
    EXPECT_EQ((len_vector{2, 4, 5}), v2.lengths());
    EXPECT_EQ((stride_vector{5, 10, 1}), v2.strides());
    EXPECT_EQ(v1.data(), v2.data());

    auto v3 = v1.permuted(vector<char>{2, 0, 1});
    EXPECT_EQ((len_vector{5, 4, 2}), v3.lengths());
    EXPECT_EQ((stride_vector{1, 10, 5}), v3.strides());
    EXPECT_EQ(v1.data(), v3.data());

    v1.permute({1, 0, 2});
    EXPECT_EQ((len_vector{2, 4, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{5, 10, 1}), v1.strides());
    EXPECT_EQ(data, v1.data());

    v1.permute(vector<char>{2, 0, 1});
    EXPECT_EQ((len_vector{5, 2, 4}), v1.lengths());
    EXPECT_EQ((stride_vector{1, 5, 10}), v1.strides());
    EXPECT_EQ(data, v1.data());

    v1.permute(0, 2, 1);
    EXPECT_EQ((len_vector{5, 4, 2}), v1.lengths());
    EXPECT_EQ((stride_vector{1, 10, 5}), v1.strides());
    EXPECT_EQ(data, v1.data());
}

TEST(marray_view, transpose)
{
    double tmp;
    double* data = &tmp;

    marray_view<double,2> v1({4, 8}, data);

    auto v2 = v1.transposed();
    EXPECT_EQ((array<len_type,2>{8, 4}), v2.lengths());
    EXPECT_EQ((array<stride_type,2>{1, 8}), v2.strides());
    EXPECT_EQ(v1.data(), v2.data());

    auto v3 = v1.T();
    EXPECT_EQ((array<len_type,2>{8, 4}), v3.lengths());
    EXPECT_EQ((array<stride_type,2>{1, 8}), v3.strides());
    EXPECT_EQ(v1.data(), v3.data());

    v1.transpose();
    EXPECT_EQ((array<len_type,2>{8, 4}), v1.lengths());
    EXPECT_EQ((array<stride_type,2>{1, 8}), v1.strides());
    EXPECT_EQ(data, v1.data());
}

TEST(marray_view, lowered)
{
    double tmp;
    double* data = &tmp;

    marray_view<double,3> v1({4, 2, 5}, data);

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
}

TEST(marray_view, lower)
{
    double tmp;
    double* data = &tmp;

    marray_view<double> v1({4, 2, 5}, data);

    auto v2 = v1.lowered({1, 2});
    EXPECT_EQ((len_vector{4, 2, 5}), v2.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v2.strides());
    EXPECT_EQ(v1.data(), v2.data());

    auto v3 = v1.lowered(vector<char>{1});
    EXPECT_EQ((len_vector{4, 10}), v3.lengths());
    EXPECT_EQ((stride_vector{10, 1}), v3.strides());
    EXPECT_EQ(v1.data(), v3.data());

    auto v4 = v1.lowered({});
    EXPECT_EQ((len_vector{40}), v4.lengths());
    EXPECT_EQ((stride_vector{1}), v4.strides());
    EXPECT_EQ(v1.data(), v4.data());

    v1.lower({1, 2});
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());
    EXPECT_EQ(data, v1.data());

    v1.lower(vector<char>{1});
    EXPECT_EQ((len_vector{4, 10}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 1}), v1.strides());
    EXPECT_EQ(data, v1.data());

    v1.lower({});
    EXPECT_EQ((len_vector{40}), v1.lengths());
    EXPECT_EQ((stride_vector{1}), v1.strides());
    EXPECT_EQ(data, v1.data());
}

TEST(marray_view, fix)
{
    marray<double> v1{4, 2, 5};
    marray<double> v2_{4, 2, 5};
    marray_view<double> v2(v2_);

    auto m1 = v1.view<3>();
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), m1.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), m1.strides());
    EXPECT_EQ(v1.data(), m1.data());

    auto m2 = v2.view<3>();
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), m2.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), m2.strides());
    EXPECT_EQ(v2.data(), m2.data());

    auto m3 = view<3>(v1);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), m3.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), m3.strides());
    EXPECT_EQ(v1.data(), m3.data());

    auto m4= view<3>(v2);
    EXPECT_EQ((array<len_type,3>{4, 2, 5}), m4.lengths());
    EXPECT_EQ((array<stride_type,3>{10, 5, 1}), m4.strides());
    EXPECT_EQ(v2.data(), m4.data());
}

TEST(marray_view, vary)
{
    marray<double,3> m1{4, 2, 5};
    marray<double,3> m2_{4, 2, 5};
    marray_view<double,3> m2(m2_);

    auto v1 = m1.view<DYNAMIC>();
    EXPECT_EQ((len_vector{4, 2, 5}), v1.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v1.strides());
    EXPECT_EQ(m1.data(), v1.data());

    auto v2 = m2.view<DYNAMIC>();
    EXPECT_EQ((len_vector{4, 2, 5}), v2.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v2.strides());
    EXPECT_EQ(m2.data(), v2.data());

    auto v3 = m2[0].view<DYNAMIC>();
    EXPECT_EQ((len_vector{2, 5}), v3.lengths());
    EXPECT_EQ((stride_vector{5, 1}), v3.strides());
    EXPECT_EQ(m2.data(), v3.data());

    auto v4 = m2[slice::all].view<DYNAMIC>();
    EXPECT_EQ((len_vector{4, 2, 5}), v4.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v4.strides());
    EXPECT_EQ(m2.data(), v4.data());

    auto v5 = view<DYNAMIC>(m1);
    EXPECT_EQ((len_vector{4, 2, 5}), v5.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v5.strides());
    EXPECT_EQ(m1.data(), v5.data());

    auto v6 = view<DYNAMIC>(m2);
    EXPECT_EQ((len_vector{4, 2, 5}), v6.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v6.strides());
    EXPECT_EQ(m2.data(), v6.data());

    auto v7 = view<DYNAMIC>(m2[0]);
    EXPECT_EQ((len_vector{2, 5}), v7.lengths());
    EXPECT_EQ((stride_vector{5, 1}), v7.strides());
    EXPECT_EQ(m2.data(), v7.data());

    auto v8 = view<DYNAMIC>(m2[slice::all]);
    EXPECT_EQ((len_vector{4, 2, 5}), v8.lengths());
    EXPECT_EQ((stride_vector{10, 5, 1}), v8.strides());
    EXPECT_EQ(m2.data(), v8.data());
}

TEST(marray_view, rotate)
{
    array<double,12> data = { 0, 1, 2,
                              3, 4, 5,
                              6, 7, 8,
                              9,10,11};

    marray_view<double,2> v1({4, 3}, data.data());

    rotate(v1, 1, 1);
    EXPECT_EQ((array<double,12>{ 1, 2, 0,
                                 4, 5, 3,
                                 7, 8, 6,
                                10,11, 9}), data);

    rotate(v1, 0, -1);
    EXPECT_EQ((array<double,12>{10,11, 9,
                                 1, 2, 0,
                                 4, 5, 3,
                                 7, 8, 6}), data);

    rotate(v1, {4,3});
    EXPECT_EQ((array<double,12>{10,11, 9,
                                 1, 2, 0,
                                 4, 5, 3,
                                 7, 8, 6}), data);

    rotate(v1, array<char,2>{1,1});
    EXPECT_EQ((array<double,12>{ 2, 0, 1,
                                 5, 3, 4,
                                 8, 6, 7,
                                11, 9,10}), data);
}

TEST(varray_view, rotate)
{
    array<double,12> data = { 0, 1, 2,
                              3, 4, 5,
                              6, 7, 8,
                              9,10,11};

    marray_view<double> v1({4, 3}, data.data());

    rotate(v1, 1, 1);
    EXPECT_EQ((array<double,12>{ 1, 2, 0,
                                 4, 5, 3,
                                 7, 8, 6,
                                10,11, 9}), data);

    rotate(v1, 0, -1);
    EXPECT_EQ((array<double,12>{10,11, 9,
                                 1, 2, 0,
                                 4, 5, 3,
                                 7, 8, 6}), data);

    rotate(v1, {4,3});
    EXPECT_EQ((array<double,12>{10,11, 9,
                                 1, 2, 0,
                                 4, 5, 3,
                                 7, 8, 6}), data);

    rotate(v1, vector<char>{1,1});
    EXPECT_EQ((array<double,12>{ 2, 0, 1,
                                 5, 3, 4,
                                 8, 6, 7,
                                11, 9,10}), data);
}

TEST(marray_view, front_back)
{
    double tmp;
    double* data = &tmp;

    marray_view<double,1> v1({8}, data);

    EXPECT_EQ(data, &v1.cfront());
    EXPECT_EQ(data, &v1.front());
    EXPECT_EQ(data+7, &v1.cback());
    EXPECT_EQ(data+7, &v1.back());

    marray_view<double,3> v2({4, 2, 5}, data);

    auto v3 = v2.cfront(0);
    EXPECT_EQ((array<len_type,2>{2, 5}), v3.lengths());
    EXPECT_EQ((array<stride_type,2>{5, 1}), v3.strides());
    EXPECT_EQ(data, v3.data());

    auto v4 = v2.front(1);
    EXPECT_EQ((array<len_type,2>{4, 5}), v4.lengths());
    EXPECT_EQ((array<stride_type,2>{10, 1}), v4.strides());
    EXPECT_EQ(data, v4.data());

    auto v5 = v2.cback(0);
    EXPECT_EQ((array<len_type,2>{2, 5}), v5.lengths());
    EXPECT_EQ((array<stride_type,2>{5, 1}), v5.strides());
    EXPECT_EQ(data + 30, v5.data());

    auto v6 = v2.back(1);
    EXPECT_EQ((array<len_type,2>{4, 5}), v6.lengths());
    EXPECT_EQ((array<stride_type,2>{10, 1}), v6.strides());
    EXPECT_EQ(data + 5, v6.data());
}

TEST(varray_view, front_back)
{
    double tmp;
    double* data = &tmp;

    marray_view<double> v2({4, 2, 5}, data);

    auto v3 = v2.cfront(0);
    EXPECT_EQ((len_vector{2, 5}), v3.lengths());
    EXPECT_EQ((stride_vector{5, 1}), v3.strides());
    EXPECT_EQ(data, v3.data());

    auto v4 = v2.front(1);
    EXPECT_EQ((len_vector{4, 5}), v4.lengths());
    EXPECT_EQ((stride_vector{10, 1}), v4.strides());
    EXPECT_EQ(data, v4.data());

    auto v5 = v2.cback(0);
    EXPECT_EQ((len_vector{2, 5}), v5.lengths());
    EXPECT_EQ((stride_vector{5, 1}), v5.strides());
    EXPECT_EQ(data + 30, v5.data());

    auto v6 = v2.back(1);
    EXPECT_EQ((len_vector{4, 5}), v6.lengths());
    EXPECT_EQ((stride_vector{10, 1}), v6.strides());
    EXPECT_EQ(data + 5, v6.data());
}

TEST(marray_view, access)
{
    array<double,12> data = { 0, 1, 2,
                              3, 4, 5,
                              6, 7, 8,
                              9,10,11};

    marray_view<double,2> v1({4, 3}, data.data());

    EXPECT_EQ( 0, v1(0, 0));
    EXPECT_EQ( 5, v1(1, 2));
    EXPECT_EQ(10, v1(3, 1));

    EXPECT_EQ( 0, v1[0][0]);
    EXPECT_EQ( 5, v1[1][2]);
    EXPECT_EQ(10, v1[3][1]);

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
}

TEST(varray_view, access)
{
    array<double,12> data = { 0, 1, 2,
                              3, 4, 5,
                              6, 7, 8,
                              9,10,11};

    marray_view<double> v1({4, 3}, data.data());

    EXPECT_EQ( 0, v1(0, 0));
    EXPECT_EQ( 5, v1(1, 2));
    EXPECT_EQ(10, v1(3, 1));

    auto v2 = view<DYNAMIC>(v1(slice::all, range(2)));
    EXPECT_EQ((len_vector{4, 2}), v2.lengths());
    EXPECT_EQ((stride_vector{3, 1}), v2.strides());
    EXPECT_EQ(v1.data(), v2.data());

    auto v3 = view<DYNAMIC>(v1(range(0, 4, 2), 1));
    EXPECT_EQ((len_vector{2}), v3.lengths());
    EXPECT_EQ((stride_vector{6}), v3.strides());
    EXPECT_EQ(v1.data() + 1, v3.data());
}

TEST(marray_view, iteration)
{
    array<array<int,3>,4> visited;
    array<array<double,3>,4> data = {{{ 0, 1, 2},
                                      { 3, 4, 5},
                                      { 6, 7, 8},
                                      { 9,10,11}}};

    marray_view<double,2> v1({4, 3}, &data[0][0]);
    marray_view<const double,2> v2({4, 3}, &data[0][0]);

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

TEST(varray_view, iteration)
{
    array<array<int,3>,4> visited;
    array<array<double,3>,4> data = {{{ 0, 1, 2},
                                      { 3, 4, 5},
                                      { 6, 7, 8},
                                      { 9,10,11}}};

    marray_view<double> v1({4, 3}, &data[0][0]);
    marray_view<const double> v2({4, 3}, &data[0][0]);

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

TEST(marray_view, length_stride)
{
    double tmp;
    double* data = &tmp;

    marray_view<double,3> v1_({100, 2, 6}, data);
    marray_view<double,3> v1 = v1_[range(4)][all][all];

    EXPECT_EQ(4, v1.length(0));
    EXPECT_EQ(4, v1.length(0, 2));
    EXPECT_EQ(2, v1.length(0));

    EXPECT_EQ(6, v1.length(2));
    EXPECT_EQ(6, v1.length(2, 3));
    EXPECT_EQ(3, v1.length(2));

    EXPECT_EQ(12, v1.stride(0));
    EXPECT_EQ(12, v1.stride(0, 24));
    EXPECT_EQ(24, v1.stride(0));

    EXPECT_EQ(1, v1.stride(2));
    EXPECT_EQ(1, v1.stride(2, 2));
    EXPECT_EQ(2, v1.stride(2));

    EXPECT_EQ((array<len_type,3>{2, 2, 3}), v1.lengths());
    EXPECT_EQ((array<stride_type,3>{24, 6, 2}), v1.strides());
}

TEST(varray_view, length_stride)
{
    double tmp;
    double* data = &tmp;

    marray_view<double> v1_({100, 2, 6}, data);
    marray_view<double> v1 = v1_(range(4),all,all);

    EXPECT_EQ(4, v1.length(0));
    EXPECT_EQ(4, v1.length(0, 2));
    EXPECT_EQ(2, v1.length(0));

    EXPECT_EQ(6, v1.length(2));
    EXPECT_EQ(6, v1.length(2, 3));
    EXPECT_EQ(3, v1.length(2));

    EXPECT_EQ(12, v1.stride(0));
    EXPECT_EQ(12, v1.stride(0, 24));
    EXPECT_EQ(24, v1.stride(0));

    EXPECT_EQ(1, v1.stride(2));
    EXPECT_EQ(1, v1.stride(2, 2));
    EXPECT_EQ(2, v1.stride(2));

    EXPECT_EQ((len_vector{2, 2, 3}), v1.lengths());
    EXPECT_EQ((stride_vector{24, 6, 2}), v1.strides());
}

TEST(marray_view, swap)
{
    double tmp1, tmp2;
    double* data1 = &tmp1;
    double* data2 = &tmp2;

    marray_view<double,3> v1({4, 2, 5}, data1);
    marray_view<double,3> v2({3, 8, 3}, data2);

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

TEST(varray_view, swap)
{
    double tmp1, tmp2;
    double* data1 = &tmp1;
    double* data2 = &tmp2;

    marray_view<double> v1({4, 2, 5}, data1);
    marray_view<double> v2({3, 8}, data2);

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

TEST(marray_view, print)
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

    marray_view<double,3> v1({2,2,3}, (double*)data, ROW_MAJOR);

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

TEST(varray_view, print)
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

    marray_view<double> v1({2,2,3}, (double*)data, ROW_MAJOR);

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
