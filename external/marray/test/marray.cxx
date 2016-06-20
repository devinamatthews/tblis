#include "marray.hpp"

#include "gtest/gtest.h"

#include <typeinfo>

using namespace std;
using namespace MArray;
using namespace MArray::detail;
using namespace MArray::slice;
using namespace MArray::transpose;

TEST(marray, align)
{
    EXPECT_EQ(0, align(0,8));
    EXPECT_EQ(8, align(4,8));
    EXPECT_EQ(8, align(1,8));
    EXPECT_EQ(8, align(7,8));
    EXPECT_EQ(8, align(8,8));
    EXPECT_EQ(280, align(277,8));
    EXPECT_EQ(160, align(160,8));
}

TEST(marray, aligned_allocator)
{
    constexpr size_t alignment = 1ull<<21; // 2MB
    vector<double,aligned_allocator<double,alignment>> x(100);
    EXPECT_EQ(0, uintptr_t(x.data())&(alignment-1));
}

TEST(marray, concat_types)
{
    EXPECT_TRUE((is_same<types<>,concat_types_t<types<>,types<>>>::value));
    EXPECT_TRUE((is_same<types<int>,concat_types_t<types<int>,types<>>>::value));
    EXPECT_TRUE((is_same<types<int>,concat_types_t<types<>,types<int>>>::value));
    EXPECT_TRUE((is_same<types<int,long>,concat_types_t<types<int>,types<long>>>::value));
}

TEST(marray, trailing_types)
{
    EXPECT_TRUE((is_same<types<int, void, char, bool, double, long, int*>, trailing_types_t<0, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<     void, char, bool, double, long, int*>, trailing_types_t<1, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<           char, bool, double, long, int*>, trailing_types_t<2, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                 bool, double, long, int*>, trailing_types_t<3, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                       double, long, int*>, trailing_types_t<4, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                               long, int*>, trailing_types_t<5, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                                     int*>, trailing_types_t<6, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                                         >, trailing_types_t<7, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                                         >, trailing_types_t<8, int, void, char, bool, double, long, int*>>::value));
}

TEST(marray, leading_types)
{
    EXPECT_TRUE((is_same<types<                                         >, leading_types_t<0, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<int                                      >, leading_types_t<1, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<int, void                                >, leading_types_t<2, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<int, void, char                          >, leading_types_t<3, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<int, void, char, bool                    >, leading_types_t<4, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<int, void, char, bool, double            >, leading_types_t<5, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<int, void, char, bool, double, long      >, leading_types_t<6, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<int, void, char, bool, double, long, int*>, leading_types_t<7, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<int, void, char, bool, double, long, int*>, leading_types_t<8, int, void, char, bool, double, long, int*>>::value));
}

TEST(marray, middle_types)
{
    EXPECT_TRUE((is_same<types<int, void, char, bool, double, long, int*>, middle_types_t<0, 7, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<     void, char, bool, double, long, int*>, middle_types_t<1, 7, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<           char, bool, double, long, int*>, middle_types_t<2, 7, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                 bool, double, long, int*>, middle_types_t<3, 7, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                       double, long, int*>, middle_types_t<4, 7, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                               long, int*>, middle_types_t<5, 7, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                                     int*>, middle_types_t<6, 7, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                                         >, middle_types_t<7, 7, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                                         >, middle_types_t<8, 7, int, void, char, bool, double, long, int*>>::value));

    EXPECT_TRUE((is_same<types<                                         >, middle_types_t<0, 0, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<int                                      >, middle_types_t<0, 1, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<int, void                                >, middle_types_t<0, 2, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<int, void, char                          >, middle_types_t<0, 3, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<int, void, char, bool                    >, middle_types_t<0, 4, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<int, void, char, bool, double            >, middle_types_t<0, 5, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<int, void, char, bool, double, long      >, middle_types_t<0, 6, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<int, void, char, bool, double, long, int*>, middle_types_t<0, 7, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<int, void, char, bool, double, long, int*>, middle_types_t<0, 8, int, void, char, bool, double, long, int*>>::value));

    EXPECT_TRUE((is_same<types<int                                      >, middle_types_t<0, 1, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<     void                                >, middle_types_t<1, 2, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<           char                          >, middle_types_t<2, 3, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                 bool                    >, middle_types_t<3, 4, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                       double            >, middle_types_t<4, 5, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                               long      >, middle_types_t<5, 6, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                                     int*>, middle_types_t<6, 7, int, void, char, bool, double, long, int*>>::value));

    EXPECT_TRUE((is_same<types<int, void                                >, middle_types_t<0, 2, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<     void, char                          >, middle_types_t<1, 3, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<           char, bool                    >, middle_types_t<2, 4, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                 bool, double            >, middle_types_t<3, 5, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                       double, long      >, middle_types_t<4, 6, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                               long, int*>, middle_types_t<5, 7, int, void, char, bool, double, long, int*>>::value));

    EXPECT_TRUE((is_same<types<int, void, char                          >, middle_types_t<0, 3, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<     void, char, bool                    >, middle_types_t<1, 4, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<           char, bool, double            >, middle_types_t<2, 5, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                 bool, double, long      >, middle_types_t<3, 6, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                       double, long, int*>, middle_types_t<4, 7, int, void, char, bool, double, long, int*>>::value));

    EXPECT_TRUE((is_same<types<int, void, char, bool                    >, middle_types_t<0, 4, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<     void, char, bool, double            >, middle_types_t<1, 5, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<           char, bool, double, long      >, middle_types_t<2, 6, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<                 bool, double, long, int*>, middle_types_t<3, 7, int, void, char, bool, double, long, int*>>::value));

    EXPECT_TRUE((is_same<types<int, void, char, bool, double            >, middle_types_t<0, 5, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<     void, char, bool, double, long      >, middle_types_t<1, 6, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<           char, bool, double, long, int*>, middle_types_t<2, 7, int, void, char, bool, double, long, int*>>::value));

    EXPECT_TRUE((is_same<types<int, void, char, bool, double, long      >, middle_types_t<0, 6, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<     void, char, bool, double, long, int*>, middle_types_t<1, 7, int, void, char, bool, double, long, int*>>::value));

    EXPECT_TRUE((is_same<types<int, void, char, bool, double, long, int*>, middle_types_t<0, 7, int, void, char, bool, double, long, int*>>::value));
}

TEST(marray, nth_type)
{
    EXPECT_TRUE((is_same<types<   int>, nth_type_t<0, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<  void>, nth_type_t<1, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<  char>, nth_type_t<2, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<  bool>, nth_type_t<3, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<double>, nth_type_t<4, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<  long>, nth_type_t<5, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<  int*>, nth_type_t<6, int, void, char, bool, double, long, int*>>::value));
    EXPECT_TRUE((is_same<types<      >, nth_type_t<7, int, void, char, bool, double, long, int*>>::value));
}

TEST(marray, apply_some_args)
{
    int tmp;

    EXPECT_TRUE((apply_some_args::leading<types<int, int>>
                                ::middle<types<long, int*>>
                                ::trailing<types<double>>()
                     ([&](long a, int* b) { return a == 4 && b == &tmp; },
                      0, 1, 4, &tmp, -1.6)));
    EXPECT_TRUE((apply_some_args::leading<types<>>
                                ::middle<types<long, int*>>
                                ::trailing<types<double>>()
                     ([&](long a, int* b) { return a == 4 && b == &tmp; },
                      4, &tmp, -1.6)));
    EXPECT_TRUE((apply_some_args::leading<types<int, int>>
                                ::middle<types<long, int*>>
                                ::trailing<types<>>()
                     ([&](long a, int* b) { return a == 4 && b == &tmp; },
                      0, 1, 4, &tmp)));
    EXPECT_TRUE((apply_some_args::leading<types<>>
                                ::middle<types<long, int*>>
                                ::trailing<types<>>()
                     ([&](long a, int* b) { return a == 4 && b == &tmp; },
                      4, &tmp)));
    EXPECT_TRUE((apply_some_args::leading<types<int, int>>
                                ::middle<types<>>
                                ::trailing<types<double>>()
                     ([]() { return true; },
                      0, 1, -1.6)));
    EXPECT_TRUE((apply_some_args::leading<types<>>
                                ::middle<types<>>
                                ::trailing<types<double>>()
                     ([]() { return true; },
                      -1.6)));
    EXPECT_TRUE((apply_some_args::leading<types<int, int>>
                                ::middle<types<>>
                                ::trailing<types<>>()
                     ([]() { return true; },
                      0, 1)));
    EXPECT_TRUE((apply_some_args::leading<types<>>
                                ::middle<types<>>
                                ::trailing<types<>>()
                     ([]() { return true; })));
}

TEST(marray, apply_trailing_args)
{
    int tmp;

    EXPECT_TRUE((apply_trailing_args<2>()
                     ([&](long a, int* b) { return a == 4 && b == &tmp; },
                      0, 1, 4, &tmp)));
    EXPECT_TRUE((apply_trailing_args<0>()
                     ([&](long a, int* b) { return a == 4 && b == &tmp; },
                      4, &tmp)));
    EXPECT_TRUE((apply_trailing_args<2>()
                     ([]() { return true; },
                      0, 1)));
    EXPECT_TRUE((apply_trailing_args<0>()
                     ([]() { return true; })));
}

TEST(marray, apply_leading_args)
{
    int tmp;

    EXPECT_TRUE((apply_leading_args<2>()
                     ([&](long a, int* b) { return a == 4 && b == &tmp; },
                      4, &tmp, -1.6)));
    EXPECT_TRUE((apply_leading_args<2>()
                     ([&](long a, int* b) { return a == 4 && b == &tmp; },
                      4, &tmp)));
    EXPECT_TRUE((apply_leading_args<0>()
                     ([]() { return true; },
                      -1.6)));
    EXPECT_TRUE((apply_leading_args<0>()
                     ([]() { return true; })));
}

TEST(marray, apply_middle_args)
{
    int tmp;

    EXPECT_TRUE((apply_middle_args<2,4>()
                     ([&](long a, int* b) { return a == 4 && b == &tmp; },
                      0, 1, 4, &tmp, -1.6)));
    EXPECT_TRUE((apply_middle_args<0,2>()
                     ([&](long a, int* b) { return a == 4 && b == &tmp; },
                      4, &tmp, -1.6)));
    EXPECT_TRUE((apply_middle_args<2,4>()
                     ([&](long a, int* b) { return a == 4 && b == &tmp; },
                      0, 1, 4, &tmp)));
    EXPECT_TRUE((apply_middle_args<0,2>()
                     ([&](long a, int* b) { return a == 4 && b == &tmp; },
                      4, &tmp)));
    EXPECT_TRUE((apply_middle_args<2,2>()
                     ([]() { return true; },
                      0, 1, -1.6)));
    EXPECT_TRUE((apply_middle_args<0,0>()
                     ([]() { return true; },
                      -1.6)));
    EXPECT_TRUE((apply_middle_args<2,2>()
                     ([]() { return true; },
                      0, 1)));
    EXPECT_TRUE((apply_middle_args<0,0>()
                     ([]() { return true; })));
}

TEST(marray, apply_nth_arg)
{

    EXPECT_TRUE((apply_nth_arg<2>()
                     ([&](long a) { return a == 4; },
                      0, 1, 4, -1.6)));
    EXPECT_TRUE((apply_nth_arg<0>()
                     ([&](long a) { return a == 4; },
                      4, -1.6)));
    EXPECT_TRUE((apply_nth_arg<2>()
                     ([&](long a) { return a == 4; },
                      0, 1, 4)));
    EXPECT_TRUE((apply_nth_arg<0>()
                     ([&](long a) { return a == 4; },
                      4)));
}

TEST(marray, make_array_from_types)
{
    auto a = make_array_from_types<int, types<int, long, double, bool>>::apply(3, 1, 2, 0);

    EXPECT_TRUE((is_same<array<int,4>,decltype(a)>::value));
    EXPECT_EQ(3, a[0]);
    EXPECT_EQ(1, a[1]);
    EXPECT_EQ(2, a[2]);
    EXPECT_EQ(0, a[3]);
}

TEST(marray, make_vector_from_types)
{
    auto a = make_vector_from_types<int, types<int, long, double, bool>>::apply(3, 1, 2, 0);

    EXPECT_TRUE((is_same<vector<int>,decltype(a)>::value));
    EXPECT_EQ(4, a.size());
    EXPECT_EQ(3, a[0]);
    EXPECT_EQ(1, a[1]);
    EXPECT_EQ(2, a[2]);
    EXPECT_EQ(0, a[3]);
}

TEST(marray, are_convertible)
{
    EXPECT_TRUE((are_convertible<int>::value));
    EXPECT_TRUE((are_convertible<int,int>::value));
    EXPECT_TRUE((are_convertible<int,bool,int>::value));
    EXPECT_TRUE((are_convertible<int,bool,long,int>::value));
    EXPECT_FALSE((are_convertible<int,void>::value));
    EXPECT_FALSE((are_convertible<int,int,vector<double>>::value));
    EXPECT_FALSE((are_convertible<int,vector<double>,int>::value));
    EXPECT_FALSE((are_convertible<int,vector<double>,void>::value));
}

TEST(marray, is_index_or_slice)
{
    EXPECT_TRUE((is_index_or_slice<int>::value));
    EXPECT_TRUE((is_index_or_slice<long>::value));
    EXPECT_TRUE((is_index_or_slice<range_t<int>>::value));
    EXPECT_TRUE((is_index_or_slice<range_t<long>>::value));
    EXPECT_TRUE((is_index_or_slice<all_t>::value));
    EXPECT_FALSE((is_index_or_slice<void>::value));
    EXPECT_FALSE((is_index_or_slice<vector<double>>::value));
}

TEST(marray, are_indices_or_slices)
{
    EXPECT_TRUE((are_indices_or_slices<>::value));
    EXPECT_TRUE((are_indices_or_slices<int>::value));
    EXPECT_TRUE((are_indices_or_slices<range_t<int>>::value));
    EXPECT_TRUE((are_indices_or_slices<all_t>::value));
    EXPECT_TRUE((are_indices_or_slices<bool,int>::value));
    EXPECT_TRUE((are_indices_or_slices<bool,long,int>::value));
    EXPECT_TRUE((are_indices_or_slices<bool,all_t,int>::value));
    EXPECT_TRUE((are_indices_or_slices<bool,range_t<int>,int>::value));
    EXPECT_TRUE((are_indices_or_slices<all_t,range_t<int>,int>::value));
    EXPECT_TRUE((are_indices_or_slices<all_t,range_t<int>,range_t<long>>::value));
    EXPECT_FALSE((are_indices_or_slices<void>::value));
    EXPECT_FALSE((are_indices_or_slices<int,vector<double>>::value));
    EXPECT_FALSE((are_indices_or_slices<all_t,vector<double>>::value));
    EXPECT_FALSE((are_indices_or_slices<vector<double>,int>::value));
    EXPECT_FALSE((are_indices_or_slices<vector<double>,all_t>::value));
    EXPECT_FALSE((are_indices_or_slices<vector<double>,void>::value));
}

TEST(marray, marray_ref_assign_to_ref)
{
    marray<double,3> a(2, 3, 2);
    marray<double,3> b(4, 3, 2);
    marray<double,4> c(4, 4, 3, 2);

    for (int i = 0;i < 4*3*2;i++)
    {
        b.data()[i] = i;
    }

    for (int i = 0;i < 4*4*3*2;i++)
    {
        c.data()[i] = i;
    }

    a[0] = b[1];

    for (int i = 0;i < 3*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+3*2, a.data()[i]);
        EXPECT_EQ(0, a.data()[i+3*2]);
    }

    a[0] = c[1][1];

    for (int i = 0;i < 3*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+3*2+4*3*2, a.data()[i]);
        EXPECT_EQ(0, a.data()[i+3*2]);
    }
}

TEST(marray, marray_ref_assign_to_slice)
{
    marray<double,3> a(2, 3, 2);
    marray<double,3> b(4, 4, 2);
    marray<double,4> c(4, 4, 4, 2);

    for (int i = 0;i < 4*4*2;i++)
    {
        b.data()[i] = i;
    }

    for (int i = 0;i < 4*4*4*2;i++)
    {
        c.data()[i] = i;
    }

    a[0] = b[1][range(3)];

    for (int i = 0;i < 3*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+4*2, a.data()[i]);
        EXPECT_EQ(0, a.data()[i+3*2]);
    }

    a[0] = c[1][1][range(3)];

    for (int i = 0;i < 3*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+4*2+4*4*2, a.data()[i]);
        EXPECT_EQ(0, a.data()[i+3*2]);
    }
}

TEST(marray, const_marray_ref_data)
{
    const marray<double,3> a(2, 3, 2);
    EXPECT_EQ(a.data()+1*3*2+2*2, a[1][2].data());
    EXPECT_TRUE((std::is_same<const double*,decltype(a[1][2].data())>::value));
}

TEST(marray, marray_ref_data)
{
    marray<double,3> a(2, 3, 2);
    EXPECT_EQ(a.data()+1*3*2+2*2, a[1][2].data());
    EXPECT_TRUE((std::is_same<double*,decltype(a[1][2].data())>::value));
}

TEST(marray, const_marray_ref_view)
{
    marray<double,3> a_(2, 3, 2);
    const marray<double,3>& a = a_;

    for (int i = 0;i < 2*3*2;i++)
    {
        a_.data()[i] = i;
    }

    const_marray_view<double,1> b = a[1][2];

    for (int i = 0;i < 2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+1*3*2+2*2, b[i]);
    }

    auto c = view(a[1][2]);

    for (int i = 0;i < 2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+1*3*2+2*2, c[i]);
    }

    EXPECT_TRUE((is_same<decltype(b),decltype(c)>::value));

    EXPECT_EQ(1*3*2+2*2+1, a[1][2][1]);
    EXPECT_EQ(1*3*2+2*2+1, a[1][2](1));
    EXPECT_EQ(1*3*2+2*2+1, a[1](2,1));
}

TEST(marray, marray_ref_view)
{
    marray<double,3> a(2, 3, 2);

    for (int i = 0;i < 2*3*2;i++)
    {
        a.data()[i] = i;
    }

    marray_view<double,1> b = a[1][2];

    for (int i = 0;i < 2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+1*3*2+2*2, b[i]);
    }

    auto c = view(a[1][2]);

    for (int i = 0;i < 2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+1*3*2+2*2, c[i]);
    }

    EXPECT_TRUE((is_same<decltype(b),decltype(c)>::value));

    EXPECT_EQ(1*3*2+2*2+1, a[1][2][1]);
    EXPECT_EQ(1*3*2+2*2+1, a[1][2](1));
    EXPECT_EQ(1*3*2+2*2+1, a[1](2,1));
}

TEST(marray, marray_slice_assign_to_ref)
{
    marray<double,3> a(2, 4, 2);
    marray<double,3> b(4, 3, 2);
    marray<double,4> c(4, 4, 3, 2);

    for (int i = 0;i < 4*3*2;i++)
    {
        b.data()[i] = i;
    }

    for (int i = 0;i < 4*4*3*2;i++)
    {
        c.data()[i] = i;
    }

    a[0][range(3)] = b[1];

    for (int i = 0;i < 3*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+3*2, a.data()[i]);
        EXPECT_EQ(0, a.data()[i+4*2]);
    }

    a[0][range(3)] = c[1][1];

    for (int i = 0;i < 3*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+3*2+4*3*2, a.data()[i]);
        EXPECT_EQ(0, a.data()[i+4*2]);
    }
}

TEST(marray, marray_slice_assign_to_slice)
{
    marray<double,3> a(2, 4, 2);
    marray<double,3> b(4, 4, 2);
    marray<double,4> c(4, 4, 4, 2);

    for (int i = 0;i < 4*4*2;i++)
    {
        b.data()[i] = i;
    }

    for (int i = 0;i < 4*4*4*2;i++)
    {
        c.data()[i] = i;
    }

    a[0][range(3)] = b[1][range(3)];

    for (int i = 0;i < 3*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+4*2, a.data()[i]);
        EXPECT_EQ(0, a.data()[i+4*2]);
    }

    a[0][range(3)] = c[1][1][range(3)];

    for (int i = 0;i < 3*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+4*2+4*4*2, a.data()[i]);
        EXPECT_EQ(0, a.data()[i+4*2]);
    }
}

TEST(marray, const_marray_slice_data)
{
    const marray<double,3> a(2, 3, 2);
    EXPECT_EQ(a.data()+1*3*2+1*2, a[1][range(1,3)].data());
    EXPECT_TRUE((std::is_same<const double*,decltype(a[1][range(1,3)].data())>::value));
}

TEST(marray, marray_slice_data)
{
    marray<double,3> a(2, 3, 2);
    EXPECT_EQ(a.data()+1*3*2+1*2, a[1][range(1,3)].data());
    EXPECT_TRUE((std::is_same<double*,decltype(a[1][range(1,3)].data())>::value));
}

TEST(marray, const_marray_slice_view)
{
    marray<double,3> a_(2, 3, 2);
    const marray<double,3>& a = a_;

    for (int i = 0;i < 2*3*2;i++)
    {
        a_.data()[i] = i;
    }

    const_marray_view<double,2> b = a[1][range(1,3)];

    for (int i = 0;i < 2*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+1*3*2+1*2, b.data()[i]);
    }

    auto c = view(a[1][range(1,3)]);

    for (int i = 0;i < 2*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+1*3*2+1*2, c.data()[i]);
    }


    auto d = a[1][range(1,3)](all);

    for (int i = 0;i < 2*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+1*3*2+1*2, d.data()[i]);
    }

    EXPECT_TRUE((is_same<decltype(b),decltype(c)>::value));
    EXPECT_TRUE((is_same<decltype(b),decltype(d)>::value));

    const_marray_view<double,1> e = a[1](range(1,3),1);

    for (int i = 0;i < 2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(1*3*2+(i+1)*2+1, e.data()[i*2]);
    }

    const_marray_view<double,1> f = a[range(0,2)](1,1);

    for (int i = 0;i < 2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i*3*2+1*2+1, f.data()[i*3*2]);
    }

    const_marray_view<double,2> g = a[range(0,2)](range(1,3),1);

    for (int i = 0;i < 2;i++)
    {
        SCOPED_TRACE(i);
        for (int j = 0;j < 2;j++)
        {
            SCOPED_TRACE(j);
            EXPECT_EQ(i*3*2+(j+1)*2+1, g.data()[i*3*2+j*2]);
        }
    }
}

TEST(marray, marray_slice_view)
{
    marray<double,3> a(2, 3, 2);

    for (int i = 0;i < 2*3*2;i++)
    {
        a.data()[i] = i;
    }

    marray_view<double,2> b = a[1][range(1,3)];

    for (int i = 0;i < 2*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+1*3*2+1*2, b.data()[i]);
    }

    auto c = view(a[1][range(1,3)]);

    for (int i = 0;i < 2*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+1*3*2+1*2, c.data()[i]);
    }

    auto d = a[1][range(1,3)](all);

    for (int i = 0;i < 2*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i+1*3*2+1*2, d.data()[i]);
    }

    EXPECT_TRUE((is_same<decltype(b),decltype(c)>::value));
    EXPECT_TRUE((is_same<decltype(b),decltype(d)>::value));

    marray_view<double,1> e = a[1](range(1,3),1);

    for (int i = 0;i < 2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(1*3*2+(i+1)*2+1, e.data()[i*2]);
    }

    marray_view<double,1> f = a[range(0,2)](1,1);

    for (int i = 0;i < 2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i*3*2+1*2+1, f.data()[i*3*2]);
    }

    marray_view<double,2> g = a[range(0,2)](range(1,3),1);

    for (int i = 0;i < 2;i++)
    {
        SCOPED_TRACE(i);
        for (int j = 0;j < 2;j++)
        {
            SCOPED_TRACE(j);
            EXPECT_EQ(i*3*2+(j+1)*2+1, g.data()[i*3*2+j*2]);
        }
    }
}

TEST(marray, default_strides)
{
    marray<double,1> a;
    marray<double,4> b;

    EXPECT_EQ(make_array(1l), a.default_strides(make_array(4)));
    EXPECT_EQ(make_array(1l), a.default_strides(make_array(4), ROW_MAJOR));
    EXPECT_EQ(make_array(1l), a.default_strides(make_array(4), COLUMN_MAJOR));

    EXPECT_EQ(make_array(60l, 20l,  4l,  1l), b.default_strides(make_array(5, 3, 5, 4)));
    EXPECT_EQ(make_array(60l, 20l,  4l,  1l), b.default_strides(make_array(5, 3, 5, 4), ROW_MAJOR));
    EXPECT_EQ(make_array( 1l,  5l, 15l, 75l), b.default_strides(make_array(5, 3, 5, 4), COLUMN_MAJOR));

    EXPECT_EQ(make_array(1l), a.default_strides(4));
    EXPECT_EQ(make_array(1l), a.default_strides(4, ROW_MAJOR));
    EXPECT_EQ(make_array(1l), a.default_strides(4, COLUMN_MAJOR));

    EXPECT_EQ(make_array(60l, 20l,  4l,  1l), b.default_strides(5, 3, 5, 4));
    EXPECT_EQ(make_array(60l, 20l,  4l,  1l), b.default_strides(5, 3, 5, 4, ROW_MAJOR));
    EXPECT_EQ(make_array( 1l,  5l, 15l, 75l), b.default_strides(5, 3, 5, 4, COLUMN_MAJOR));
}

TEST(marray, const_marray_view_construct_empty)
{
    const_marray_view<double,1> a;
    const_marray_view<double,4> b;

    EXPECT_EQ(make_array(0u), a.lengths());
    EXPECT_EQ(make_array(0l), a.strides());
    EXPECT_EQ(nullptr, a.data());

    EXPECT_EQ(make_array(0u, 0u, 0u, 0u), b.lengths());
    EXPECT_EQ(make_array(0l, 0l, 0l, 0l), b.strides());
    EXPECT_EQ(nullptr, b.data());
}

TEST(marray, const_marray_view_construct_copy)
{
    double tmp;
    marray<double,1> a(4);
    marray<double,4> b(2, 5, 4, 2);
    marray_view<double,1> av(4, &tmp, 3);
    marray_view<double,4> bv(2, 5, 4, 2, &tmp, 1, 5, 20, 45);
    const_marray_view<double,1> acv(4, &tmp, 3);
    const_marray_view<double,4> bcv(2, 5, 4, 2, &tmp, 1, 5, 20, 45);

    const_marray_view<double,1> c(a);
    const_marray_view<double,4> d(b);

    EXPECT_EQ(make_array(4u), c.lengths());
    EXPECT_EQ(make_array(1l), c.strides());
    EXPECT_EQ(a.data(), c.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), d.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), d.strides());
    EXPECT_EQ(b.data(), d.data());

    const_marray_view<double,1> e(av);
    const_marray_view<double,4> f(bv);

    EXPECT_EQ(make_array(4u), e.lengths());
    EXPECT_EQ(make_array(3l), e.strides());
    EXPECT_EQ(&tmp, e.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), f.lengths());
    EXPECT_EQ(make_array(1l, 5l, 20l, 45l), f.strides());
    EXPECT_EQ(&tmp, f.data());

    const_marray_view<double,1> g(acv);
    const_marray_view<double,4> h(bcv);

    EXPECT_EQ(make_array(4u), g.lengths());
    EXPECT_EQ(make_array(3l), g.strides());
    EXPECT_EQ(&tmp, g.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), h.lengths());
    EXPECT_EQ(make_array(1l, 5l, 20l, 45l), h.strides());
    EXPECT_EQ(&tmp, h.data());
}

TEST(marray, const_marray_view_construct_array)
{
    double tmp;
    const_marray_view<double,1> a(make_array(4), &tmp, ROW_MAJOR);
    const_marray_view<double,4> b(make_array(2, 5, 4, 2), &tmp, ROW_MAJOR);
    const_marray_view<double,1> c(make_array(4), &tmp, COLUMN_MAJOR);
    const_marray_view<double,4> d(make_array(2, 5, 4, 2), &tmp, COLUMN_MAJOR);
    const_marray_view<double,1> e(make_array(4), &tmp, make_array(3));
    const_marray_view<double,4> f(make_array(2, 5, 4, 2), &tmp, make_array(1, 5, 20, 45));

    EXPECT_EQ(make_array(4u), a.lengths());
    EXPECT_EQ(make_array(1l), a.strides());
    EXPECT_EQ(&tmp, a.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b.strides());
    EXPECT_EQ(&tmp, b.data());

    EXPECT_EQ(make_array(4u), c.lengths());
    EXPECT_EQ(make_array(1l), c.strides());
    EXPECT_EQ(&tmp, c.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), d.lengths());
    EXPECT_EQ(make_array(1l, 2l, 10l, 40l), d.strides());
    EXPECT_EQ(&tmp, d.data());

    EXPECT_EQ(make_array(4u), e.lengths());
    EXPECT_EQ(make_array(3l), e.strides());
    EXPECT_EQ(&tmp, e.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), f.lengths());
    EXPECT_EQ(make_array(1l, 5l, 20l, 45l), f.strides());
    EXPECT_EQ(&tmp, f.data());
}

TEST(marray, const_marray_view_construct_direct)
{
    double tmp;
    const_marray_view<double,1> a(4, &tmp, ROW_MAJOR);
    const_marray_view<double,4> b(2, 5, 4, 2, &tmp, ROW_MAJOR);
    const_marray_view<double,1> c(4, &tmp, COLUMN_MAJOR);
    const_marray_view<double,4> d(2, 5, 4, 2, &tmp, COLUMN_MAJOR);
    const_marray_view<double,1> e(4, &tmp, 3);
    const_marray_view<double,4> f(2, 5, 4, 2, &tmp, 1, 5, 20, 45);

    EXPECT_EQ(make_array(4u), a.lengths());
    EXPECT_EQ(make_array(1l), a.strides());
    EXPECT_EQ(&tmp, a.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b.strides());
    EXPECT_EQ(&tmp, b.data());

    EXPECT_EQ(make_array(4u), c.lengths());
    EXPECT_EQ(make_array(1l), c.strides());
    EXPECT_EQ(&tmp, c.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), d.lengths());
    EXPECT_EQ(make_array(1l, 2l, 10l, 40l), d.strides());
    EXPECT_EQ(&tmp, d.data());

    EXPECT_EQ(make_array(4u), e.lengths());
    EXPECT_EQ(make_array(3l), e.strides());
    EXPECT_EQ(&tmp, e.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), f.lengths());
    EXPECT_EQ(make_array(1l, 5l, 20l, 45l), f.strides());
    EXPECT_EQ(&tmp, f.data());
}

TEST(marray, const_marray_view_permute)
{
    double tmp;
    const_marray_view<double,1> a(4, &tmp, ROW_MAJOR);
    const_marray_view<double,4> b(2, 5, 4, 3, &tmp, ROW_MAJOR);

    auto ap1 = a.permute(make_array(0));
    auto ap2 = a.permute(0);

    auto bp1 = b.permute(make_array(2, 0, 3, 1));
    auto bp2 = b.permute(2, 0, 3, 1);

    EXPECT_TRUE((is_same<const_marray_view<double,1>,decltype(ap1)>::value));
    EXPECT_TRUE((is_same<const_marray_view<double,1>,decltype(ap2)>::value));
    EXPECT_TRUE((is_same<const_marray_view<double,4>,decltype(bp1)>::value));
    EXPECT_TRUE((is_same<const_marray_view<double,4>,decltype(bp2)>::value));

    EXPECT_EQ(make_array(4u), ap1.lengths());
    EXPECT_EQ(make_array(1l), ap1.strides());
    EXPECT_EQ(&tmp, ap1.data());

    EXPECT_EQ(make_array(4u), ap2.lengths());
    EXPECT_EQ(make_array(1l), ap2.strides());
    EXPECT_EQ(&tmp, ap2.data());

    EXPECT_EQ(make_array(4u, 2u, 3u, 5u), bp1.lengths());
    EXPECT_EQ(make_array(3l, 60l, 1l, 12l), bp1.strides());
    EXPECT_EQ(&tmp, bp1.data());

    EXPECT_EQ(make_array(4u, 2u, 3u, 5u), bp2.lengths());
    EXPECT_EQ(make_array(3l, 60l, 1l, 12l), bp2.strides());
    EXPECT_EQ(&tmp, bp2.data());
}

TEST(marray, const_marray_view_front)
{
    double tmp;
    const_marray_view<double,1> a(4, &tmp, ROW_MAJOR);
    const_marray_view<double,4> b(2, 5, 4, 3, &tmp, ROW_MAJOR);

    auto& af1 = a.front();
    auto& af2 = a.front(0);

    auto bf1 = b.front(0);
    auto bf2 = b.front(2);

    EXPECT_TRUE((is_same<const double&,decltype(af1)>::value));
    EXPECT_TRUE((is_same<const double&,decltype(af2)>::value));
    EXPECT_TRUE((is_same<const_marray_view<double,3>,decltype(bf1)>::value));
    EXPECT_TRUE((is_same<const_marray_view<double,3>,decltype(bf2)>::value));

    EXPECT_EQ(&tmp, &af1);
    EXPECT_EQ(&tmp, &af2);

    EXPECT_EQ(make_array(5u, 4u, 3u), bf1.lengths());
    EXPECT_EQ(make_array(12l, 3l, 1l), bf1.strides());
    EXPECT_EQ(&tmp, bf1.data());

    EXPECT_EQ(make_array(2u, 5u, 3u), bf2.lengths());
    EXPECT_EQ(make_array(60l, 12l, 1l), bf2.strides());
    EXPECT_EQ(&tmp, bf2.data());
}

TEST(marray, const_marray_view_back)
{
    double tmp;
    const_marray_view<double,1> a(4, &tmp, ROW_MAJOR);
    const_marray_view<double,4> b(2, 5, 4, 3, &tmp, ROW_MAJOR);

    auto& ab1 = a.back();
    auto& ab2 = a.back(0);

    auto bb1 = b.back(0);
    auto bb2 = b.back(2);

    EXPECT_TRUE((is_same<const double&,decltype(ab1)>::value));
    EXPECT_TRUE((is_same<const double&,decltype(ab2)>::value));
    EXPECT_TRUE((is_same<const_marray_view<double,3>,decltype(bb1)>::value));
    EXPECT_TRUE((is_same<const_marray_view<double,3>,decltype(bb2)>::value));

    EXPECT_EQ(&tmp+3, &ab1);
    EXPECT_EQ(&tmp+3, &ab2);

    EXPECT_EQ(make_array(5u, 4u, 3u), bb1.lengths());
    EXPECT_EQ(make_array(12l, 3l, 1l), bb1.strides());
    EXPECT_EQ(&tmp+60*1, bb1.data());

    EXPECT_EQ(make_array(2u, 5u, 3u), bb2.lengths());
    EXPECT_EQ(make_array(60l, 12l, 1l), bb2.strides());
    EXPECT_EQ(&tmp+3*3, bb2.data());
}

TEST(marray, const_marray_brackets)
{
    double tmp;
    const_marray_view<double,1> a(4, &tmp, ROW_MAJOR);
    const_marray_view<double,4> b(2, 5, 4, 3, &tmp, ROW_MAJOR);

    const double& a1 = a[0];
    const double& a2 = a[3];

    const_marray_view<double,3> b1 = b[0];
    const_marray_view<double,3> b2 = b[1];

    EXPECT_EQ(&tmp, &a1);
    EXPECT_EQ(&tmp+3, &a2);

    EXPECT_EQ(make_array(5u, 4u, 3u), b1.lengths());
    EXPECT_EQ(make_array(12l, 3l, 1l), b1.strides());
    EXPECT_EQ(&tmp, b1.data());

    EXPECT_EQ(make_array(5u, 4u, 3u), b2.lengths());
    EXPECT_EQ(make_array(12l, 3l, 1l), b2.strides());
    EXPECT_EQ(&tmp+60*1, b2.data());

    const_marray_view<double,1> a3 = a[range(2,4)];
    const_marray_view<double,1> a4 = a[all];

    const_marray_view<double,4> b3 = b[range(1,2)];
    const_marray_view<double,4> b4 = b[all];

    EXPECT_EQ(make_array(2u), a3.lengths());
    EXPECT_EQ(make_array(1l), a3.strides());
    EXPECT_EQ(&tmp+2, a3.data());

    EXPECT_EQ(make_array(4u), a4.lengths());
    EXPECT_EQ(make_array(1l), a4.strides());
    EXPECT_EQ(&tmp, a4.data());

    EXPECT_EQ(make_array(1u, 5u, 4u, 3u), b3.lengths());
    EXPECT_EQ(make_array(60l, 12l, 3l, 1l), b3.strides());
    EXPECT_EQ(&tmp+60*1, b3.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 3u), b4.lengths());
    EXPECT_EQ(make_array(60l, 12l, 3l, 1l), b4.strides());
    EXPECT_EQ(&tmp, b4.data());
}

TEST(marray, const_marray_parentheses)
{
    double tmp;
    const_marray_view<double,1> a(4, &tmp, ROW_MAJOR);
    const_marray_view<double,4> b(2, 5, 4, 3, &tmp, ROW_MAJOR);

    const double& a1 = a(0);
    const double& a2 = a(3);

    const double& b1 = b(0, 0, 0, 0);
    const double& b2 = b(1, 4, 1, 2);

    EXPECT_EQ(&tmp, &a1);
    EXPECT_EQ(&tmp+3, &a2);

    EXPECT_EQ(&tmp, &b1);
    EXPECT_EQ(&tmp+1*60+4*12+1*3+2, &b2);

    const_marray_view<double,1> a3 = a(range(2,4));
    const_marray_view<double,1> a4 = a(all);

    const_marray_view<double,1> b3 = b(1,range(1,2),0,2);
    const_marray_view<double,2> b4 = b(all,1,range(3),1);
    const_marray_view<double,4> b5 = b(all,range(3,4),range(3),all);

    EXPECT_EQ(make_array(2u), a3.lengths());
    EXPECT_EQ(make_array(1l), a3.strides());
    EXPECT_EQ(&tmp+2, a3.data());

    EXPECT_EQ(make_array(4u), a4.lengths());
    EXPECT_EQ(make_array(1l), a4.strides());
    EXPECT_EQ(&tmp, a4.data());

    EXPECT_EQ(make_array(1u), b3.lengths());
    EXPECT_EQ(make_array(12l), b3.strides());
    EXPECT_EQ(&tmp+1*60+1*12+2, b3.data());

    EXPECT_EQ(make_array(2u, 3u), b4.lengths());
    EXPECT_EQ(make_array(60l, 3l), b4.strides());
    EXPECT_EQ(&tmp+1*12+1, b4.data());

    EXPECT_EQ(make_array(2u, 1u, 3u, 3u), b5.lengths());
    EXPECT_EQ(make_array(60l, 12l, 3l, 1l), b5.strides());
    EXPECT_EQ(&tmp+3*12, b5.data());
}

TEST(marray, const_marray_view_accessors)
{
    double tmp;
    const_marray_view<double,1> a(4, &tmp, ROW_MAJOR);
    const_marray_view<double,4> b(2, 5, 4, 2, &tmp, ROW_MAJOR);

    EXPECT_EQ(make_array(4u), a.lengths());
    EXPECT_EQ(make_array(1l), a.strides());
    EXPECT_EQ(4, a.length());
    EXPECT_EQ(1, a.stride());
    EXPECT_EQ(4, a.length(0));
    EXPECT_EQ(1, a.stride(0));
    EXPECT_EQ(&tmp, a.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b.strides());
    EXPECT_EQ(2, b.length(0));
    EXPECT_EQ(5, b.length(1));
    EXPECT_EQ(4, b.length(2));
    EXPECT_EQ(2, b.length(3));
    EXPECT_EQ(40, b.stride(0));
    EXPECT_EQ(8, b.stride(1));
    EXPECT_EQ(2, b.stride(2));
    EXPECT_EQ(1, b.stride(3));
    EXPECT_EQ(&tmp, b.data());
}

TEST(marray, marray_view_construct_empty)
{
    marray_view<double,1> a;
    marray_view<double,4> b;

    EXPECT_EQ(make_array(0u), a.lengths());
    EXPECT_EQ(make_array(0l), a.strides());
    EXPECT_EQ(nullptr, a.data());

    EXPECT_EQ(make_array(0u, 0u, 0u, 0u), b.lengths());
    EXPECT_EQ(make_array(0l, 0l, 0l, 0l), b.strides());
    EXPECT_EQ(nullptr, b.data());
}

TEST(marray, marray_view_construct_copy)
{
    double tmp;
    marray<double,1> a(4);
    marray<double,4> b(2, 5, 4, 2);
    marray_view<double,1> av(4, &tmp, 3);
    marray_view<double,4> bv(2, 5, 4, 2, &tmp, 1, 5, 20, 45);

    marray_view<double,1> c(a);
    marray_view<double,4> d(b);

    EXPECT_EQ(make_array(4u), c.lengths());
    EXPECT_EQ(make_array(1l), c.strides());
    EXPECT_EQ(a.data(), c.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), d.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), d.strides());
    EXPECT_EQ(b.data(), d.data());

    marray_view<double,1> e(av);
    marray_view<double,4> f(bv);

    EXPECT_EQ(make_array(4u), e.lengths());
    EXPECT_EQ(make_array(3l), e.strides());
    EXPECT_EQ(&tmp, e.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), f.lengths());
    EXPECT_EQ(make_array(1l, 5l, 20l, 45l), f.strides());
    EXPECT_EQ(&tmp, f.data());
}

TEST(marray, marray_view_construct_array)
{
    double tmp;
    marray_view<double,1> a(make_array(4), &tmp, ROW_MAJOR);
    marray_view<double,4> b(make_array(2, 5, 4, 2), &tmp, ROW_MAJOR);
    marray_view<double,1> c(make_array(4), &tmp, COLUMN_MAJOR);
    marray_view<double,4> d(make_array(2, 5, 4, 2), &tmp, COLUMN_MAJOR);
    marray_view<double,1> e(make_array(4), &tmp, make_array(3));
    marray_view<double,4> f(make_array(2, 5, 4, 2), &tmp, make_array(1, 5, 20, 45));

    EXPECT_EQ(make_array(4u), a.lengths());
    EXPECT_EQ(make_array(1l), a.strides());
    EXPECT_EQ(&tmp, a.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b.strides());
    EXPECT_EQ(&tmp, b.data());

    EXPECT_EQ(make_array(4u), c.lengths());
    EXPECT_EQ(make_array(1l), c.strides());
    EXPECT_EQ(&tmp, c.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), d.lengths());
    EXPECT_EQ(make_array(1l, 2l, 10l, 40l), d.strides());
    EXPECT_EQ(&tmp, d.data());

    EXPECT_EQ(make_array(4u), e.lengths());
    EXPECT_EQ(make_array(3l), e.strides());
    EXPECT_EQ(&tmp, e.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), f.lengths());
    EXPECT_EQ(make_array(1l, 5l, 20l, 45l), f.strides());
    EXPECT_EQ(&tmp, f.data());
}

TEST(marray, marray_view_construct_direct)
{
    double tmp;
    marray_view<double,1> a(4, &tmp, ROW_MAJOR);
    marray_view<double,4> b(2, 5, 4, 2, &tmp, ROW_MAJOR);
    marray_view<double,1> c(4, &tmp, COLUMN_MAJOR);
    marray_view<double,4> d(2, 5, 4, 2, &tmp, COLUMN_MAJOR);
    marray_view<double,1> e(4, &tmp, 3);
    marray_view<double,4> f(2, 5, 4, 2, &tmp, 1, 5, 20, 45);

    EXPECT_EQ(make_array(4u), a.lengths());
    EXPECT_EQ(make_array(1l), a.strides());
    EXPECT_EQ(&tmp, a.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b.strides());
    EXPECT_EQ(&tmp, b.data());

    EXPECT_EQ(make_array(4u), c.lengths());
    EXPECT_EQ(make_array(1l), c.strides());
    EXPECT_EQ(&tmp, c.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), d.lengths());
    EXPECT_EQ(make_array(1l, 2l, 10l, 40l), d.strides());
    EXPECT_EQ(&tmp, d.data());

    EXPECT_EQ(make_array(4u), e.lengths());
    EXPECT_EQ(make_array(3l), e.strides());
    EXPECT_EQ(&tmp, e.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), f.lengths());
    EXPECT_EQ(make_array(1l, 5l, 20l, 45l), f.strides());
    EXPECT_EQ(&tmp, f.data());
}

TEST(marray, marray_view_permute)
{
    double tmp;
    marray_view<double,1> a(4, &tmp, ROW_MAJOR);
    marray_view<double,4> b(2, 5, 4, 3, &tmp, ROW_MAJOR);
    const marray_view<double,1>& ac = a;
    const marray_view<double,4>& bc = b;

    auto acp1 = ac.permute(make_array(0));
    auto acp2 = ac.permute(0);

    auto bcp1 = bc.permute(make_array(2, 0, 3, 1));
    auto bcp2 = bc.permute(2, 0, 3, 1);

    EXPECT_TRUE((is_same<const_marray_view<double,1>,decltype(acp1)>::value));
    EXPECT_TRUE((is_same<const_marray_view<double,1>,decltype(acp2)>::value));
    EXPECT_TRUE((is_same<const_marray_view<double,4>,decltype(bcp1)>::value));
    EXPECT_TRUE((is_same<const_marray_view<double,4>,decltype(bcp2)>::value));

    auto ap1 = a.permute(make_array(0));
    auto ap2 = a.permute(0);

    auto bp1 = b.permute(make_array(2, 0, 3, 1));
    auto bp2 = b.permute(2, 0, 3, 1);

    EXPECT_TRUE((is_same<marray_view<double,1>,decltype(ap1)>::value));
    EXPECT_TRUE((is_same<marray_view<double,1>,decltype(ap2)>::value));
    EXPECT_TRUE((is_same<marray_view<double,4>,decltype(bp1)>::value));
    EXPECT_TRUE((is_same<marray_view<double,4>,decltype(bp2)>::value));

    EXPECT_EQ(make_array(4u), ap1.lengths());
    EXPECT_EQ(make_array(1l), ap1.strides());
    EXPECT_EQ(&tmp, ap1.data());

    EXPECT_EQ(make_array(4u), ap2.lengths());
    EXPECT_EQ(make_array(1l), ap2.strides());
    EXPECT_EQ(&tmp, ap2.data());

    EXPECT_EQ(make_array(4u, 2u, 3u, 5u), bp1.lengths());
    EXPECT_EQ(make_array(3l, 60l, 1l, 12l), bp1.strides());
    EXPECT_EQ(&tmp, bp1.data());

    EXPECT_EQ(make_array(4u, 2u, 3u, 5u), bp2.lengths());
    EXPECT_EQ(make_array(3l, 60l, 1l, 12l), bp2.strides());
    EXPECT_EQ(&tmp, bp2.data());
}

TEST(marray, marray_view_front)
{
    double tmp;
    marray_view<double,1> a(4, &tmp, ROW_MAJOR);
    marray_view<double,4> b(2, 5, 4, 3, &tmp, ROW_MAJOR);
    const marray_view<double,1>& ac = a;
    const marray_view<double,4>& bc = b;

    auto& acf1 = ac.front();
    auto& acf2 = ac.front(0);

    auto bcf1 = bc.front(0);
    auto bcf2 = bc.front(2);

    EXPECT_TRUE((is_same<const double&,decltype(acf1)>::value));
    EXPECT_TRUE((is_same<const double&,decltype(acf2)>::value));
    EXPECT_TRUE((is_same<const_marray_view<double,3>,decltype(bcf1)>::value));
    EXPECT_TRUE((is_same<const_marray_view<double,3>,decltype(bcf2)>::value));

    auto& af1 = a.front();
    auto& af2 = a.front(0);

    auto bf1 = b.front(0);
    auto bf2 = b.front(2);

    EXPECT_TRUE((is_same<double&,decltype(af1)>::value));
    EXPECT_TRUE((is_same<double&,decltype(af2)>::value));
    EXPECT_TRUE((is_same<marray_view<double,3>,decltype(bf1)>::value));
    EXPECT_TRUE((is_same<marray_view<double,3>,decltype(bf2)>::value));

    EXPECT_EQ(&tmp, &af1);
    EXPECT_EQ(&tmp, &af2);

    EXPECT_EQ(make_array(5u, 4u, 3u), bf1.lengths());
    EXPECT_EQ(make_array(12l, 3l, 1l), bf1.strides());
    EXPECT_EQ(&tmp, bf1.data());

    EXPECT_EQ(make_array(2u, 5u, 3u), bf2.lengths());
    EXPECT_EQ(make_array(60l, 12l, 1l), bf2.strides());
    EXPECT_EQ(&tmp, bf2.data());
}

TEST(marray, marray_view_back)
{
    double tmp;
    marray_view<double,1> a(4, &tmp, ROW_MAJOR);
    marray_view<double,4> b(2, 5, 4, 3, &tmp, ROW_MAJOR);
    const marray_view<double,1>& ac = a;
    const marray_view<double,4>& bc = b;

    auto& acb1 = ac.back();
    auto& acb2 = ac.back(0);

    auto bcb1 = bc.back(0);
    auto bcb2 = bc.back(2);

    EXPECT_TRUE((is_same<const double&,decltype(acb1)>::value));
    EXPECT_TRUE((is_same<const double&,decltype(acb2)>::value));
    EXPECT_TRUE((is_same<const_marray_view<double,3>,decltype(bcb1)>::value));
    EXPECT_TRUE((is_same<const_marray_view<double,3>,decltype(bcb2)>::value));

    auto& ab1 = a.back();
    auto& ab2 = a.back(0);

    auto bb1 = b.back(0);
    auto bb2 = b.back(2);

    EXPECT_TRUE((is_same<double&,decltype(ab1)>::value));
    EXPECT_TRUE((is_same<double&,decltype(ab2)>::value));
    EXPECT_TRUE((is_same<marray_view<double,3>,decltype(bb1)>::value));
    EXPECT_TRUE((is_same<marray_view<double,3>,decltype(bb2)>::value));

    EXPECT_EQ(&tmp+3, &ab1);
    EXPECT_EQ(&tmp+3, &ab2);

    EXPECT_EQ(make_array(5u, 4u, 3u), bb1.lengths());
    EXPECT_EQ(make_array(12l, 3l, 1l), bb1.strides());
    EXPECT_EQ(&tmp+60*1, bb1.data());

    EXPECT_EQ(make_array(2u, 5u, 3u), bb2.lengths());
    EXPECT_EQ(make_array(60l, 12l, 1l), bb2.strides());
    EXPECT_EQ(&tmp+3*3, bb2.data());
}

TEST(marray, marray_brackets)
{
    double tmp;
    marray_view<double,1> a(4, &tmp, ROW_MAJOR);
    marray_view<double,4> b(2, 5, 4, 3, &tmp, ROW_MAJOR);
    const marray_view<double,1>& ac = a;
    const marray_view<double,4>& bc = b;

    EXPECT_FALSE((is_convertible<decltype(ac[0]),decltype(a[0])>::value));
    EXPECT_FALSE((is_convertible<decltype(bc[0]),decltype(b[0])>::value));

    double& a1 = a[0];
    double& a2 = a[3];

    marray_view<double,3> b1 = b[0];
    marray_view<double,3> b2 = b[1];

    EXPECT_EQ(&tmp, &a1);
    EXPECT_EQ(&tmp+3, &a2);

    EXPECT_EQ(make_array(5u, 4u, 3u), b1.lengths());
    EXPECT_EQ(make_array(12l, 3l, 1l), b1.strides());
    EXPECT_EQ(&tmp, b1.data());

    EXPECT_EQ(make_array(5u, 4u, 3u), b2.lengths());
    EXPECT_EQ(make_array(12l, 3l, 1l), b2.strides());
    EXPECT_EQ(&tmp+60*1, b2.data());

    marray_view<double,1> a3 = a[range(2,4)];
    marray_view<double,1> a4 = a[all];

    marray_view<double,4> b3 = b[range(1,2)];
    marray_view<double,4> b4 = b[all];

    EXPECT_EQ(make_array(2u), a3.lengths());
    EXPECT_EQ(make_array(1l), a3.strides());
    EXPECT_EQ(&tmp+2, a3.data());

    EXPECT_EQ(make_array(4u), a4.lengths());
    EXPECT_EQ(make_array(1l), a4.strides());
    EXPECT_EQ(&tmp, a4.data());

    EXPECT_EQ(make_array(1u, 5u, 4u, 3u), b3.lengths());
    EXPECT_EQ(make_array(60l, 12l, 3l, 1l), b3.strides());
    EXPECT_EQ(&tmp+60*1, b3.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 3u), b4.lengths());
    EXPECT_EQ(make_array(60l, 12l, 3l, 1l), b4.strides());
    EXPECT_EQ(&tmp, b4.data());
}

TEST(marray, marray_parentheses)
{
    double tmp;
    marray_view<double,1> a(4, &tmp, ROW_MAJOR);
    marray_view<double,4> b(2, 5, 4, 3, &tmp, ROW_MAJOR);
    const marray_view<double,1>& ac = a;
    const marray_view<double,4>& bc = b;

    EXPECT_FALSE((is_convertible<decltype(ac(0)),decltype(a(0))>::value));
    EXPECT_FALSE((is_convertible<decltype(ac(range(2,4))),decltype(a(range(2,4)))>::value));
    EXPECT_FALSE((is_convertible<decltype(ac(all)),decltype(a(all))>::value));
    EXPECT_FALSE((is_convertible<decltype(bc(0, 0, 0, 0)),decltype(b(0, 0, 0, 0))>::value));
    EXPECT_FALSE((is_convertible<decltype(bc(1,range(1,2),0,2)),decltype(b(1,range(1,2),0,2))>::value));
    EXPECT_FALSE((is_convertible<decltype(bc(all,1,range(3),1)),decltype(b(all,1,range(3),1))>::value));
    EXPECT_FALSE((is_convertible<decltype(bc(all,range(3,4),range(3),all)),decltype(b(all,range(3,4),range(3),all))>::value));

    double& a1 = a(0);
    double& a2 = a(3);

    double& b1 = b(0, 0, 0, 0);
    double& b2 = b(1, 4, 1, 2);

    EXPECT_EQ(&tmp, &a1);
    EXPECT_EQ(&tmp+3, &a2);

    EXPECT_EQ(&tmp, &b1);
    EXPECT_EQ(&tmp+1*60+4*12+1*3+2, &b2);

    marray_view<double,1> a3 = a(range(2,4));
    marray_view<double,1> a4 = a(all);

    marray_view<double,1> b3 = b(1,range(1,2),0,2);
    marray_view<double,2> b4 = b(all,1,range(3),1);
    marray_view<double,4> b5 = b(all,range(3,4),range(3),all);

    EXPECT_EQ(make_array(2u), a3.lengths());
    EXPECT_EQ(make_array(1l), a3.strides());
    EXPECT_EQ(&tmp+2, a3.data());

    EXPECT_EQ(make_array(4u), a4.lengths());
    EXPECT_EQ(make_array(1l), a4.strides());
    EXPECT_EQ(&tmp, a4.data());

    EXPECT_EQ(make_array(1u), b3.lengths());
    EXPECT_EQ(make_array(12l), b3.strides());
    EXPECT_EQ(&tmp+1*60+1*12+2, b3.data());

    EXPECT_EQ(make_array(2u, 3u), b4.lengths());
    EXPECT_EQ(make_array(60l, 3l), b4.strides());
    EXPECT_EQ(&tmp+1*12+1, b4.data());

    EXPECT_EQ(make_array(2u, 1u, 3u, 3u), b5.lengths());
    EXPECT_EQ(make_array(60l, 12l, 3l, 1l), b5.strides());
    EXPECT_EQ(&tmp+3*12, b5.data());
}

TEST(marray, marray_view_accessors)
{
    double tmp;
    marray_view<double,1> a(4, &tmp, ROW_MAJOR);
    marray_view<double,4> b(2, 5, 4, 2, &tmp, ROW_MAJOR);
    const marray_view<double,1>& ac = a;
    const marray_view<double,4>& bc = b;

    EXPECT_FALSE((is_convertible<decltype(ac.data()),decltype(a.data())>::value));
    EXPECT_FALSE((is_convertible<decltype(bc.data()),decltype(b.data())>::value));

    EXPECT_EQ(make_array(4u), a.lengths());
    EXPECT_EQ(make_array(1l), a.strides());
    EXPECT_EQ(4, a.length());
    EXPECT_EQ(1, a.stride());
    EXPECT_EQ(4, a.length(0));
    EXPECT_EQ(1, a.stride(0));
    EXPECT_EQ(&tmp, a.data());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b.strides());
    EXPECT_EQ(2, b.length(0));
    EXPECT_EQ(5, b.length(1));
    EXPECT_EQ(4, b.length(2));
    EXPECT_EQ(2, b.length(3));
    EXPECT_EQ(40, b.stride(0));
    EXPECT_EQ(8, b.stride(1));
    EXPECT_EQ(2, b.stride(2));
    EXPECT_EQ(1, b.stride(3));
    EXPECT_EQ(&tmp, b.data());
}

TEST(marray, marray_view_assign)
{
    marray<double,3> a(2,3,4);
    marray_view<double,3> av(a);
    const_marray_view<double,3> acv(a);
    marray<double,3> b(2,3,4);
    marray_view<double,3> bv(b);

    for (int i = 0;i < 2*3*4;i++)
    {
        a.data()[i] = i;
    }

    bv = a;

    for (int i = 0;i < 2*3*4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i, b.data()[i]);
    }

    for (int i = 0;i < 2*3*4;i++)
    {
        a.data()[i] = 2*3*4-i;
    }

    bv = av;

    for (int i = 0;i < 2*3*4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(2*3*4-i, b.data()[i]);
    }

    for (int i = 0;i < 2*3*4;i++)
    {
        a.data()[i] = 5;
    }

    bv = acv;

    for (int i = 0;i < 2*3*4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(5, b.data()[i]);
    }
}

TEST(marray, marray_view_rotate)
{
    marray<double,1> a(27);
    marray_view<double,1> av(a);
    marray<double,3> b(3,3,3);
    marray_view<double,3> bv(b);

    for (int i = 0;i < 27;i++)
    {
        a.data()[i] = i;
        b.data()[i] = i;
    }

    av.rotate_dim(0, 3);

    for (int i = 0;i < 27;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ((i+3)%27, a[i]);
    }

    av.rotate_dim(0, -10);

    for (int i = 0;i < 27;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ((i+20)%27, a[i]);
    }

    av.rotate(make_array(3));

    for (int i = 0;i < 27;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ((i+23)%27, a[i]);
    }

    av.rotate(19);

    for (int i = 0;i < 27;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ((i+15)%27, a[i]);
    }

    bv.rotate_dim(0, 5);

    for (int i = 0;i < 3;i++)
    {
        SCOPED_TRACE(i);
        for (int j = 0;j < 3;j++)
        {
            SCOPED_TRACE(j);
            for (int k = 0;k < 3;k++)
            {
                SCOPED_TRACE(k);
                EXPECT_EQ(((i+5)%3)*9+((j+0)%3)*3+(k+0)%3, b[i][j][k]);
            }
        }
    }

    bv.rotate_dim(2, -4);

    for (int i = 0;i < 3;i++)
    {
        SCOPED_TRACE(i);
        for (int j = 0;j < 3;j++)
        {
            SCOPED_TRACE(j);
            for (int k = 0;k < 3;k++)
            {
                SCOPED_TRACE(k);
                EXPECT_EQ(((i+5)%3)*9+((j+0)%3)*3+(k+2)%3, b[i][j][k]);
            }
        }
    }

    bv.rotate(make_array(0,1,1));

    for (int i = 0;i < 3;i++)
    {
        SCOPED_TRACE(i);
        for (int j = 0;j < 3;j++)
        {
            SCOPED_TRACE(j);
            for (int k = 0;k < 3;k++)
            {
                SCOPED_TRACE(k);
                EXPECT_EQ(((i+5)%3)*9+((j+1)%3)*3+(k+0)%3, b[i][j][k]);
            }
        }
    }

    bv.rotate(0,1,1);

    for (int i = 0;i < 3;i++)
    {
        SCOPED_TRACE(i);
        for (int j = 0;j < 3;j++)
        {
            SCOPED_TRACE(j);
            for (int k = 0;k < 3;k++)
            {
                SCOPED_TRACE(k);
                EXPECT_EQ(((i+5)%3)*9+((j+2)%3)*3+(k+1)%3, b[i][j][k]);
            }
        }
    }
}

TEST(marray, marray_construct_empty)
{
    marray<double,1> a;
    marray<double,4> b;

    EXPECT_EQ(make_array(0u), a.lengths());
    EXPECT_EQ(make_array(0l), a.strides());
    EXPECT_EQ(nullptr, a.data());

    EXPECT_EQ(make_array(0u, 0u, 0u, 0u), b.lengths());
    EXPECT_EQ(make_array(0l, 0l, 0l, 0l), b.strides());
    EXPECT_EQ(nullptr, b.data());
}

TEST(marray, marray_construct_copy)
{
    double tmp;
    marray<double,1> a(4);
    marray<double,4> b(2, 5, 4, 2);
    marray_view<double,1> av(a);
    marray_view<double,4> bv(b);
    const_marray_view<double,1> acv(a);
    const_marray_view<double,4> bcv(b);

    for (int i = 0;i < 4;i++) a.data()[i] = i;
    for (int i = 0;i < 2*5*4*2;i++) b.data()[i] = i;

    marray<double,1> c(a);
    marray<double,4> d(b);

    EXPECT_EQ(make_array(4u), c.lengths());
    EXPECT_EQ(make_array(1l), c.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i, c.data()[i]);
    }

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), d.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), d.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i, d.data()[i]);
    }

    marray<double,1> e(av);
    marray<double,4> f(bv);

    EXPECT_EQ(make_array(4u), e.lengths());
    EXPECT_EQ(make_array(1l), e.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i, e.data()[i]);
    }

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), f.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), f.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i, f.data()[i]);
    }

    marray<double,1> g(acv);
    marray<double,4> h(bcv);

    EXPECT_EQ(make_array(4u), g.lengths());
    EXPECT_EQ(make_array(1l), g.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i, g.data()[i]);
    }

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), h.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), h.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i, h.data()[i]);
    }
}

TEST(marray, marray_construct_array)
{
    marray<double,1> a1(make_array(4));
    marray<double,4> b1(make_array(2, 5, 4, 2));
    marray<double,1> a2(make_array(4), 3.0);
    marray<double,4> b2(make_array(2, 5, 4, 2), 3.0);
    marray<double,1> a3(make_array(4), 3.0, COLUMN_MAJOR);
    marray<double,4> b3(make_array(2, 5, 4, 2), 3.0, COLUMN_MAJOR);
    marray<double,1> a4(make_array(4), uninitialized);
    marray<double,4> b4(make_array(2, 5, 4, 2), uninitialized);
    marray<double,1> a5(make_array(4), uninitialized, COLUMN_MAJOR);
    marray<double,4> b5(make_array(2, 5, 4, 2), uninitialized, COLUMN_MAJOR);

    EXPECT_EQ(make_array(4u), a1.lengths());
    EXPECT_EQ(make_array(1l), a1.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(0.0, a1.data()[i]);
    }

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b1.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b1.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(0.0, b1.data()[i]);
    }

    EXPECT_EQ(make_array(4u), a2.lengths());
    EXPECT_EQ(make_array(1l), a2.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, a2.data()[i]);
    }

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b2.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b2.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, b2.data()[i]);
    }

    EXPECT_EQ(make_array(4u), a3.lengths());
    EXPECT_EQ(make_array(1l), a3.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, a3.data()[i]);
    }

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b3.lengths());
    EXPECT_EQ(make_array(1l, 2l, 10l, 40l), b3.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, b3.data()[i]);
    }

    EXPECT_EQ(make_array(4u), a4.lengths());
    EXPECT_EQ(make_array(1l), a4.strides());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b4.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b4.strides());

    EXPECT_EQ(make_array(4u), a5.lengths());
    EXPECT_EQ(make_array(1l), a5.strides());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b5.lengths());
    EXPECT_EQ(make_array(1l, 2l, 10l, 40l), b5.strides());
}

TEST(marray, marray_construct_direct)
{
    marray<double,1> a1(4);
    marray<double,4> b1(2, 5, 4, 2);
    marray<double,1> a2(4, 3.0);
    marray<double,4> b2(2, 5, 4, 2, 3.0);
    marray<double,1> a3(4, 3.0, COLUMN_MAJOR);
    marray<double,4> b3(2, 5, 4, 2, 3.0, COLUMN_MAJOR);
    marray<double,1> a4(4, uninitialized);
    marray<double,4> b4(2, 5, 4, 2, uninitialized);
    marray<double,1> a5(4, uninitialized, COLUMN_MAJOR);
    marray<double,4> b5(2, 5, 4, 2, uninitialized, COLUMN_MAJOR);

    EXPECT_EQ(make_array(4u), a1.lengths());
    EXPECT_EQ(make_array(1l), a1.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(0.0, a1.data()[i]);
    }

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b1.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b1.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(0.0, b1.data()[i]);
    }

    EXPECT_EQ(make_array(4u), a2.lengths());
    EXPECT_EQ(make_array(1l), a2.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, a2.data()[i]);
    }

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b2.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b2.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, b2.data()[i]);
    }

    EXPECT_EQ(make_array(4u), a3.lengths());
    EXPECT_EQ(make_array(1l), a3.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, a3.data()[i]);
    }

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b3.lengths());
    EXPECT_EQ(make_array(1l, 2l, 10l, 40l), b3.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, b3.data()[i]);
    }

    EXPECT_EQ(make_array(4u), a4.lengths());
    EXPECT_EQ(make_array(1l), a4.strides());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b4.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b4.strides());

    EXPECT_EQ(make_array(4u), a5.lengths());
    EXPECT_EQ(make_array(1l), a5.strides());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b5.lengths());
    EXPECT_EQ(make_array(1l, 2l, 10l, 40l), b5.strides());
}

TEST(marray, marray_destruct)
{
    marray<double,1> a(4);
    marray<double,4> b(2, 5, 4, 2);

    a.~marray();
    b.~marray();

    EXPECT_EQ(make_array(0u), a.lengths());
    EXPECT_EQ(make_array(0l), a.strides());
    EXPECT_EQ(nullptr, a.data());

    EXPECT_EQ(make_array(0u, 0u, 0u, 0u), b.lengths());
    EXPECT_EQ(make_array(0l, 0l, 0l, 0l), b.strides());
    EXPECT_EQ(nullptr, b.data());

    new (&a) marray<double,1>();
    new (&b) marray<double,4>();
}

TEST(marray, marray_assign)
{
    marray<double,3> a(2,3,4);
    marray_view<double,3> av(a);
    const_marray_view<double,3> acv(a);
    marray<double,3> b(2,3,4);

    for (int i = 0;i < 2*3*4;i++)
    {
        a.data()[i] = i;
    }

    b = a;

    for (int i = 0;i < 2*3*4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(i, b.data()[i]);
    }

    for (int i = 0;i < 2*3*4;i++)
    {
        a.data()[i] = 2*3*4-i;
    }

    b = av;

    for (int i = 0;i < 2*3*4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(2*3*4-i, b.data()[i]);
    }

    for (int i = 0;i < 2*3*4;i++)
    {
        a.data()[i] = 5;
    }

    b = acv;

    for (int i = 0;i < 2*3*4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(5, b.data()[i]);
    }
}

TEST(marray, marray_reset_array)
{
    marray<double,1> a0(1);
    marray<double,4> b0(1, 1, 1, 1);
    marray<double,1> a1(1);
    marray<double,4> b1(1, 1, 1, 1);
    marray<double,1> a2(1);
    marray<double,4> b2(1, 1, 1, 1);
    marray<double,1> a3(1);
    marray<double,4> b3(1, 1, 1, 1);
    marray<double,1> a4(1);
    marray<double,4> b4(1, 1, 1, 1);
    marray<double,1> a5(1);
    marray<double,4> b5(1, 1, 1, 1);

    a0.reset();
    b0.reset();
    a1.reset(make_array(4));
    b1.reset(make_array(2, 5, 4, 2));
    a2.reset(make_array(4), 3.0);
    b2.reset(make_array(2, 5, 4, 2), 3.0);
    a3.reset(make_array(4), 3.0, COLUMN_MAJOR);
    b3.reset(make_array(2, 5, 4, 2), 3.0, COLUMN_MAJOR);
    a4.reset(make_array(4), uninitialized);
    b4.reset(make_array(2, 5, 4, 2), uninitialized);
    a5.reset(make_array(4), uninitialized, COLUMN_MAJOR);
    b5.reset(make_array(2, 5, 4, 2), uninitialized, COLUMN_MAJOR);

    EXPECT_EQ(make_array(0u), a0.lengths());
    EXPECT_EQ(make_array(0l), a0.strides());
    EXPECT_EQ(nullptr, a0.data());

    EXPECT_EQ(make_array(0u, 0u, 0u, 0u), b0.lengths());
    EXPECT_EQ(make_array(0l, 0l, 0l, 0l), b0.strides());
    EXPECT_EQ(nullptr, b0.data());

    EXPECT_EQ(make_array(4u), a1.lengths());
    EXPECT_EQ(make_array(1l), a1.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(0.0, a1.data()[i]);
    }

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b1.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b1.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(0.0, b1.data()[i]);
    }

    EXPECT_EQ(make_array(4u), a2.lengths());
    EXPECT_EQ(make_array(1l), a2.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, a2.data()[i]);
    }

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b2.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b2.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, b2.data()[i]);
    }

    EXPECT_EQ(make_array(4u), a3.lengths());
    EXPECT_EQ(make_array(1l), a3.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, a3.data()[i]);
    }

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b3.lengths());
    EXPECT_EQ(make_array(1l, 2l, 10l, 40l), b3.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, b3.data()[i]);
    }

    EXPECT_EQ(make_array(4u), a4.lengths());
    EXPECT_EQ(make_array(1l), a4.strides());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b4.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b4.strides());

    EXPECT_EQ(make_array(4u), a5.lengths());
    EXPECT_EQ(make_array(1l), a5.strides());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b5.lengths());
    EXPECT_EQ(make_array(1l, 2l, 10l, 40l), b5.strides());
}

TEST(marray, marray_reset_direct)
{
    marray<double,1> a1(1);
    marray<double,4> b1(1, 1, 1, 1);
    marray<double,1> a2(1);
    marray<double,4> b2(1, 1, 1, 1);
    marray<double,1> a3(1);
    marray<double,4> b3(1, 1, 1, 1);
    marray<double,1> a4(1);
    marray<double,4> b4(1, 1, 1, 1);
    marray<double,1> a5(1);
    marray<double,4> b5(1, 1, 1, 1);

    a1.reset(4);
    b1.reset(2, 5, 4, 2);
    a2.reset(4, 3.0);
    b2.reset(2, 5, 4, 2, 3.0);
    a3.reset(4, 3.0, COLUMN_MAJOR);
    b3.reset(2, 5, 4, 2, 3.0, COLUMN_MAJOR);
    a4.reset(4, uninitialized);
    b4.reset(2, 5, 4, 2, uninitialized);
    a5.reset(4, uninitialized, COLUMN_MAJOR);
    b5.reset(2, 5, 4, 2, uninitialized, COLUMN_MAJOR);

    EXPECT_EQ(make_array(4u), a1.lengths());
    EXPECT_EQ(make_array(1l), a1.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(0.0, a1.data()[i]);
    }

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b1.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b1.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(0.0, b1.data()[i]);
    }

    EXPECT_EQ(make_array(4u), a2.lengths());
    EXPECT_EQ(make_array(1l), a2.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, a2.data()[i]);
    }

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b2.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b2.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, b2.data()[i]);
    }

    EXPECT_EQ(make_array(4u), a3.lengths());
    EXPECT_EQ(make_array(1l), a3.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, a3.data()[i]);
    }

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b3.lengths());
    EXPECT_EQ(make_array(1l, 2l, 10l, 40l), b3.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, b3.data()[i]);
    }

    EXPECT_EQ(make_array(4u), a4.lengths());
    EXPECT_EQ(make_array(1l), a4.strides());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b4.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b4.strides());

    EXPECT_EQ(make_array(4u), a5.lengths());
    EXPECT_EQ(make_array(1l), a5.strides());

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b5.lengths());
    EXPECT_EQ(make_array(1l, 2l, 10l, 40l), b5.strides());
}

TEST(marray, marray_resize)
{
    marray<double,1> a(4, 3.0);
    marray<double,4> b(2, 5, 4, 2, 3.0);

    a.resize(make_array(6));

    EXPECT_EQ(make_array(6u), a.lengths());
    EXPECT_EQ(make_array(1l), a.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, a.data()[i]);
    }

    for (int i = 4;i < 6;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(0.0, a.data()[i]);
    }

    a.resize(make_array(8), 1.0);

    EXPECT_EQ(make_array(8u), a.lengths());
    EXPECT_EQ(make_array(1l), a.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, a.data()[i]);
    }

    for (int i = 4;i < 6;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(0.0, a.data()[i]);
    }

    for (int i = 6;i < 8;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(1.0, a.data()[i]);
    }

    a.resize(4);

    EXPECT_EQ(make_array(4u), a.lengths());
    EXPECT_EQ(make_array(1l), a.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, a.data()[i]);
    }

    a.resize(6, -2.0);

    EXPECT_EQ(make_array(6u), a.lengths());
    EXPECT_EQ(make_array(1l), a.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, a.data()[i]);
    }

    for (int i = 4;i < 6;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(-2.0, a.data()[i]);
    }

    b.resize(make_array(4, 5, 4, 2));

    EXPECT_EQ(make_array(4u, 5u, 4u, 2u), b.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, b.data()[i]);
    }

    for (int i = 2*5*4*2;i < 4*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(0.0, b.data()[i]);
    }

    b.resize(make_array(6, 5, 4, 2), 1.0);

    EXPECT_EQ(make_array(6u, 5u, 4u, 2u), b.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, b.data()[i]);
    }

    for (int i = 2*5*4*2;i < 4*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(0.0, b.data()[i]);
    }

    for (int i = 4*5*4*2;i < 6*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(1.0, b.data()[i]);
    }

    b.resize(2, 4, 3, 2);

    EXPECT_EQ(make_array(2u, 4u, 3u, 2u), b.lengths());
    EXPECT_EQ(make_array(24l, 6l, 2l, 1l), b.strides());

    for (int i = 0;i < 2*4*3*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, b.data()[i]);
    }

    b.resize(5, 4, 3, 2, -2.0);

    EXPECT_EQ(make_array(5u, 4u, 3u, 2u), b.lengths());
    EXPECT_EQ(make_array(24l, 6l, 2l, 1l), b.strides());

    for (int i = 0;i < 2*4*3*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, b.data()[i]);
    }

    for (int i = 2*4*3*2;i < 5*4*3*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(-2.0, b.data()[i]);
    }
}

TEST(marray, marray_push_back)
{
    marray<double,1> a(4, 3.0);
    marray<double,4> b(2, 5, 4, 2, 3.0);
    marray<double,3> c(5, 4, 2, 1.0);
    marray<double,3> d(4, 4, 2);

    a.push_back(1.0);
    a.push_back(0, 1.0);

    EXPECT_EQ(make_array(6u), a.lengths());
    EXPECT_EQ(make_array(1l), a.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, a.data()[i]);
    }

    for (int i = 4;i < 6;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(1.0, a.data()[i]);
    }

    b.push_back(0, c);
    b.push_back(0, c);

    EXPECT_EQ(make_array(4u, 5u, 4u, 2u), b.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, b.data()[i]);
    }

    for (int i = 2*5*4*2;i < 4*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(1.0, b.data()[i]);
    }

    b.push_back(1, d);

    EXPECT_EQ(make_array(4u, 6u, 4u, 2u), b.lengths());
    EXPECT_EQ(make_array(48l, 8l, 2l, 1l), b.strides());
}

TEST(marray, marray_pop_back)
{
    marray<double,1> a(6, 3.0);
    marray<double,4> b(4, 6, 4, 2, 3.0);

    a.pop_back();
    a.pop_back(0);

    EXPECT_EQ(make_array(4u), a.lengths());
    EXPECT_EQ(make_array(1l), a.strides());

    for (int i = 0;i < 4;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, a.data()[i]);
    }

    b.pop_back(0);
    b.pop_back(0);

    EXPECT_EQ(make_array(2u, 6u, 4u, 2u), b.lengths());
    EXPECT_EQ(make_array(48l, 8l, 2l, 1l), b.strides());

    for (int i = 0;i < 2*6*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, b.data()[i]);
    }

    b.pop_back(1);

    EXPECT_EQ(make_array(2u, 5u, 4u, 2u), b.lengths());
    EXPECT_EQ(make_array(40l, 8l, 2l, 1l), b.strides());

    for (int i = 0;i < 2*5*4*2;i++)
    {
        SCOPED_TRACE(i);
        EXPECT_EQ(3.0, b.data()[i]);
    }
}

TEST(marray, marray_swap)
{
    marray<double,1> a1(6);
    marray<double,4> b1(4, 6, 4, 2);
    marray<double,1> a2(3);
    marray<double,4> b2(2, 4, 1, 3);

    double* a1p = a1.data();
    double* b1p = b1.data();
    double* a2p = a2.data();
    double* b2p = b2.data();

    a1.swap(a2);
    b1.swap(b2);

    EXPECT_EQ(make_array(3u), a1.lengths());
    EXPECT_EQ(make_array(1l), a1.strides());
    EXPECT_EQ(a2p, a1.data());

    EXPECT_EQ(make_array(2u, 4u, 1u, 3u), b1.lengths());
    EXPECT_EQ(make_array(12l, 3l, 3l, 1l), b1.strides());
    EXPECT_EQ(b2p, b1.data());

    EXPECT_EQ(make_array(6u), a2.lengths());
    EXPECT_EQ(make_array(1l), a2.strides());
    EXPECT_EQ(a1p, a2.data());

    EXPECT_EQ(make_array(4u, 6u, 4u, 2u), b2.lengths());
    EXPECT_EQ(make_array(48l, 8l, 2l, 1l), b2.strides());
    EXPECT_EQ(b1p, b2.data());

    swap(a1, a2);
    swap(b1, b2);

    EXPECT_EQ(make_array(6u), a1.lengths());
    EXPECT_EQ(make_array(1l), a1.strides());
    EXPECT_EQ(a1p, a1.data());

    EXPECT_EQ(make_array(4u, 6u, 4u, 2u), b1.lengths());
    EXPECT_EQ(make_array(48l, 8l, 2l, 1l), b1.strides());
    EXPECT_EQ(b1p, b1.data());

    EXPECT_EQ(make_array(3u), a2.lengths());
    EXPECT_EQ(make_array(1l), a2.strides());
    EXPECT_EQ(a2p, a2.data());

    EXPECT_EQ(make_array(2u, 4u, 1u, 3u), b2.lengths());
    EXPECT_EQ(make_array(12l, 3l, 3l, 1l), b2.strides());
    EXPECT_EQ(b2p, b2.data());
}
