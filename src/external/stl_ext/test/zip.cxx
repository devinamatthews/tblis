#include <vector>
#include <memory>

#include "gtest/gtest.h"

#include "type_traits.hpp"
#include "vector.hpp"
#include "zip.hpp"

using namespace std;
using namespace stl_ext;

TEST(unit_zip, min_size)
{
    auto v =  make_tuple(vec(0,1), vec(2), vec(3,4,5,6));
    EXPECT_EQ(1, detail::min_size(v));
}

TEST(unit_zip, cbegin)
{
    auto v =  make_tuple(vec(0,1), vec(2), vec(3,4,5,6));
    auto i = detail::cbegin(v);
    EXPECT_EQ(make_tuple(get<0>(v).cbegin(),
                         get<1>(v).cbegin(),
                         get<2>(v).cbegin()), i);
}

TEST(unit_zip, increment)
{
    auto v =  make_tuple(vec(0,1), vec(2), vec(3,4,5,6));
    auto i = detail::cbegin(v);
    detail::increment(i);
    EXPECT_EQ(make_tuple(next(get<0>(v).cbegin()),
                         next(get<1>(v).cbegin()),
                         next(get<2>(v).cbegin())), i);
}

TEST(unit_zip, not_end)
{
    auto v =  make_tuple(vec(0,1), vec(2), vec(3,4,5,6));
    auto i = detail::cbegin(v);
    EXPECT_EQ(true, detail::not_end(i, v));
    detail::increment(i);
    EXPECT_EQ(false, detail::not_end(i, v));
}

TEST(unit_zip, reserve)
{
    auto v = make_tuple(vector<int>(), vector<int>());
    EXPECT_EQ(0, get<0>(v).capacity());
    EXPECT_EQ(0, get<1>(v).capacity());
    detail::reserve(v, 10);
    EXPECT_GE(10, get<0>(v).capacity());
    EXPECT_GE(10, get<1>(v).capacity());
}

TEST(unit_zip, emplace_back)
{
    auto v = make_tuple(vector<int>(), vector<int>());
    auto e = make_tuple(0, 2);
    detail::emplace_back(v, e);
    detail::emplace_back(v, make_tuple(1, 3));
    EXPECT_EQ(make_tuple(vec(0,1), vec(2,3)), v);
}

template <typename T, T... S>
struct print_integer_sequence;

template <typename T>
struct print_integer_sequence<T>
{
    print_integer_sequence(ostream& os)
    {
        os << "(null)";
    }
};

template <typename T, T S1>
struct print_integer_sequence<T, S1>
{
    print_integer_sequence(ostream& os)
    {
        os << S1;
    }
};

template <typename T, T S1, T... S>
struct print_integer_sequence<T, S1, S...>
{
    print_integer_sequence(ostream& os)
    {
        os << S1 << ' ';
        print_integer_sequence<T, S...>{os};
    }
};

template <typename T, T... S>
ostream& operator<<(ostream& os, const detail::integer_sequence<T,S...>& x)
{
    print_integer_sequence<T,S...>{os};
    return os;
}

TEST(unit_zip, concat_sequences)
{
    EXPECT_TRUE((is_same<detail::integer_sequence<size_t>,
                         typename detail::concat_sequences<size_t,
                             detail::integer_sequence<size_t>,
                             detail::integer_sequence<size_t>>::type>::value));
    EXPECT_TRUE((is_same<detail::integer_sequence<size_t,0>,
                         typename detail::concat_sequences<size_t,
                             detail::integer_sequence<size_t,0>,
                             detail::integer_sequence<size_t>>::type>::value));
    EXPECT_TRUE((is_same<detail::integer_sequence<size_t,0>,
                         typename detail::concat_sequences<size_t,
                             detail::integer_sequence<size_t>,
                             detail::integer_sequence<size_t,0>>::type>::value));
    EXPECT_TRUE((is_same<detail::integer_sequence<size_t,0,1>,
                         typename detail::concat_sequences<size_t,
                             detail::integer_sequence<size_t,0>,
                             detail::integer_sequence<size_t,0>>::type>::value));
    EXPECT_TRUE((is_same<detail::integer_sequence<size_t,0,1,2,3,4,5>,
                         typename detail::concat_sequences<size_t,
                             detail::integer_sequence<size_t,0,1,2>,
                             detail::integer_sequence<size_t,0,1,2>>::type>::value));
}

TEST(unit_zip, static_range)
{
    EXPECT_TRUE((is_same<detail::integer_sequence<size_t>,
                         detail::static_range<0>>::value));
    EXPECT_TRUE((is_same<detail::integer_sequence<size_t,0>,
                         detail::static_range<1>>::value));
    EXPECT_TRUE((is_same<detail::integer_sequence<size_t,0,1>,
                         detail::static_range<2>>::value));
    EXPECT_TRUE((is_same<detail::integer_sequence<size_t,0,1,2>,
                         detail::static_range<3>>::value));
    EXPECT_TRUE((is_same<detail::integer_sequence<size_t,0,1,2,3>,
                         detail::static_range<4>>::value));
    EXPECT_TRUE((is_same<detail::integer_sequence<size_t,0,1,2,3,4>,
                         detail::static_range<5>>::value));
    EXPECT_TRUE((is_same<detail::integer_sequence<size_t,0,1,2,3,4,5>,
                         detail::static_range<6>>::value));
    EXPECT_TRUE((is_same<detail::integer_sequence<size_t,0,1,2,3,4,5,6>,
                         detail::static_range<7>>::value));
    EXPECT_TRUE((is_same<detail::integer_sequence<size_t,0,1,2,3,4,5,6,7>,
                         detail::static_range<8>>::value));
}

TEST(unit_zip, call)
{
    call([](int x, unique_ptr<int>&& y, const vector<int>& z)
         {
             EXPECT_EQ(3, x);
             EXPECT_EQ(1, *y);
             EXPECT_EQ(vector<int>({0,1,2}), z);
         },
         forward_as_tuple(3, unique_ptr<int>(new int(1)), vec(0,1,2)));

    int x = 3;
    unique_ptr<int> y(new int(1));
    vector<int> z = {0,1,2};
    call([](int x, unique_ptr<int>&& y, const vector<int>& z)
         {
             EXPECT_EQ(3, x);
             EXPECT_EQ(1, *y);
             EXPECT_EQ(vector<int>({0,1,2}), z);
         },
         forward_as_tuple(x, std::move(y), z));
}

TEST(unit_zip, zip)
{
    EXPECT_EQ(vec(make_tuple(0,3), make_tuple(1,4), make_tuple(2,5)),
              zip(forward_as_tuple(vec(0,1,2), vec(3,4,5))));
    EXPECT_EQ(vec(make_tuple(0,3), make_tuple(1,4), make_tuple(2,5)),
              zip(vec(0,1,2), vec(3,4,5)));
    vector<int> v1{0,1,2};
    vector<int> v2{3,4,5};
    EXPECT_EQ(vec(make_tuple(0,3), make_tuple(1,4), make_tuple(2,5)),
              zip(forward_as_tuple(v1, v2)));
    EXPECT_EQ(vec(make_tuple(0,3), make_tuple(1,4), make_tuple(2,5)),
              zip(v1, v2));
}

TEST(unit_zip, unzip)
{
    auto v = vec(make_tuple(0,3), make_tuple(1,4), make_tuple(2,5));
    EXPECT_EQ(make_tuple(vec(0,1,2), vec(3,4,5)), unzip(v));
    EXPECT_EQ(make_tuple(vec(0,1,2), vec(3,4,5)),
              unzip(vec(make_tuple(0,3), make_tuple(1,4), make_tuple(2,5))));
}
