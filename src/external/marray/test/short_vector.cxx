#include "short_vector.hpp"
#include "gtest/gtest.h"

#include <list>
#include <vector>

using namespace std;
using namespace MArray;

TEST(short_vector, constructor)
{
    short_vector<int, 4> v1;
    EXPECT_EQ(0u, v1.size());
    EXPECT_EQ(4u, v1.capacity());

    short_vector<int, 4> v2(2);
    EXPECT_EQ(2u, v2.size());
    EXPECT_EQ(4u, v2.capacity());

    short_vector<int, 4> v3(6);
    EXPECT_EQ(6u, v3.size());
    EXPECT_LE(6u, v3.capacity());

    vector<int> v{0, 2};
    short_vector<int, 4> v4(v.begin(), v.end());
    EXPECT_EQ(2u, v4.size());
    EXPECT_EQ(4u, v4.capacity());

    short_vector<int, 4> v5(v4);
    EXPECT_EQ(2u, v5.size());
    EXPECT_EQ(4u, v5.capacity());
    EXPECT_EQ(v5, v4);

    short_vector<int, 4> v6(short_vector<int, 4>(3));
    EXPECT_EQ(3u, v6.size());
    EXPECT_EQ(4u, v6.capacity());

    short_vector<int, 4> v7{0, 2, 4};
    EXPECT_EQ(3u, v7.size());
    EXPECT_EQ(4u, v7.capacity());

    short_vector<int, 4> v8{0, 1, 2, 3, 4, 5, 6};
    EXPECT_EQ(7u, v8.size());
    EXPECT_LE(7u, v8.capacity());
}

TEST(short_vector, assignment)
{
    short_vector<int, 4> v1{0, 2, 4};
    short_vector<int, 4> v2;
    EXPECT_NE(v1, v2);
    v2 = v1;
    EXPECT_EQ(v1, v2);
    v2.clear();
    EXPECT_NE(v1, v2);
    v2 = short_vector<int, 4>{0, 2, 4};
    EXPECT_EQ(v1, v2);

    short_vector<int, 4> v3{0, 2, 4, 6, 9, 2, 5};
    short_vector<int, 4> v4;
    EXPECT_NE(v3, v4);
    v4 = v3;
    EXPECT_EQ(v3, v4);
    v4 = v1;
    EXPECT_EQ(v1, v4);
    v2 = v3;
    EXPECT_EQ(v3, v2);
    v4.clear();
    EXPECT_NE(v3, v4);
    v4 = short_vector<int, 4>{0, 2, 4};
    EXPECT_EQ(v1, v4);
    v4 = short_vector<int, 4>{0, 2, 4, 6, 9, 2, 5};
    EXPECT_EQ(v3, v4);
    v2 = short_vector<int, 4>{0, 2, 4, 6, 9, 2, 5};
    EXPECT_EQ(v3, v2);

    short_vector<int, 4> v5;
    v5 = {0, 2, 4};
    EXPECT_EQ(v1, v5);
    v5 = {0, 2, 4, 6, 9, 2, 5};
    EXPECT_EQ(v3, v5);
    v5 = {0, 2, 4};
    EXPECT_EQ(v1, v5);
}

TEST(short_vector, assign)
{
    short_vector<int, 4> v1{0, 2, 4};
    short_vector<int, 4> v2{0, 2, 4, 6, 9, 2, 5};
    short_vector<int, 4> v3{5, 5, 5};
    short_vector<int, 4> v4{5, 5, 5, 5, 5, 5, 5};

    short_vector<int, 4> v5;
    v5.assign({0, 2, 4});
    EXPECT_EQ(v1, v5);
    v5.assign({0, 2, 4, 6, 9, 2, 5});
    EXPECT_EQ(v2, v5);
    v5.assign({0, 2, 4});
    EXPECT_EQ(v1, v5);

    short_vector<int, 4> v6;
    v6.assign(v1.begin(), v1.end());
    EXPECT_EQ(v1, v6);
    v6.assign(v2.begin(), v2.end());
    EXPECT_EQ(v2, v6);
    v6.assign(v1.begin(), v1.end());
    EXPECT_EQ(v1, v6);

    short_vector<int, 4> v7;
    v7.assign(3, 5);
    EXPECT_EQ(v3, v7);
    v7.assign(7, 5);
    EXPECT_EQ(v4, v7);
}

TEST(short_vector, access)
{
    short_vector<int, 4> v1{0, 2, 4};
    short_vector<int, 4> v2{0, 2, 4, 6, 9, 2, 5};

    EXPECT_EQ(0, v1.at(0));
    EXPECT_EQ(4, v1.at(2));
    EXPECT_EQ(0, v1[0]);
    EXPECT_EQ(4, v1[2]);
    EXPECT_THROW(v1.at(-1), out_of_range);
    EXPECT_THROW(v1.at(3), out_of_range);
    EXPECT_EQ(0, v1.front());
    EXPECT_EQ(4, v1.back());
    EXPECT_EQ(0, v1.data()[0]);
    EXPECT_EQ(4, v1.data()[2]);

    EXPECT_EQ(0, v2.at(0));
    EXPECT_EQ(4, v2.at(2));
    EXPECT_EQ(9, v2.at(4));
    EXPECT_EQ(2, v2.at(5));
    EXPECT_EQ(0, v2[0]);
    EXPECT_EQ(4, v2[2]);
    EXPECT_EQ(9, v2[4]);
    EXPECT_EQ(2, v2[5]);
    EXPECT_THROW(v2.at(-1), out_of_range);
    EXPECT_THROW(v2.at(7), out_of_range);
    EXPECT_EQ(0, v2.front());
    EXPECT_EQ(5, v2.back());
    EXPECT_EQ(0, v2.data()[0]);
    EXPECT_EQ(4, v2.data()[2]);
    EXPECT_EQ(9, v2.data()[4]);
    EXPECT_EQ(2, v2.data()[5]);

    EXPECT_FALSE(v1.data() == v2.data());
}

TEST(short_vector, iterator)
{
    short_vector<int, 4> v1{0, 2, 4};
    short_vector<int, 4> v2{0, 2, 4, 6, 9, 2, 5};

    EXPECT_EQ(0, *v1.begin());
    EXPECT_EQ(2, *next(v1.begin()));
    EXPECT_EQ(4, *prev(v1.end()));
    EXPECT_EQ(4, *v1.rbegin());
    EXPECT_EQ(2, *next(v1.rbegin()));
    EXPECT_EQ(0, *prev(v1.rend()));
    EXPECT_EQ(0, *v1.cbegin());
    EXPECT_EQ(2, *next(v1.cbegin()));
    EXPECT_EQ(4, *prev(v1.cend()));
    EXPECT_EQ(4, *v1.crbegin());
    EXPECT_EQ(2, *next(v1.crbegin()));
    EXPECT_EQ(0, *prev(v1.crend()));

    EXPECT_EQ(0, *v2.begin());
    EXPECT_EQ(2, *next(v2.begin()));
    EXPECT_EQ(5, *prev(v2.end()));
    EXPECT_EQ(5, *v2.rbegin());
    EXPECT_EQ(2, *next(v2.rbegin()));
    EXPECT_EQ(0, *prev(v2.rend()));
    EXPECT_EQ(0, *v2.cbegin());
    EXPECT_EQ(2, *next(v2.cbegin()));
    EXPECT_EQ(5, *prev(v2.cend()));
    EXPECT_EQ(5, *v2.crbegin());
    EXPECT_EQ(2, *next(v2.crbegin()));
    EXPECT_EQ(0, *prev(v2.crend()));
}

TEST(short_vector, size_capacity_empty)
{
    short_vector<int, 4> v1;

    EXPECT_EQ(0u, v1.size());
    EXPECT_EQ(4u, v1.capacity());
    EXPECT_TRUE(v1.empty());

    v1 = {0, 3, 4};

    EXPECT_EQ(3u, v1.size());
    EXPECT_EQ(4u, v1.capacity());
    EXPECT_FALSE(v1.empty());

    v1 = {0, 3, 4, 0, 4, 2, 4, 8};

    EXPECT_EQ(8u, v1.size());
    EXPECT_LE(8u, v1.capacity());
    EXPECT_FALSE(v1.empty());

    v1.clear();

    EXPECT_EQ(0u, v1.size());
    EXPECT_LE(8u, v1.capacity());
    EXPECT_TRUE(v1.empty());

    v1 = {0, 3, 4, 0, 4};
    v1.shrink_to_fit();

    EXPECT_EQ(5u, v1.size());
    EXPECT_EQ(5u, v1.capacity());
    EXPECT_FALSE(v1.empty());

    v1.clear();
    v1.shrink_to_fit();

    EXPECT_EQ(0u, v1.size());
    EXPECT_EQ(4u, v1.capacity());
    EXPECT_TRUE(v1.empty());

    v1.reserve(567);

    EXPECT_EQ(0u, v1.size());
    EXPECT_LE(567u, v1.capacity());
    EXPECT_TRUE(v1.empty());
}

TEST(short_vector, insert_emplace)
{
    short_vector<int, 4> v1;

    v1.insert(v1.begin(), 5);
    EXPECT_EQ((short_vector<int, 4>{5}), v1);
    v1.insert(v1.begin(), 4);
    EXPECT_EQ((short_vector<int, 4>{4, 5}), v1);
    v1.insert(v1.end(), 6);
    EXPECT_EQ((short_vector<int, 4>{4, 5, 6}), v1);
    v1.insert(next(v1.begin()), 3);
    EXPECT_EQ((short_vector<int, 4>{4, 3, 5, 6}), v1);
    v1.insert(v1.end(), 7);
    EXPECT_EQ((short_vector<int, 4>{4, 3, 5, 6, 7}), v1);
    v1.insert(next(v1.begin()), 2);
    EXPECT_EQ((short_vector<int, 4>{4, 2, 3, 5, 6, 7}), v1);
    v1.insert(v1.begin(), 1);
    EXPECT_EQ((short_vector<int, 4>{1, 4, 2, 3, 5, 6, 7}), v1);
    v1.emplace(v1.begin(), 0);
    EXPECT_EQ((short_vector<int, 4>{0, 1, 4, 2, 3, 5, 6, 7}), v1);
    v1.emplace(v1.begin()+4, 8);
    EXPECT_EQ((short_vector<int, 4>{0, 1, 4, 2, 8, 3, 5, 6, 7}), v1);
    v1.emplace(v1.end(), 4);
    EXPECT_EQ((short_vector<int, 4>{0, 1, 4, 2, 8, 3, 5, 6, 7, 4}), v1);

    short_vector<int, 4> v2;

    v2.insert(v2.begin(), 2, 5);
    EXPECT_EQ((short_vector<int, 4>{5, 5}), v2);
    v2.insert(v2.begin(), 2, 4);
    EXPECT_EQ((short_vector<int, 4>{4, 4, 5, 5}), v2);
    v2.insert(v2.end(), 2, 6);
    EXPECT_EQ((short_vector<int, 4>{4, 4, 5, 5, 6, 6}), v2);
    v2.insert(next(v2.begin()), 3, 3);
    EXPECT_EQ((short_vector<int, 4>{4, 3, 3, 3, 4, 5, 5, 6, 6}), v2);
    v2.insert(v2.end(), 1, 7);
    EXPECT_EQ((short_vector<int, 4>{4, 3, 3, 3, 4, 5, 5, 6, 6, 7}), v2);

    short_vector<int, 4> v3;
    vector<int> v{1, 2};
    list<int> l{3, 4};

    v3.insert(v3.begin(), v.begin(), v.end());
    EXPECT_EQ((short_vector<int, 4>{1, 2}), v3);
    v3.insert(v3.begin(), l.begin(), l.end());
    EXPECT_EQ((short_vector<int, 4>{3, 4, 1, 2}), v3);
    v3.insert(v3.end(), l.begin(), l.end());
    EXPECT_EQ((short_vector<int, 4>{3, 4, 1, 2, 3, 4}), v3);
    v3.insert(next(v3.begin()), l.begin(), l.end());
    EXPECT_EQ((short_vector<int, 4>{3, 3, 4, 4, 1, 2, 3, 4}), v3);
    v3.insert(prev(v3.end()), v.begin(), v.end());
    EXPECT_EQ((short_vector<int, 4>{3, 3, 4, 4, 1, 2, 3, 1, 2, 4}), v3);

    short_vector<int, 4> v4;

    v4.insert(v4.begin(), {1, 2});
    EXPECT_EQ((short_vector<int, 4>{1, 2}), v4);
    v4.insert(v4.begin(), {3, 4});
    EXPECT_EQ((short_vector<int, 4>{3, 4, 1, 2}), v4);
    v4.insert(v4.end(), {3, 4});
    EXPECT_EQ((short_vector<int, 4>{3, 4, 1, 2, 3, 4}), v4);
    v4.insert(next(v4.begin()), {3, 4});
    EXPECT_EQ((short_vector<int, 4>{3, 3, 4, 4, 1, 2, 3, 4}), v4);
    v4.insert(prev(v4.end()), {1, 2});
    EXPECT_EQ((short_vector<int, 4>{3, 3, 4, 4, 1, 2, 3, 1, 2, 4}), v4);
}

TEST(short_vector, erase)
{
    short_vector<int, 4> v1{1, 2, 3, 4, 5, 6, 7, 8, 9};

    v1.erase(v1.begin());
    EXPECT_EQ((short_vector<int, 4>{2, 3, 4, 5, 6, 7, 8, 9}), v1);
    v1.erase(prev(v1.end()));
    EXPECT_EQ((short_vector<int, 4>{2, 3, 4, 5, 6, 7, 8}), v1);
    v1.erase(v1.begin()+3);
    EXPECT_EQ((short_vector<int, 4>{2, 3, 4, 6, 7, 8}), v1);
    v1.erase(v1.begin(), v1.begin()+2);
    EXPECT_EQ((short_vector<int, 4>{4, 6, 7, 8}), v1);
    v1.erase(v1.begin()+1, v1.begin()+3);
    EXPECT_EQ((short_vector<int, 4>{4, 8}), v1);
}

TEST(short_vector, push_pop)
{
    short_vector<int, 4> v1;

    v1.push_back(2);
    EXPECT_EQ((short_vector<int, 4>{2}), v1);
    v1.push_back(3);
    EXPECT_EQ((short_vector<int, 4>{2, 3}), v1);
    v1.emplace_back(1);
    EXPECT_EQ((short_vector<int, 4>{2, 3, 1}), v1);
    v1.emplace_back(4);
    EXPECT_EQ((short_vector<int, 4>{2, 3, 1, 4}), v1);
    v1.push_back(4);
    EXPECT_EQ((short_vector<int, 4>{2, 3, 1, 4, 4}), v1);
    v1.pop_back();
    EXPECT_EQ((short_vector<int, 4>{2, 3, 1, 4}), v1);
    v1.pop_back();
    EXPECT_EQ((short_vector<int, 4>{2, 3, 1}), v1);
    v1.pop_back();
    EXPECT_EQ((short_vector<int, 4>{2, 3}), v1);
    v1.pop_back();
    EXPECT_EQ((short_vector<int, 4>{2}), v1);
}

TEST(short_vector, resize)
{
    short_vector<int, 4> v1;

    v1.resize(2);
    EXPECT_EQ((short_vector<int, 4>{0, 0}), v1);
    v1.resize(5, 1);
    EXPECT_EQ((short_vector<int, 4>{0, 0, 1, 1, 1}), v1);
    v1.resize(3);
    EXPECT_EQ((short_vector<int, 4>{0, 0, 1}), v1);
    v1.resize(6, 4);
    EXPECT_EQ((short_vector<int, 4>{0, 0, 1, 4, 4, 4}), v1);
    v1.resize(1, 8);
    EXPECT_EQ((short_vector<int, 4>{0}), v1);
}

TEST(short_vector, swap)
{
    short_vector<int, 4> v1{1, 2, 3};
    short_vector<int, 4> v2{9, 2};
    short_vector<int, 4> v3{0, 5, 9, 4, 5};
    short_vector<int, 4> v4{6, 5, 4, 3, 2, 1, 0};

    v1.swap(v2);
    v3.swap(v4);
    EXPECT_EQ((short_vector<int, 4>{1, 2, 3}), v2);
    EXPECT_EQ((short_vector<int, 4>{9, 2}), v1);
    EXPECT_EQ((short_vector<int, 4>{0, 5, 9, 4, 5}), v4);
    EXPECT_EQ((short_vector<int, 4>{6, 5, 4, 3, 2, 1, 0}), v3);
    swap(v1, v3);
    swap(v4, v2);
    EXPECT_EQ((short_vector<int, 4>{1, 2, 3}), v4);
    EXPECT_EQ((short_vector<int, 4>{9, 2}), v3);
    EXPECT_EQ((short_vector<int, 4>{0, 5, 9, 4, 5}), v2);
    EXPECT_EQ((short_vector<int, 4>{6, 5, 4, 3, 2, 1, 0}), v1);
}

TEST(short_vector, operators)
{
    short_vector<int, 4> v1{0, 1};
    short_vector<int, 4> v2{0, 2, 4};

    EXPECT_TRUE(v1 == v1);
    EXPECT_TRUE(v2 == v2);
    EXPECT_TRUE(v1 != v2);
    EXPECT_TRUE(v2 > v1);
    EXPECT_TRUE(v1 >= v1);
    EXPECT_TRUE(v1 <= v1);
    EXPECT_TRUE(v1 <= v2);
    EXPECT_TRUE(v1 < v2);
    EXPECT_FALSE(v2 < v1);
    EXPECT_FALSE(v1 < v1);
    EXPECT_FALSE(v1 >= v2);
    EXPECT_FALSE(v2 <= v1);
    EXPECT_FALSE(v1 > v2);
}
