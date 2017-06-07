#include "gtest/gtest.h"
#include "viterator.hpp"

using namespace std;
using namespace MArray;

TEST(viterator, next)
{
    len_type off1, off2;

    viterator<> m1(vector<int>{}, vector<int>{});

    off1 = 0;
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_FALSE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);

    viterator<2> m2(vector<int>{5}, vector<int>{1}, vector<int>{2});

    off1 = 0;
    off2 = 0;
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(1, off1);
    EXPECT_EQ(2, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(2, off1);
    EXPECT_EQ(4, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(3, off1);
    EXPECT_EQ(6, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(4, off1);
    EXPECT_EQ(8, off2);
    EXPECT_FALSE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(1, off1);
    EXPECT_EQ(2, off2);

    viterator<> m3(vector<int>{2,3}, vector<int>{1,2});

    off1 = 0;
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(1, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(2, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(5, off1);
    EXPECT_FALSE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(1, off1);

    viterator<> m4(vector<int>{0}, vector<int>{1});

    off1 = 0;
    EXPECT_FALSE(m4.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_FALSE(m4.next(off1));
    EXPECT_EQ(0, off1);
}

TEST(viterator, reset)
{
    len_type off1;

    viterator<> m1(vector<int>{2,3}, vector<int>{1,2});

    off1 = 0;
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(1, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(2, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(3, off1);

    off1 = 0;
    m1.reset();
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(1, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(2, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(5, off1);

    viterator<> m4(vector<int>{0}, vector<int>{1});

    off1 = 0;
    EXPECT_FALSE(m4.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_FALSE(m4.next(off1));
    EXPECT_EQ(0, off1);
}

TEST(viterator, position)
{
    len_type off1;

    viterator<> m1(vector<int>{2,3}, vector<int>{1,2});

    off1 = 0;
    m1.position(3, off1);
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(5, off1);

    off1 = 0;
    m1.position(vector<int>{1,1}, off1);
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(5, off1);

    EXPECT_EQ(1, m1.position(0));
    EXPECT_EQ(2, m1.position(1));
    EXPECT_EQ((vector<len_type>{1,2}), m1.position());
}

TEST(viterator, assign)
{
    len_type off1;

    viterator<> m1(vector<int>{2,3}, vector<int>{1,2});
    viterator<> m2(vector<int>{2,3}, vector<int>{1,3});

    off1 = 0;
    m1.position(vector<int>{1,1}, off1);
    EXPECT_EQ(3, off1);

    off1 = 0;
    m2.position(vector<int>{1,1}, off1);
    EXPECT_EQ(4, off1);

    m1 = m2;

    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(6, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(7, off1);
}

TEST(viterator, swap)
{
    len_type off1, off2;

    viterator<> m1(vector<int>{2,3}, vector<int>{1,2});
    viterator<> m2(vector<int>{2,3}, vector<int>{1,3});

    off1 = 0;
    m1.position(vector<int>{1,1}, off1);
    EXPECT_EQ(3, off1);

    off2 = 0;
    m2.position(vector<int>{1,1}, off2);
    EXPECT_EQ(4, off2);

    m1.swap(m2);

    EXPECT_TRUE(m1.next(off2));
    EXPECT_EQ(4, off2);
    EXPECT_TRUE(m1.next(off2));
    EXPECT_EQ(6, off2);
    EXPECT_TRUE(m1.next(off2));
    EXPECT_EQ(7, off2);

    EXPECT_TRUE(m2.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m2.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m2.next(off1));
    EXPECT_EQ(5, off1);
}

TEST(viterator, make_iterator)
{
    len_type off1, off2;

    auto m1 = make_iterator(vector<len_type>{}, vector<stride_type>{});

    off1 = 0;
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_FALSE(m1.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m1.next(off1));
    EXPECT_EQ(0, off1);

    auto m2 = make_iterator(vector<len_type>{5}, vector<stride_type>{1}, vector<stride_type>{2});

    off1 = 0;
    off2 = 0;
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(1, off1);
    EXPECT_EQ(2, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(2, off1);
    EXPECT_EQ(4, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(3, off1);
    EXPECT_EQ(6, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(4, off1);
    EXPECT_EQ(8, off2);
    EXPECT_FALSE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(0, off1);
    EXPECT_EQ(0, off2);
    EXPECT_TRUE(m2.next(off1, off2));
    EXPECT_EQ(1, off1);
    EXPECT_EQ(2, off2);

    auto m3 = make_iterator(vector<len_type>{2,3}, vector<stride_type>{1,2});

    off1 = 0;
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(1, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(2, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(3, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(4, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(5, off1);
    EXPECT_FALSE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(0, off1);
    EXPECT_TRUE(m3.next(off1));
    EXPECT_EQ(1, off1);
}

#if 0
template <unsigned N=1>
class viterator
{
    public:
        typedef unsigned len_type;
        typedef ptrdiff_t stride_type;

        viterator() {}

        viterator(const viterator&) = default;

        viterator(viterator&&) = default;

        template <typename Len, typename... Strides,
                  typename=typename std::enable_if<detail::is_container<Len>::value &&
                                                   detail::are_containers<Strides...>::value &&
                                                   sizeof...(Strides) == N>::type>
        viterator(const Len& len, const Strides&... strides)
        : ndim_(len.size()), pos_(len.size()), len_(len.size()), first_(true), empty_(false)
        {
            for (unsigned i = 0;i < ndim_;i++) if (len[i] == 0) empty_ = true;
            std::copy_n(len.begin(), ndim_, len_.begin());
            detail::set_strides(strides_, strides...);
        }

        viterator& operator=(const viterator&) = default;

        viterator& operator=(viterator&&) = default;

        void reset()
        {
            pos_.assign(ndim_, 0);
            first_ = true;
        }

        template <typename... Offsets,
                  typename=typename std::enable_if<sizeof...(Offsets) == N>::type>
        bool next(Offsets&... off)
        {
            if (empty_) return false;

            if (first_)
            {
                first_ = false;
                return true;
            }

            if (ndim_ == 0)
            {
                first_ = true;
                return false;
            }

            for (unsigned i = 0;i < ndim_;i++)
            {
                if (pos_[i] == len_[i]-1)
                {
                    detail::dec_offsets(i, pos_, strides_, off...);
                    pos_[i] = 0;

                    if (i == ndim_-1)
                    {
                        first_ = true;
                        return false;
                    }
                }
                else
                {
                    detail::inc_offsets(i, strides_, off...);
                    pos_[i]++;
                    return true;
                }
            }

            return true;
        }

        template <typename... Offsets,
                  typename=typename std::enable_if<sizeof...(Offsets) == N>::type>
        void position(stride_type pos, Offsets&... off)
        {
            for (size_t i = 0;i < ndim_;i++)
            {
                pos_[i] = pos%len_[i];
                pos = pos/len_[i];
            }
            assert(pos == 0);

            position(pos_, off...);
        }

        template <typename Pos, typename... Offsets,
                  typename=typename std::enable_if<detail::is_container_of<Pos, len_type>::value &&
                                                   sizeof...(Offsets) == N>::type>
        void position(const Pos& pos, Offsets&... off)
        {
            assert(pos.size() == ndim_);

            pos_ = pos;

            for (size_t i = 0;i < ndim_;i++)
            {
                assert(pos_[i] >= 0 && pos_[i] < len_[i]);
            }

            detail::move_offsets(pos_, strides_, off...);

            first_ = true;
        }

        unsigned dimension() const
        {
            return ndim_;
        }

        len_type position(unsigned dim) const
        {
            return pos_[dim];
        }

        const std::vector<len_type>& position() const
        {
            return pos_;
        }

        len_type length(unsigned dim) const
        {
            return len_[dim];
        }

        const std::vector<len_type>& lengths() const
        {
            return len_;
        }

        stride_type stride(unsigned i, unsigned dim) const
        {
            return strides_[i][dim];
        }

        const std::vector<stride_type>& strides(unsigned i) const
        {
            return strides_[i];
        }

        friend void swap(viterator& i1, viterator& i2)
        {
            using std::swap;
            swap(i1.ndim_, i2.ndim_);
            swap(i1.pos_, i2.pos_);
            swap(i1.len_, i2.len_);
            swap(i1.strides_, i2.strides_);
            swap(i1.first_, i2.first_);
            swap(i1.empty_, i2.empty_);
        }

    private:
        size_t ndim_ = 0;
        std::vector<len_type> pos_;
        std::vector<len_type> len_;
        std::array<std::vector<stride_type>,N> strides_;
        bool first_ = true;
        bool empty_ = true;
};

template <typename len_type, typename stride_type, typename... Strides,
          typename=typename std::enable_if<detail::are_containers<Strides...>::value>::type>
viterator<1+sizeof...(Strides)>
make_iterator(const std::vector<len_type>& len,
              const std::vector<stride_type>& stride0,
              const Strides&... strides)
{
    return {len, stride0, strides...};
}

}

#endif
