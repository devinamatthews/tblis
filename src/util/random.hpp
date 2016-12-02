#ifndef _TBLIS_RANDOM_HPP_
#define _TBLIS_RANDOM_HPP_

#include <random>

#include "assert.h"
#include "basic_types.h"

namespace tblis
{

extern std::mt19937 rand_engine;

/*
 * Returns a pseudo-random number uniformly distributed in the range [mn,mx).
 */
template <typename T>
enable_if_floating_point_t<real_type_t<T>,T> random_number(T mn, T mx)
{
    std::uniform_real_distribution<real_type_t<T>> d(mn, mx);
    return d(rand_engine);
}

/*
 * Returns a pseudo-random number uniformly distributed in the range [0,mx).
 */
template <typename T>
enable_if_floating_point_t<real_type_t<T>,T> random_number(T mx)
{
    return random_number<T>(0, mx);
}

/*
 * Returns a pseudo-random number uniformly distributed in the range [0,1).
 */
template <typename T>
enable_if_floating_point_t<real_type_t<T>,T> random_number()
{
    return random_number<T>(0, 1);
}

/*
 * Returns a random integer uniformly distributed in the range [mn,mx]
 */
template <typename T>
enable_if_integral_t<T,T> random_number(T mn, T mx)
{
    std::uniform_int_distribution<T> d(mn, mx);
    return d(rand_engine);
}

/*
 * Returns a random integer uniformly distributed in the range [0,mx]
 */
template <typename T>
enable_if_integral_t<T,T> random_number(T mx)
{
    return random_number<T>(0, mx);
}

/*
 * Returns a pseudo-random number uniformly distributed in the range (-1,1).
 */
template <typename T>
enable_if_floating_point_t<T,T> random_unit()
{
    T val;
    do
    {
        val = 2*random_number<T>()-1;
    } while (val == -1.0);
    return val;
}

/*
 * Returns a psuedo-random complex number unformly distirbuted in the
 * interior of the unit circle.
 */
template <typename T>
enable_if_complex_t<T,T> random_unit()
{
    using R = real_type_t<T>;

    R r, i;
    do
    {
        r = random_unit<R>();
        i = random_unit<R>();
    }
    while (r*r+i*i >= 1.0);

    return {r, i};
}

inline
bool random_choice()
{
    return random_number(1);
}

/*
 * Returns a random choice from a set of objects with non-negative weights w,
 * which do not need to sum to unity.
 */
template <typename Weights>
int random_weighted_choice(const Weights& w)
{
    using T = typename Weights::value_type;

    auto n = w.size();
    TBLIS_ASSERT(n > 0);

    T s = 0;
    for (unsigned i = 0;i < n;i++)
    {
        TBLIS_ASSERT(w[i] >= 0);
        s += w[i];
    }

    T c = random_number(s);
    for (unsigned i = 0;i < n;i++)
    {
        if (c < w[i]) return i;
        c -= w[i];
    }

    return n-1;
}

/*
 * Returns a sequence of n non-negative numbers such that sum_i n_i = s and
 * and n_i >= mn_i, with uniform distribution.
 */
template <typename T>
std::vector<T> random_sum_constrained_sequence(unsigned n, T s, const std::vector<T>& mn)
{
    TBLIS_ASSERT(n > 0);
    TBLIS_ASSERT(s >= 0);
    TBLIS_ASSERT(mn.size() == n);

    for (unsigned i = 0;i < n;i++)
    {
        TBLIS_ASSERT(mn[i] >= 0);
        s -= mn[i];
        TBLIS_ASSERT(s >= 0);
    }

    std::vector<T> p(n+1);

    p[0] = 0;
    p[n] = s;
    for (unsigned i = 1;i < n;i++)
    {
        p[i] = random_number<T>(s);
    }
    std::sort(p.begin(), p.end());

    for (unsigned i = 0;i < n;i++)
    {
        p[i] = p[i+1]-p[i]+mn[i];
    }
    p.resize(n);
    //cout << s << p << accumulate(p.begin(), p.end(), 0.0) << endl;

    return p;
}

/*
 * Returns a sequence of n non-negative numbers such that sum_i n_i = s,
 * with uniform distribution.
 */
template <typename T>
std::vector<T> random_sum_constrained_sequence(unsigned n, T s)
{
    TBLIS_ASSERT(n > 0);
    return random_sum_constrained_sequence(n, s, std::vector<T>(n));
}

/*
 * Returns a sequence of n numbers such than prod_i n_i = p and n_i >= mn_i,
 * where n_i and p are >= 1 and with uniform distribution.
 */
template <typename T>
enable_if_floating_point_t<T,std::vector<T>>
random_product_constrained_sequence(unsigned n, T p, const std::vector<T>& mn)
{
    TBLIS_ASSERT(n >  0);
    TBLIS_ASSERT(p >= 1);
    TBLIS_ASSERT(mn.size() == n);

    std::vector<T> log_mn(n);
    for (unsigned i = 0;i < n;i++)
    {
        log_mn[i] = (mn[i] <= 0.0 ? 1.0 : std::log(mn[i]));
    }

    std::vector<T> s = random_sum_constrained_sequence<T>(n, std::log(p), log_mn);
    for (unsigned i = 0;i < n;i++) s[i] = std::exp(s[i]);
    //cout << p << s << accumulate(s.begin(), s.end(), 1.0, multiplies<double>()) << endl;
    return s;
}

/*
 * Returns a sequence of n numbers such than prod_i n_i = p, where n_i and
 * p are >= 1 and with uniform distribution.
 */
template <typename T>
enable_if_floating_point_t<T,std::vector<T>>
random_product_constrained_sequence(unsigned n, T p)
{
    TBLIS_ASSERT(n > 0);
    return random_product_constrained_sequence(n, p, std::vector<T>(n, 1.0));
}

enum rounding_mode {ROUND_UP, ROUND_DOWN, ROUND_NEAREST};

/*
 * Returns a sequence of n numbers such that p/2^d <= prod_i n_i <= p and
 * n_i >= mn_i, where n_i and p are >= 1 and with uniform distribution.
 */
template <typename T, rounding_mode Mode=ROUND_DOWN>
enable_if_integral_t<T,std::vector<T>>
random_product_constrained_sequence(unsigned n, T p, const std::vector<T>& mn)
{
    TBLIS_ASSERT(n >  0);
    TBLIS_ASSERT(p >= 1);
    TBLIS_ASSERT(mn.size() == n);

    std::vector<double> mnd(n);
    for (unsigned i = 0;i < n;i++)
    {
        mnd[i] = std::max(T(1), mn[i]);
    }

    std::vector<double> sd = random_product_constrained_sequence<double>(n, p, mnd);
    std::vector<T> si(n);
    for (unsigned i = 0;i < n;i++)
    {
        switch (Mode)
        {
            case      ROUND_UP: si[i] = lrint( ceil(sd[i])); break;
            case    ROUND_DOWN: si[i] = lrint(floor(sd[i])); break;
            case ROUND_NEAREST: si[i] = lrint(      sd[i] ); break;
        }
        si[i] = std::max(si[i], mn[i]);
    }
    //cout << p << si << accumulate(si.begin(), si.end(), T(1), multiplies<T>()) << endl;

    return si;
}

/*
 * Returns a sequence of n numbers such that p/2^d <= prod_i n_i <= p, where
 * n_i and p are >= 1 and with uniform distribution.
 */
template <typename T, rounding_mode Mode=ROUND_DOWN>
enable_if_integral_t<T,std::vector<T>>
random_product_constrained_sequence(unsigned n, T p)
{
    TBLIS_ASSERT(n > 0);
    return random_product_constrained_sequence<T, Mode>(n, p, std::vector<T>(n, T(1)));
}

}

#endif
