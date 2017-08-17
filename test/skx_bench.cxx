#include <cstdlib>
#include <algorithm>
#include <limits>
#include <stdint.h>
#include <iostream>
#include <random>
#include <numeric>
#include <getopt.h>
#include <sstream>
#include <type_traits>
#include <iomanip>
#include <functional>
#include <set>
#include <map>

#include "tblis.h"
#include "util/time.hpp"
#include "util/tensor.hpp"
#include "util/random.hpp"
#include "configs/skx/config.hpp"

using namespace std;
using namespace tblis;
using namespace stl_ext;

const config* configs[] =
{
    //&skx_32x6_l1_config::instance(),
    //&skx_32x6_l2_config::instance(),
    //&skx_24x8_l1_config::instance(),
    //&skx_24x8_l2_config::instance(),
    //&skx_16x12_l1_config::instance(),
    //&skx_16x12_l2_config::instance(),
    //&skx_12x16_l1_config::instance(),
    //&skx_12x16_l2_config::instance(),
    //&skx_8x24_l1_config::instance(),
    //&skx_8x24_l2_config::instance(),
    //&skx_6x32_l1_config::instance(),
    //&skx_6x32_l2_config::instance(),
    &skx_knl_config::instance(),
};
constexpr auto num_configs = sizeof(configs)/sizeof(configs[0]);

template <typename T>
void gemm_ref(T alpha, matrix_view<const T> A,
                       matrix_view<const T> B,
              T  beta,       matrix_view<T> C)
{
    const T* ptr_A = A.data();
    const T* ptr_B = B.data();
          T* ptr_C = C.data();

    len_type m_A = A.length(0);
    len_type m_C = C.length(0);
    len_type n_B = B.length(1);
    len_type n_C = C.length(1);
    len_type k_A = A.length(1);
    len_type k_B = B.length(0);

    stride_type rs_A = A.stride(0);
    stride_type cs_A = A.stride(1);
    stride_type rs_B = B.stride(0);
    stride_type cs_B = B.stride(1);
    stride_type rs_C = C.stride(0);
    stride_type cs_C = C.stride(1);

    TBLIS_ASSERT(m_A == m_C);
    TBLIS_ASSERT(n_B == n_C);
    TBLIS_ASSERT(k_A == k_B);

    len_type m = m_A;
    len_type n = n_B;
    len_type k = k_A;

    for (len_type i = 0;i < m;i++)
    {
        for (len_type j = 0;j < n;j++)
        {
            T tmp = T();

            if (alpha != T(0))
            {
                for (len_type ik = 0;ik < k;ik++)
                {
                    tmp += ptr_A[i*rs_A + ik*cs_A]*ptr_B[ik*rs_B + j*cs_B];
                }
            }

            if (beta == T(0))
            {
                ptr_C[i*rs_C + j*cs_C] = alpha*tmp;
            }
            else
            {
                ptr_C[i*rs_C + j*cs_C] = alpha*tmp + beta*ptr_C[i*rs_C + j*cs_C];
            }
        }
    }
}

range_t<stride_type> parse_range(const string & s)
{
    stride_type mn, mx;
    stride_type delta = 1;

    size_t colon1 = s.find(':');
    size_t colon2 = s.find(':', colon1 == string::npos ? colon1 : colon1+1);

    if (colon1 == string::npos)
    {
        mn = mx = stol(s);
    }
    else if (colon2 == string::npos)
    {
        mn = stol(s.substr(0, colon1));
        mx = stol(s.substr(colon1+1));
    }
    else
    {
        mn = stol(s.substr(0, colon1));
        mx = stol(s.substr(colon1+1, colon2-colon1-1));
        delta = stol(s.substr(colon2+1));
    }

    return
    {
        mn, mx+delta, delta
    };
}

template<typename Kernel, typename ...Args>
double run_kernel(len_type R, const Kernel & kernel, Args &&...args)
{
    double bias = numeric_limits<double>::max();
    for (len_type r = 0;r < R;r++)
    {
        double t0 = tic();
        double t1 = tic();
        bias = min(bias, t1-t0);
    }

    double dt = numeric_limits<double>::max();
    for (len_type r = 0;r < R;r++)
    {
        double t0 = tic();
        kernel(args...);
        double t1 = tic();
        dt = min(dt, t1-t0);
    }

    return dt - bias;
}

template<typename Experiment>
void iterate_over_ranges_helper(const Experiment & experiment,
                                const map<char,range_t<stride_type>> &ranges,
                                map<char,range_t<stride_type>>::const_iterator range,
                                map<char,len_type> &values)
{
    if (range == ranges.end())
    {
        len_type var = 0;

        for (auto & r : ranges)
        {
            if (r.second.size() > 1)
            {
                var = values[r.first];
            }
        }

        for (auto & r : ranges)
        {
            if (r.second.front() == -1)
            {
                values[r.first] = var;
            }
        }

        experiment(values);
    }
    else
    {
        //cout << range->second.front() << " " << range->second.back() << endl;
        for (stride_type v : range->second)
        {
            //cout << v << endl;
            values[range->first] = v;
            iterate_over_ranges_helper(experiment, ranges, next(range), values);
        }
    }
}

template<typename Experiment>
void iterate_over_ranges(const Experiment & experiment,
                         const map<char,range_t<stride_type>> &ranges)
{
    map<char, len_type> values;
    iterate_over_ranges_helper(experiment, ranges, ranges.begin(), values);
}

template<typename T>
struct gemm_experiment
{
    len_type R;

    gemm_experiment(len_type R, const range_t<stride_type> &m_range,
                    const range_t<stride_type> &n_range,
                    const range_t<stride_type> &k_range)
    : R(R)
    {
        iterate_over_ranges(*this, {{'m', m_range}, {'n', n_range}, {'k', k_range}});
    }

    void operator()(const map<char, len_type> &values) const
    {
        stride_type m = values.at('m');
        stride_type n = values.at('n');
        stride_type k = values.at('k');

        matrix<T> A({m, k});
        matrix<T> B({k, n});
        matrix<T> C({m, n});

        tblis_matrix At(A.view());
        tblis_matrix Bt(B.view());
        tblis_matrix Ct(C.view());

        double gflops = 2*m*n*k*1e-9;

        printf("%ld %ld %ld ", m, n, k);

        for (size_t i = 0;i < num_configs;i++)
        {
            double perf = gflops/run_kernel(R,
            [&]
            {
                tblis_matrix_mult(tblis_single, *configs[i], &At, &Bt, &Ct);
            });

            printf("%e ", perf);
        }

        printf("\n");
        fflush(stdout);
    }
};

template <typename T>
void test_gemm(len_type m, len_type n, len_type k)
{
    matrix<T> A({m, k});
    matrix<T> B({k, n});
    matrix<T> C({m, n});

    A.for_each_element([](T& e) { e = random_unit<T>(); });
    B.for_each_element([](T& e) { e = random_unit<T>(); });
    C.for_each_element([](T& e) { e = random_unit<T>(); });

    tblis_matrix At(A.view());
    tblis_matrix Bt(B.view());

    printf("%ld %ld %ld\n", m, n, k);

    for (size_t i = 0;i < num_configs;i++)
    {
        matrix<T> C_skx(C);
        matrix<T> C_ref(C);
        tblis_matrix Ct(C_skx.view());

        printf("%s: ", configs[i]->name);

        tblis_matrix_mult(tblis_single, *configs[i], &At, &Bt, &Ct);
        gemm_ref<T>(T(1), A, B, T(1), C_ref);
        add<T>(T(-1), C_ref, T(1), C_skx);
        double err = reduce<T>(REDUCE_NORM_2, C_skx).first;

        printf("%e\n", err/(m*n*k));
    }

    printf("\n");
}

int main(int argc, char** argv)
{
    int R = 10;
    time_t seed = time(nullptr);

    struct option opts[] = {{"rep", required_argument, NULL, 'r'},
                            {"seed", required_argument, NULL, 's'},
                            {0, 0, 0, 0}};

    int arg;
    int index;
    while ((arg = getopt_long(argc, argv, "r:s:", opts, &index)) != -1)
    {
        istringstream iss;
        switch (arg)
        {
            case 'r':
                iss.str(optarg);
                iss >> R;
                break;
            case 's':
                iss.str(optarg);
                iss >> seed;
                break;
            case '?':
                abort();
                break;
        }
    }

    cout << "Using mt19937 with seed " << seed << endl;
    rand_engine.seed(seed);

    printf("Testing SKX DGEMM:\n");
    test_gemm<double>(100, 100, 100);

    printf("Getting SKX DGEMM performance:\n");
    string line;
    while (getline(cin, line) && !line.empty())
    {
        if (line[0] == '#') continue;

        istringstream iss(line);

        string m_range, n_range, k_range;
        iss >> m_range >> n_range >> k_range;

        auto m = parse_range(m_range);
        auto n = parse_range(n_range);
        auto k = parse_range(k_range);

        gemm_experiment<double>(R, m, n, k);
    }

    return 0;
}
