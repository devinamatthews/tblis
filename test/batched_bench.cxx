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
#include <atomic>

#include "tblis.h"
#include "util/time.hpp"
#include "util/tensor.hpp"
#include "util/random.hpp"
#include "internal/3t/dense/mult.hpp"

int dumb = 0;
int check = 0;

using namespace std;
using namespace tblis;
using namespace stl_ext;

len_type v = 30;
len_type o = 5;

template <typename Kernel, typename ...Args>
double run_kernel(len_type R, Kernel&& kernel, Args&&...args)
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

template <typename T>
void init(indexed_marray<T>& A, const string& dense, const string& batch)
{
    auto n = dense.size();
    auto m = batch.size();

    len_vector len;
    for (auto c : dense+batch)
        len.push_back(tolower(c) >= 'a' && tolower(c) <= 'h' ? v :
                      tolower(c) >= 'i' && tolower(c) <= 'p' ? o : len.back());

    len_type size = 1;
    auto i = 0;
    while (i < m)
    {
        int j = i+1;
        while (j < m && batch[j] == '=') j++;

        len_type s = len[n+i];

        switch (j-i)
        {
            case 1: size *= s; break;
            case 2: size *= s*(s+1)/2; break;
            case 3: size *= s*(s+1)*(s+2)/6-s; break;
            case 4: size *= s*(s+1)*(s+2)*(s+3)/24-s*s; break;
        }

        i = j;
    }

    len_vector cur_idx(m);
    for (auto i : range(m))
    {
        if (i == 0 || batch[i] != '=')
        {
            cur_idx[i] = 0;
        }
        else if (i == 1 || batch[i-1] != '=' || cur_idx[i-2] != cur_idx[i-1])
        {
            cur_idx[i] = cur_idx[i-1];
        }
        else
        {
            cur_idx[i] = cur_idx[i-1]+1;
        }
    }

    matrix<len_type> idx({size, (len_type)m}, 0, ROW_MAJOR);

    if (m > 0 && size > 0)
    {
        len_type off = 0;
        for (bool done = false;!done;)
        {
            for (auto i : range(m)) idx[off][i] = cur_idx[i];
            off++;

            for (auto i : reversed_range(m))
            {
                bool over = (i == m-1 || batch[i+1] != '=')

                                ? (cur_idx[i] >= len[n+i]-1) :

                            (i == m-2 || batch[i+2] != '=' ||
                             cur_idx[i+1] != cur_idx[i+2])

                                ? (cur_idx[i] >= cur_idx[i+1])

                                : (cur_idx[i] >= cur_idx[i+1]-1);

                if (over)
                {
                    if (i == 0) done = true;
                }
                else
                {
                    cur_idx[i]++;

                    for (i++;i < m;i++)
                    {
                        if (i == 0 || batch[i] != '=')
                        {
                            cur_idx[i] = 0;
                        }
                        else if (i == 1 || batch[i-1] != '=' || cur_idx[i-2] != cur_idx[i-1])
                        {
                            cur_idx[i] = cur_idx[i-1];
                        }
                        else
                        {
                            cur_idx[i] = cur_idx[i-1]+1;
                        }
                    }

                    break;
                }
            }
        }

        assert(off == size);
    }

    A.reset(len, idx.view());

    if (check)
    {
        len.resize(n);

        for (auto i : range(size))
        {
            double* data = A.data(i);
            viterator<> it(len, A.dense_strides());
            while (it.next(data)) *data = random_number<double>();
        }
    }
}

template <typename T>
double diff(const indexed_marray_view<T>& A,
            const indexed_marray_view<T>& B)
{
    double d = 0.0;

    viterator<> it(A.dense_lengths(), A.dense_strides());

    for (len_type i = 0;i < A.num_indices();i++)
    {
        const T* a = A.data(i);
        const T* b = B.data(i);

        stride_type off = 0;
        while (it.next(off)) d += norm2(a[off]-b[off]);
    }

    return sqrt(d);
}

template <typename T>
void bench(int R,
           T alpha, indexed_marray_view<const T> A, const std::string& typea,
                    indexed_marray_view<const T> B, const std::string& typeb,
           T  beta, indexed_marray_view<      T> C, const std::string& typec)
{
    flops = 0;

    double t1 = run_kernel(R,
    [&]
    {
        mult<double>(alpha, A, typea.data(),
                            B, typeb.data(),
                      beta, C, typec.data());
    });

    long flops1 = flops.load()/R;

    printf("%ld %g %g\n", flops1, t1, flops1/t1/1e9);
}

int main(int argc, char** argv)
{
    int R = 5;
    time_t seed = time(nullptr);

    struct option opts[] = {{"rep", required_argument, NULL, 'r'},
                            {"seed", required_argument, NULL, 's'},
                            {"check", no_argument, &check, 1},
                            {"no-check", no_argument, &check, 0},
                            {"dumb", no_argument, &dumb, 1},
                            {"no-dumb", no_argument, &dumb, 0},
                            {"inout-ratio", required_argument, NULL, 'i'},
                            {"occ", required_argument, NULL, 'o'},
                            {"vrt", required_argument, NULL, 'v'},
                            {0, 0, 0, 0}};

    while (true)
    {
        istringstream iss;
        int arg = getopt_long(argc, argv, "r:s:v:o:i:", opts, NULL);

        if (arg == -1) break;

        switch (arg)
        {
            case 'i':
                iss.str(optarg);
                iss >> inout_ratio;
                break;
            case 'v':
                iss.str(optarg);
                iss >> v;
                break;
            case 'o':
                iss.str(optarg);
                iss >> o;
                break;
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

    constexpr bool test0 = true;
    constexpr bool test1 = false;
    constexpr bool test2 = false;
    constexpr bool test3 = true;
    constexpr bool test4 = true;
    constexpr bool test5 = true;

    if (test0)
    {
        indexed_marray<double> T4;
        indexed_marray<double> T3;
        indexed_marray<double> Wa;

        init(T4, "ABCD", "I===");
        init(T3,  "ABC",  "I==");
        init(Wa,  "ABC",  "I=K");

        bench<double>(R, 1.0, T3,   "ABEIJM",
                              Wa,   "CDEKLM",
                         1.0, T4, "ABCDIJKL");
    }

    if (test1)
    {
        indexed_marray<double> T2;
        indexed_marray<double> W;
        indexed_marray<double> T3;

        init(T2, "ABIJ", "");
        init(W, "IJKA", "");
        init(T3, "ABC", "I==");

        bench<double>(R, 1.0, T2,   "ABIM",
                               W,   "JKMC",
                         1.0, T3, "ABCIJK");
    }

    if (test2)
    {
        indexed_marray<double> T2;
        indexed_marray<double> W;
        indexed_marray<double> T3;

        init(T2, "ABIJ", "");
        init(W, "ABCI", "");
        init(T3, "ABC", "I==");

        bench<double>(R, 1.0, T2,   "AEIJ",
                               W,   "BCEK",
                         1.0, T3, "ABCIJK");
    }

    if (test3)
    {
        indexed_marray<double> T4;
        indexed_marray<double> Z4;
        indexed_marray<double> Wa;

        init(T4, "ABCD", "I===");
        init(Z4, "ABCD", "I===");
        init(Wa, "AIBJ", "");

        bench<double>(R, 1.0, T4, "ABCEIJKM",
                              Wa,     "DMEL",
                         1.0, Z4, "ABCDIJKL");
    }

    if (test4)
    {
        indexed_marray<double> T4;
        indexed_marray<double> T3;
        indexed_marray<double> W;

        init(T4, "ABCD", "I===");
        init(T3,  "ABC",  "I==");
        init(W,  "IJKA",  "");

        bench<double>(R, 1.0, T3,   "ABCIJM",
                               W,     "KLMD",
                         1.0, T4, "ABCDIJKL");
    }

    if (test5)
    {
        indexed_marray<double> T3;
        indexed_marray<double> Z4;
        indexed_marray<double> W;

        init(T3, "ABC", "I==");
        init(Z4, "ABCD", "I===");
        init(W, "ABCI", "");

        bench<double>(R, 1.0, T3,   "ABEIJK",
                               W,     "CDEL",
                         1.0, Z4, "ABCDIJKL");
    }

    return 0;
}
