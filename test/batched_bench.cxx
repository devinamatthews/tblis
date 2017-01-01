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
#include "internal/3t/mult.hpp"
#include "tensor/tblis_batched_tensor.hpp"
#include "tensor/tblis_batched_tensor_contract.hpp"

int dumb = 0;
int check = 0;

using namespace std;
using namespace tblis;
using namespace stl_ext;

std::atomic<long> flops;

len_type v = 30;
len_type o = 5;

namespace tblis
{

extern len_type inout_ratio;
extern int outer_threading;

}

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
void init(batched_tensor<T>& A, const string& dense, const string& batch)
{
    unsigned n = dense.size();
    unsigned m = batch.size();

    vector<len_type> len;
    for (auto c : dense+batch)
        len.push_back(tolower(c) >= 'a' && tolower(c) <= 'h' ? v :
                      tolower(c) >= 'i' && tolower(c) <= 'p' ? o : len.back());

    len_type size = 1;
    for (unsigned i = 0;i < m;)
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

    vector<len_type> cur_idx(m);
    for (unsigned i = 0;i < m;i++)
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

    matrix<len_type> idx({size, m}, 0, ROW_MAJOR);

    if (m > 0 && size > 0)
    {
        len_type off = 0;
        for (bool done = false;!done;)
        {
            for (unsigned i = 0;i < m;i++) idx[off][i] = cur_idx[i];
            off++;

            for (unsigned i = m;i --> 0;)
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

    A.reset(len, idx);

    if (check)
    {
        len.resize(n);

        for (len_type i = 0;i < size;i++)
        {
            double* data = A.batch_data(i);
            MArray::viterator<> it(len, A.strides());
            while (it.next(data)) *data = random_number<double>();
        }
    }
}

template <typename T>
double diff(const_batched_tensor_view<T> A,
            const_batched_tensor_view<T> B)
{
    double d = 0.0;

    std::vector<len_type> dense_len(A.lengths());
    dense_len.resize(A.dense_dimension());

    for (len_type i = 0;i < A.num_batches();i++)
    {
        const T* a = A.batch_data(i);
        const T* b = B.batch_data(i);

        stride_type off = 0;
        MArray::viterator<> it(dense_len, A.strides());
        while (it.next(off)) d += norm2(a[off]-b[off]);
    }

    return sqrt(d);
}

template <typename T>
void bench(int R,
           T alpha, const batched_tensor<T>& A, const std::string& typea,
                    const batched_tensor<T>& B, const std::string& typeb,
           T  beta, const batched_tensor<T>& C, const std::string& typec)
{
    batched_tensor<double> tmp0_, tmp1_, tmp2_, tmp3_;
    batched_tensor_view<T> tmp0, tmp1, tmp2, tmp3;

    if (check)
    {
        tmp0_.reset(C);
        tmp1_.reset(C);
        tmp2_.reset(C);
        tmp3_.reset(C);
        tmp0.reset(tmp0_);
        tmp1.reset(tmp1_);
        tmp2.reset(tmp2_);
        tmp3.reset(tmp3_);
    }
    else
    {
        tmp0.reset(const_cast<batched_tensor<T>&>(C));
        tmp1.reset(const_cast<batched_tensor<T>&>(C));
        tmp2.reset(const_cast<batched_tensor<T>&>(C));
        tmp3.reset(const_cast<batched_tensor<T>&>(C));
    }

    if (dumb)
    {
        contract_batch_dumb<double>(alpha,    A, typea.data(),
                                              B, typeb.data(),
                                     beta, tmp0, typec.data());
    }

    flops = 0;

    double t1 = run_kernel(R,
    [&]
    {
        contract_batch_ref<double>(alpha,    A, typea.data(),
                                             B, typeb.data(),
                                    beta, tmp1, typec.data());
    });

    auto flops1 = flops.load();
    printf("%ld\n", flops1);
    flops = 0;

    double t2 = run_kernel(R,
    [&]
    {
        contract_batch<double>(alpha,    A, typea.data(),
                                         B, typeb.data(),
                                beta, tmp2, typec.data());
    });

    auto flops2 = flops.load();
    printf("%ld\n", flops2);
    flops = 0;

    double t3 = run_kernel(R,
    [&]
    {
        contract_batch2<double>(alpha,    A, typea.data(),
                                          B, typeb.data(),
                                 beta, tmp3, typec.data());
    });

    auto flops3 = flops.load();
    printf("%ld\n", flops3);

    if (check)
    {
        double d1 = diff(dumb ? tmp0 : tmp1, tmp1);
        double d2 = diff(dumb ? tmp0 : tmp1, tmp2);
        double d3 = diff(dumb ? tmp0 : tmp1, tmp3);
        printf("%g %g %g\n", d1, d2, d3);
    }

    printf("%g %g %g\n", t1, t2, t3);
    printf("%g %g %g\n", flops1/t1/1e9/R, flops2/t2/1e9/R, flops3/t3/1e9/R);
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
                            {"outer-threading", no_argument, &outer_threading, 1},
                            {"no-outer-threading", no_argument, &outer_threading, 0},
                            {"inner-threading", no_argument, &outer_threading, 0},
                            {"no-inner-threading", no_argument, &outer_threading, 1},
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

    constexpr bool test0 = false;
    constexpr bool test1 = false;
    constexpr bool test2 = false;
    constexpr bool test3 = true;

    if (test0)
    {
        batched_tensor<double> T4;
        batched_tensor<double> T3;
        batched_tensor<double> Wa;

        init(T4, "ABCD", "I===");
        init(T3,  "ABC",  "I==");
        init(Wa,  "ABC",  "I=K");

        bench(R, 1.0, T3,   "ABEIJM",
                      Wa,   "CDEKLM",
                 1.0, T4, "ABCDIJKL");
    }

    if (test1)
    {
        batched_tensor<double> T2;
        batched_tensor<double> W;
        batched_tensor<double> T3;

        init(T2, "ABIJ", "");
        init(W, "IJKA", "");
        init(T3, "ABC", "I==");

        bench(R, 1.0, T2,   "ABIM",
                       W,   "JKMC",
                 1.0, T3, "ABCIJK");
    }

    if (test2)
    {
        batched_tensor<double> T2;
        batched_tensor<double> W;
        batched_tensor<double> T3;

        init(T2, "ABIJ", "");
        init(W, "ABCI", "");
        init(T3, "ABC", "I==");

        bench(R, 1.0, T2,   "AEIJ",
                       W,   "BCEK",
                 1.0, T3, "ABCIJK");
    }

    if (test3)
    {
        batched_tensor<double> T4;
        batched_tensor<double> T3;
        batched_tensor<double> W;

        init(T4, "ABCD", "I===");
        init(T3,  "ABC",  "I==");
        init(W,  "IJKA",  "");

        bench(R, 1.0, T3,   "ABCIJM",
                       W,     "KLMD",
                 1.0, T4, "ABCDIJKL");
    }

    return 0;
}
