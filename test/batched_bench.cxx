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

using namespace std;
using namespace tblis;
using namespace stl_ext;

template<typename Kernel, typename ...Args>
double run_kernel(len_type R, const Kernel& kernel, Args&&...args)
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

std::atomic<long> flops;

template <typename T>
double diff(const batched_tensor<T>& A, const batched_tensor<T>& B)
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

int main(int argc, char** argv)
{
    int R = 5;
    time_t seed = time(nullptr);

    struct option opts[] = {{"rep", required_argument, NULL, 'r'},
                            {"seed", required_argument, NULL, 's'},
                            {0, 0, 0, 0}};

    int arg;
    int index;
    istringstream iss;
    while ((arg = getopt_long(argc, argv, "r:s:", opts, &index)) != -1)
    {
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

    len_type v = 10;
    len_type o = 4;

    matrix<len_type> T4_idx({o*(o+1)*(o+2)*(o+3)/24 - o*o, 4}, 0, ROW_MAJOR);
    for (len_type i = 0, off = 0;i < o;i++)
    {
        for (len_type j = 0;j <= i;j++)
        {
            for (len_type k = 0;k <= j;k++)
            {
                if (i == j && j == k) continue;
                for (len_type l = 0;l <= k;l++)
                {
                    if (j == k && k == l) continue;
                    T4_idx[off][0] = i;
                    T4_idx[off][1] = j;
                    T4_idx[off][2] = k;
                    T4_idx[off][3] = l;
                    off++;
                }
            }
        }
        if (i == o-1) assert(off == T4_idx.length(0));
    }

    matrix<len_type> T3_idx({o*(o+1)*(o+2)/6 - o, 3}, 0, ROW_MAJOR);
    for (len_type i = 0, off = 0;i < o;i++)
    {
        for (len_type j = 0;j <= i;j++)
        {
            for (len_type k = 0;k <= j;k++)
            {
                if (i == j && j == k) continue;
                T3_idx[off][0] = i;
                T3_idx[off][1] = j;
                T3_idx[off][2] = k;
                off++;
            }
        }
        if (i == o-1) assert(off == T3_idx.length(0));
    }

    matrix<len_type> Wa_idx({o*o*(o+1)/2, 3}, 0, ROW_MAJOR);
    for (len_type i = 0, off = 0;i < o;i++)
    {
        for (len_type j = 0;j <= i;j++)
        {
            for (len_type k = 0;k < o;k++)
            {
                Wa_idx[off][0] = i;
                Wa_idx[off][1] = j;
                Wa_idx[off][2] = k;
                off++;
            }
        }
        if (i == o-1) assert(off == Wa_idx.length(0));
    }

    matrix<len_type> Wb_idx({o*o*o, 3}, 0, ROW_MAJOR);
    for (len_type i = 0, off = 0;i < o;i++)
    {
        for (len_type j = 0;j < o;j++)
        {
            for (len_type k = 0;k < o;k++)
            {
                Wb_idx[off][0] = i;
                Wb_idx[off][1] = j;
                Wb_idx[off][2] = k;
                off++;
            }
        }
        if (i == o-1) assert(off == Wb_idx.length(0));
    }

    batched_tensor<double> T4({v,v,v,v,o,o,o,o}, T4_idx);
    batched_tensor<double> T3({  v,v,v,  o,o,o}, T3_idx);
    batched_tensor<double> Wa({  v,v,v,  o,o,o}, Wa_idx);
    //batched_tensor<double> Wb({  v,v,v,  o,o,o}, Wb_idx);

    for (auto t : {&T4, &T3, &Wa})
    {
        std::vector<len_type> len(t->lengths());
        len.resize(t->dense_dimension());

        for (len_type i = 0;i < t->num_batches();i++)
        {
            double* data = t->batch_data(i);
            MArray::viterator<> it(len, t->strides());
            while (it.next(data)) *data = random_number<double>();
        }
    }

    batched_tensor<double> tmp0(T4);
    batched_tensor<double> tmp1(T4);
    batched_tensor<double> tmp2(T4);
    batched_tensor<double> tmp3(T4);

    double t0 = run_kernel(R,
    [&]
    {
        contract_batch_dumb<double>(1.0,   T3,   "ABEIJM",
                                           Wa,   "CDEKLM",
                                    0.5, tmp0, "ABCDIJKL");
    });

    flops = 0;

    double t1 = run_kernel(R,
    [&]
    {
        contract_batch_ref<double>(1.0,  T3,   "ABEIJM",
                                         Wa,   "CDEKLM",
                                   0.5, tmp1, "ABCDIJKL");
    });

    auto flops1 = flops.load();
    printf("%ld\n", flops1);
    flops = 0;

    double t2 = run_kernel(R,
    [&]
    {
        contract_batch<double>(1.0,  T3,   "ABEIJM",
                                     Wa,   "CDEKLM",
                               0.5, tmp2, "ABCDIJKL");
    });

    auto flops2 = flops.load();
    printf("%ld\n", flops2);
    flops = 0;

    double t3 = run_kernel(R,
    [&]
    {
        contract_batch2<double>(1.0,  T3,   "ABEIJM",
                                      Wa,   "CDEKLM",
                                0.5, tmp3, "ABCDIJKL");
    });

    auto flops3 = flops.load();
    printf("%ld\n", flops3);

    double d1 = diff(tmp0, tmp1);
    double d2 = diff(tmp0, tmp2);
    double d3 = diff(tmp0, tmp3);

    printf("%g %g %g\n", d1, d2, d3);
    printf("%g %g %g\n", t1, t2, t3);
    printf("%g %g %g\n", flops1/t1/1e9/R, flops2/t2/1e9/R, flops3/t3/1e9/R);

    return 0;
}
