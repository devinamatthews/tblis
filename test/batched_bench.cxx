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
#include "internal/3t/mult.hpp"
#include "tensor/tblis_batched_tensor.hpp"

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

int main(int argc, char** argv)
{
    int R = 10;
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

    len_type v = 40;
    len_type o = 10;

    matrix<len_type> Z_idx({o*(o+1)*(o+2)*(o+3)/24 - o*o, 4});
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
                    Z_idx[off][0] = i;
                    Z_idx[off][1] = j;
                    Z_idx[off][2] = k;
                    Z_idx[off][3] = l;
                    off++;
                }
            }
        }
        if (i == o-1) assert(off == Z_idx.length(0));
    }

    matrix<len_type> T_idx({o*(o+1)*(o+2)/6 - o, 3});
    for (len_type i = 0, off = 0;i < o;i++)
    {
        for (len_type j = 0;j <= i;j++)
        {
            for (len_type k = 0;k <= j;k++)
            {
                if (i == j && j == k) continue;
                T_idx[off][0] = i;
                T_idx[off][1] = j;
                T_idx[off][2] = k;
                off++;
            }
        }
        if (i == o-1) assert(off == T_idx.length(0));
    }

    matrix<len_type> W_idx({o*o*(o+1)/2, 3});
    for (len_type i = 0, off = 0;i < o;i++)
    {
        for (len_type j = 0;j <= i;j++)
        {
            for (len_type k = 0;k < o;k++)
            {
                W_idx[off][0] = i;
                W_idx[off][1] = j;
                W_idx[off][2] = k;
                off++;
            }
        }
        if (i == o-1) assert(off == W_idx.length(0));
    }

    batched_tensor<double> Z({v,v,v,v,o,o,o,o}, Z_idx);
    batched_tensor<double> T({  v,v,v,  o,o,o}, T_idx);
    batched_tensor<double> W({  v,v,v,  o,o,o}, W_idx);

    double t1 = run_kernel(R,
    [&]
    {
        contract_batch_ref<double>(1.0, T,   "ABEIJM",
                                        W,   "CDEKLM",
                                   1.0, Z, "ABCDIJKL");
    });

    double t2 = run_kernel(R,
    [&]
    {
        contract_batch<double>(1.0, T,   "ABEIJM",
                                    W,   "CDEKLM",
                               1.0, Z, "ABCDIJKL");
    });

    //TODO: count FLOPs

    return 0;
}
