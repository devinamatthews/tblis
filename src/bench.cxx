#include <cstdlib>
#include <algorithm>
#include <limits>
#include <stdint.h>
#include <iostream>
#include <random>
#include <numeric>
#include <getopt.h>
#include <sstream>
#include <iomanip>
#include <functional>

#include "core/tensor_iface.hpp"
#include "util/iterator.hpp"
#include "util/util.hpp"
#include "tblis/gemm.hpp"
#include "tblis/normfm.hpp"

using namespace std;
using namespace blis;
using namespace tblis;
using namespace blas;
using namespace tensor;
using namespace tensor::impl;
using namespace tensor::util;

namespace tensor
{
namespace util
{
mt19937 engine;
}
}

template <dim_t Rsub=1>
double RunKernel(dim_t R, const function<void()>& kernel)
{
    double bias = numeric_limits<double>::max();
    for (dim_t r = 0;r < R;r++)
    {
        double t0 = bli_clock();
        double t1 = bli_clock();
        bias = min(bias, t1-t0);
    }

    double dt = numeric_limits<double>::max();
    for (dim_t r = 0;r < R;r++)
    {
        double t0 = bli_clock();
        for (dim_t rs = 0;rs < Rsub;rs++) kernel();
        double t1 = bli_clock();
        dt = min(dt, t1-t0);
    }

    return dt-bias;
}

void RunExperiment(dim_t m0, dim_t m1, dim_t m_step,
                   dim_t n0, dim_t n1, dim_t n_step,
                   dim_t k0, dim_t k1, dim_t k_step,
                   const function<void(dim_t,dim_t,dim_t)>& experiment)
{
    for (dim_t m = m0, n = n0, k = k0;
         m >= min(m0,m1) && m <= max(m0,m1) &&
         n >= min(n0,n1) && n <= max(n0,n1) &&
         k >= min(k0,k1) && k <= max(k0,k1);
         m += m_step, n += n_step, k += k_step)
    {
        experiment(m, n, k);
    }
}

template <typename T>
void Benchmark(gint_t R)
{
    using namespace std::placeholders;

    FILE *mout, *tout;

    auto experiment =
    [&](dim_t m, dim_t n, dim_t k, const std::function<double(double,dim_t,dim_t,dim_t)>& eff_dim)
    {
        printf("%ld %ld %ld\n", m, n, k);

        T alpha = 1.0;
        T beta = 0.0;

        {
            Matrix<T> A(m, k);
            Matrix<T> B(k, n);
            Matrix<T> C(m, n);
            A = 0.0;
            B = 0.0;
            C = 0.0;

            Scalar<T> alp(alpha);
            Scalar<T> bet(beta);

            double gflops = 2*m*n*k*1e-9;
            double dt_1 = RunKernel(R, [&]{ bli_gemm(alp, A, B, bet, C); });
            double dt_2 = RunKernel(R,
            [&]
            {
                Matrix<T> A2(m, k);
                Matrix<T> B2(k, n);
                Matrix<T> C2(m, n);
                bli_dcopyv(BLIS_NO_CONJUGATE, m*k, (T*)A, 1, (T*)A2, 1);
                bli_dcopyv(BLIS_NO_CONJUGATE, k*n, (T*)B, 1, (T*)B2, 1);
                bli_gemm(alp, A2, B2, bet, C2);
                bli_daxpyv(BLIS_NO_CONJUGATE, m*n, &alpha, (T*)C2, 1, (T*)C, 1);
            });
            fprintf(mout, "%e %e %e\n", eff_dim(gflops,m,n,k), gflops/dt_1, gflops/dt_2);
        }

        for (int i = 0;i < 5;i++)
        {
            vector<dim_t> len_m = RandomProductConstrainedSequence<dim_t, ROUND_NEAREST>(RandomInteger(1, 3), m);
            vector<dim_t> len_n = RandomProductConstrainedSequence<dim_t, ROUND_NEAREST>(RandomInteger(1, 3), n);
            vector<dim_t> len_k = RandomProductConstrainedSequence<dim_t, ROUND_NEAREST>(RandomInteger(1, 3), k);

            string idx_A, idx_B, idx_C;
            vector<dim_t> len_A, len_B, len_C;
            char idx = 'a';

            dim_t tm = 1;
            for (dim_t len : len_m)
            {
                idx_A.push_back(idx);
                len_A.push_back(len);
                idx_C.push_back(idx);
                len_C.push_back(len);
                idx++;
                tm *= len;
            }

            dim_t tn = 1;
            for (dim_t len : len_n)
            {
                idx_B.push_back(idx);
                len_B.push_back(len);
                idx_C.push_back(idx);
                len_C.push_back(len);
                idx++;
                tn *= len;
            }

            dim_t tk = 1;
            for (dim_t len : len_k)
            {
                idx_A.push_back(idx);
                len_A.push_back(len);
                idx_B.push_back(idx);
                len_B.push_back(len);
                idx++;
                tk *= len;
            }

            vector<int> reorder_A = range<int>(len_A.size());
            vector<int> reorder_B = range<int>(len_B.size());
            vector<int> reorder_C = range<int>(len_C.size());

            random_shuffle(reorder_A.begin(), reorder_A.end());
            random_shuffle(reorder_B.begin(), reorder_B.end());
            random_shuffle(reorder_C.begin(), reorder_C.end());

            idx_A = permute(idx_A, reorder_A);
            len_A = permute(len_A, reorder_A);
            idx_B = permute(idx_B, reorder_B);
            len_B = permute(len_B, reorder_B);
            idx_C = permute(idx_C, reorder_C);
            len_C = permute(len_C, reorder_C);

            Tensor<T> A(len_A.size(), len_A);
            Tensor<T> B(len_B.size(), len_B);
            Tensor<T> C(len_C.size(), len_C);

            double gflops = 2*tm*tn*tk*1e-9;
            impl_type = BLAS_BASED;
            double dt_blas = RunKernel(R, [&]{ tensor_contract(alpha, A, idx_A, B, idx_B, beta, C, idx_C); });
            impl_type = BLIS_BASED;
            double dt_blis = RunKernel(R, [&]{ tensor_contract(alpha, A, idx_A, B, idx_B, beta, C, idx_C); });
            fprintf(tout, "%e %e %e\n", eff_dim(gflops,m,n,k), gflops/dt_blas, gflops/dt_blis);
        }
    };

    auto square_exp = bind(experiment, _1, _2, _3,
                           [](double gflops, dim_t m, dim_t n, dim_t k)
                           { return pow(gflops/2e-9, 1.0/3.0); });

    auto rankk_exp = bind(experiment, _1, _2, _3,
                          [](double gflops, dim_t m, dim_t n, dim_t k)
                          { return gflops/2e-9/m/n; });

    auto pp_exp = bind(experiment, _1, _2, _3,
                       [](double gflops, dim_t m, dim_t n, dim_t k)
                       { return sqrt(gflops/2e-9/k); });

    auto pb_exp = bind(experiment, _1, _2, _3,
                       [](double gflops, dim_t m, dim_t n, dim_t k)
                       { return gflops/2e-9/n/k; });

    auto bp_exp = bind(experiment, _1, _2, _3,
                       [](double gflops, dim_t m, dim_t n, dim_t k)
                       { return gflops/2e-9/m/k; });

    mout = fopen("out.mat.square", "w");
    tout = fopen("out.tensor.square", "w");

    RunExperiment(10, 500, 10,
                  10, 500, 10,
                  10, 500, 10,
                  square_exp);

    fclose(mout);
    fclose(tout);

    mout = fopen("out.mat.rankk", "w");
    tout = fopen("out.tensor.rankk", "w");

    RunExperiment(500, 500, 0,
                  500, 500, 0,
                  10, 500, 10,
                  rankk_exp);

    fclose(mout);
    fclose(tout);

    mout = fopen("out.mat.pp", "w");
    tout = fopen("out.tensor.pp", "w");

    RunExperiment(10, 600, 10,
                  10, 600, 10,
                  128, 128, 0,
                  pp_exp);

    fclose(mout);
    fclose(tout);

    mout = fopen("out.mat.pb", "w");
    tout = fopen("out.tensor.pb", "w");

    RunExperiment(10, 1000, 10,
                  64, 64, 0,
                  128, 128, 0,
                  pb_exp);

    fclose(mout);
    fclose(tout);

    mout = fopen("out.mat.bp", "w");
    tout = fopen("out.tensor.bp", "w");

    RunExperiment(64, 64, 0,
                  10, 1000, 10,
                  128, 128, 0,
                  bp_exp);

    fclose(mout);
    fclose(tout);
}

int main(int argc, char **argv)
{
    gint_t R = 30;

    bli_init();

    struct option opts[] = {{"rep", required_argument, NULL, 'r'},
                            {0, 0, 0, 0}};

    int arg;
    int index;
    istringstream iss;
    while ((arg = getopt_long(argc, argv, "r:", opts, &index)) != -1)
    {
        switch (arg)
        {
            case 'r':
                iss.str(optarg);
                iss >> R;
                break;
            case '?':
                abort();
                break;
        }
    }

    Benchmark<double>(R);

    bli_finalize();

    return 0;
}
