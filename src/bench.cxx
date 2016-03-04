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

void RunExperiment(dim_t m, dim_t n, dim_t k,
                   dim_t lb, dim_t ub, dim_t step,
                   const function<void(dim_t,dim_t,dim_t)>& experiment)
{
    bool scale_m = m == 0;
    bool scale_n = n == 0;
    bool scale_k = k == 0;

    for (dim_t s = lb;s <= ub;s += step)
    {
        if (scale_m) m = s;
        if (scale_n) n = s;
        if (scale_k) k = s;

        experiment(m, n, k);
    }
}

template <typename T>
void Benchmark(gint_t R)
{
    FILE* mout = fopen("out.mat", "w");
    FILE* tout = fopen("out.tensor", "w");

    RunExperiment(0, 0, 0, 10, 500, 10,
    [&](dim_t m, dim_t n, dim_t k)
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

            double flops = 2*m*n*k*1e-9;
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
            printf("\n");
            printf("%e %e %e\n", flops, flops/dt_1, flops/dt_2);
            printf("\n");
            fprintf(mout, "%e %e %e\n", flops, flops/dt_1, flops/dt_2);
        }

        for (int i = 0;i < 10;i++)
        {
            vector<dim_t> len_m = RandomProductConstrainedSequence<dim_t, ROUND_NEAREST>(RandomInteger(1, 3), m);
            vector<dim_t> len_n = RandomProductConstrainedSequence<dim_t, ROUND_NEAREST>(RandomInteger(1, 3), n);
            vector<dim_t> len_k = RandomProductConstrainedSequence<dim_t, ROUND_NEAREST>(RandomInteger(1, 3), k);

            string idx_A, idx_B, idx_C;
            vector<dim_t> len_A, len_B, len_C;
            char idx = 'a';

            cout << len_m << " " << len_n << " " << len_k << endl;

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

            double flops = 2*tm*tn*tk*1e-9;
            impl_type = BLAS_BASED;
            double dt_blas = RunKernel(R, [&]{ tensor_contract(alpha, A, idx_A, B, idx_B, beta, C, idx_C); });
            impl_type = BLIS_BASED;
            double dt_blis = RunKernel(R, [&]{ tensor_contract(alpha, A, idx_A, B, idx_B, beta, C, idx_C); });
            printf("%e %e %e\n", flops, flops/dt_blas, flops/dt_blis);
            fprintf(tout, "%e %e %e\n", flops, flops/dt_blas, flops/dt_blis);
        }
    });

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
