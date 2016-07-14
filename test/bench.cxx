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
#include <set>
#include <map>

#include "tblis.hpp"

using tblis::scomplex;
using tblis::dcomplex;

#define LAWRAP_COMPLEX_DEFINED
#include "external/lawrap/blas.h"

using namespace std;
using namespace tblis;
using namespace tblis::util;
using namespace tblis::impl;
using namespace LAWrap;

namespace tblis
{
namespace util
{
mt19937 engine;
}
}

range_t<stride_type> parse_range(const string& s)
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
        mn = stol(s.substr(0,colon1));
        mx = stol(s.substr(colon1+1));
    }
    else
    {
        mn = stol(s.substr(0,colon1));
        mx = stol(s.substr(colon1+1,colon2-colon1-1));
        delta = stol(s.substr(colon2+1));
    }

    return {mn, mx+delta, delta};
}

template <typename Kernel, typename... Args>
double run_kernel(idx_type R, const Kernel& kernel, Args&&... args)
{
    double bias = numeric_limits<double>::max();
    for (idx_type r = 0;r < R;r++)
    {
        double t0 = tic();
        double t1 = tic();
        bias = min(bias, t1-t0);
    }

    double dt = numeric_limits<double>::max();
    for (idx_type r = 0;r < R;r++)
    {
        double t0 = tic();
        kernel(std::forward<Args>(args)...);
        double t1 = tic();
        dt = min(dt, t1-t0);
    }

    return dt-bias;
}

template <typename Experiment>
void iterate_over_ranges_helper(const Experiment& experiment,
                                const map<char,range_t<stride_type>>& ranges,
                                map<char,range_t<stride_type>>::const_iterator range,
                                map<char,idx_type>& values)
{
    if (range == ranges.end())
    {
        idx_type var = 0;

        for (auto& r : ranges)
        {
            if (r.second.size() > 1)
            {
                var = values[r.first];
            }
        }

        for (auto& r : ranges)
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

template <typename Experiment>
void iterate_over_ranges(const Experiment& experiment,
                         const map<char,range_t<stride_type>>& ranges)
{
    map<char,idx_type> values;
    iterate_over_ranges_helper(experiment, ranges, ranges.begin(), values);
}

enum algo_t
{
    BLIS,
    BLIS_COPY,
    BLAS,
    BLAS_COPY
};

template <typename T> struct type_char;
template <> struct type_char<   float> { static constexpr char value = 's'; };
template <> struct type_char<  double> { static constexpr char value = 'd'; };
template <> struct type_char<scomplex> { static constexpr char value = 'c'; };
template <> struct type_char<dcomplex> { static constexpr char value = 'z'; };

template <typename T, algo_t Algorithm>
struct gemm_experiment
{
    idx_type R;

    gemm_experiment(idx_type R,
                    const range_t<stride_type>& m_range,
                    const range_t<stride_type>& n_range,
                    const range_t<stride_type>& k_range)
    : R(R)
    {
        iterate_over_ranges(*this, {{'m', m_range}, {'n', n_range}, {'k', k_range}});
    }

    void operator()(const map<char,idx_type>& values) const
    {
        idx_type m = values.at('m');
        idx_type n = values.at('n');
        idx_type k = values.at('k');

        matrix<T> A(m, k);
        matrix<T> B(k, n);
        matrix<T> C(m, n);
        matrix<T> A_copy(m, k);
        matrix<T> B_copy(k, n);
        matrix<T> C_copy(m, n);

        double dt = run_kernel(R,
        [&]
        {
            if (Algorithm == BLIS)
            {
                tblis_gemm<T>(1.0, A, B, 0.0, C);
            }
            else if (Algorithm == BLIS_COPY)
            {
                tblis_copyv(false, m*k, A.data(), 1, A_copy.data(), 1);
                tblis_copyv(false, k*n, B.data(), 1, B_copy.data(), 1);
                tblis_gemm<T>(1.0, A_copy, B_copy, 0.0, C_copy);
                tblis_copyv(false, m*n, C_copy.data(), 1, C.data(), 1);
            }
            else if (Algorithm == BLAS)
            {
                gemm('N', 'N', m, n, k,
                     1.0, A.data(), m,
                          B.data(), k,
                     0.0, C.data(), m);
            }
            else if (Algorithm == BLAS_COPY)
            {
                copy(m*k, A.data(), 1, A_copy.data(), 1);
                copy(k*n, B.data(), 1, B_copy.data(), 1);
                gemm('N', 'N', m, n, k,
                     1.0, A_copy.data(), m,
                          B_copy.data(), k,
                     0.0, C_copy.data(), m);
                copy(m*n, C_copy.data(), 1, C.data(), 1);
            }
        });
        double gflops = 2*m*n*k*1e-9;

        printf("%e %e -- %s %c %d %d %d\n", gflops, gflops/dt,
               (Algorithm == BLIS ? "blis" :
                Algorithm == BLIS_COPY ? "blis+copy" :
                Algorithm == BLAS ? "blas" : "blas+copy"),
               type_char<T>::value, m, n, k);
        fflush(stdout);
    }
};

template <typename T, algo_t Implementation, int N=3>
struct random_contraction
{
    idx_type R;

    random_contraction(idx_type R,
                    const range_t<stride_type>& m_range,
                    const range_t<stride_type>& n_range,
                    const range_t<stride_type>& k_range)
    : R(R)
    {
        iterate_over_ranges(*this, {{'m', m_range}, {'n', n_range}, {'k', k_range}});
    }

    void operator()(const map<char,idx_type>& values) const
    {
        idx_type m = values.at('m');
        idx_type n = values.at('n');
        idx_type k = values.at('k');

        for (int i = 0;i < N;i++)
        {
            vector<idx_type> len_m = RandomProductConstrainedSequence<idx_type, ROUND_NEAREST>(RandomInteger(1, 3), m);
            vector<idx_type> len_n = RandomProductConstrainedSequence<idx_type, ROUND_NEAREST>(RandomInteger(1, 3), n);
            vector<idx_type> len_k = RandomProductConstrainedSequence<idx_type, ROUND_NEAREST>(RandomInteger(1, 3), k);

            string idx_A, idx_B, idx_C;
            vector<idx_type> len_A, len_B, len_C;
            char idx = 'a';

            map<char,idx_type> lengths;

            idx_type tm = 1;
            for (idx_type len : len_m)
            {
                idx_A.push_back(idx);
                len_A.push_back(len);
                idx_C.push_back(idx);
                len_C.push_back(len);
                lengths[idx] = len;
                idx++;
                tm *= len;
            }

            idx_type tn = 1;
            for (idx_type len : len_n)
            {
                idx_B.push_back(idx);
                len_B.push_back(len);
                idx_C.push_back(idx);
                len_C.push_back(len);
                lengths[idx] = len;
                idx++;
                tn *= len;
            }

            idx_type tk = 1;
            for (idx_type len : len_k)
            {
                idx_A.push_back(idx);
                len_A.push_back(len);
                idx_B.push_back(idx);
                len_B.push_back(len);
                lengths[idx] = len;
                idx++;
                tk *= len;
            }

            vector<unsigned> reorder_A = range<unsigned>(len_A.size());
            vector<unsigned> reorder_B = range<unsigned>(len_B.size());
            vector<unsigned> reorder_C = range<unsigned>(len_C.size());

            random_shuffle(reorder_A.begin(), reorder_A.end());
            random_shuffle(reorder_B.begin(), reorder_B.end());
            random_shuffle(reorder_C.begin(), reorder_C.end());

            idx_A = permute(idx_A, reorder_A);
            len_A = permute(len_A, reorder_A);
            idx_B = permute(idx_B, reorder_B);
            len_B = permute(len_B, reorder_B);
            idx_C = permute(idx_C, reorder_C);
            len_C = permute(len_C, reorder_C);

            tensor<T> A(len_A);
            tensor<T> B(len_B);
            tensor<T> C(len_C);

            double gflops = 2*tm*tn*tk*1e-9;
            impl_type = (Implementation == BLAS ? BLAS_BASED : BLIS_BASED);
            double dt = run_kernel(R, [&]{ tensor_contract<T>(1.0, A, idx_A, B, idx_B, 0.0, C, idx_C); });

            printf("%e %e -- %s %c %s %s %s", gflops, gflops/dt,
                   (Implementation == BLIS ? "rand_blis" : "rand_blas"),
                   type_char<T>::value, idx_A.c_str(), idx_B.c_str(), idx_C.c_str());

            for (auto& l : lengths) printf(" %d", l.second);
            printf("\n");
            fflush(stdout);
        }
    }
};

template <typename T, algo_t Implementation>
struct regular_contraction
{
    idx_type R;
    string idx_A, idx_B, idx_C;

    regular_contraction(idx_type R, const string& idx_A, const string& idx_B, const string& idx_C,
                        const map<char,range_t<stride_type>>& ranges)
    : R(R), idx_A(idx_A), idx_B(idx_B), idx_C(idx_C)
    {
        iterate_over_ranges(*this, ranges);
    }

    void operator()(const map<char,idx_type>& lengths) const
    {
        vector<idx_type> len_A, len_B, len_C;

        stride_type ntot = 1;

        for (char c : idx_A)
        {
            len_A.push_back(lengths.at(c));
            ntot *= len_A.back();
        }

        for (char c : idx_B)
        {
            len_B.push_back(lengths.at(c));
            ntot *= len_B.back();
        }

        for (char c : idx_C)
        {
            len_C.push_back(lengths.at(c));
            ntot *= len_C.back();
        }

        tensor<T> A(len_A);
        tensor<T> B(len_B);
        tensor<T> C(len_C);

        double gflops = 2*ntot*1e-9;
        impl_type = (Implementation == BLAS ? BLAS_BASED : BLIS_BASED);
        double dt = run_kernel(R, [&]{ tensor_contract<T>(1.0, A, idx_A, B, idx_B, 0.0, C, idx_C); });

        printf("%e %e -- %s %c %s %s %s", gflops, gflops/dt,
               (Implementation == BLIS ? "reg_blis" : "reg_blas"),
               type_char<T>::value, idx_A.c_str(), idx_B.c_str(), idx_C.c_str());

        for (auto& l : lengths) printf(" %d", l.second);
        printf("\n");
        fflush(stdout);
    }
};

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
    engine.seed(seed);

    string line;
    while (getline(cin, line) && !line.empty())
    {
        if (line[0] == '#') continue;

        string algo;
        char dt;
        istringstream iss(line);
        iss >> algo >> dt;

        if (string("sdcz").find(dt) == string::npos)
        {
            cerr << "Unknown datatype: " << dt << endl;
            exit(1);
        }

        if (algo == "blis" || algo == "blis+copy" ||
            algo == "blas" || algo == "blas+copy" ||
            algo == "rand_blas" || algo == "rand_blis")
        {
            string m_range, n_range, k_range;
            iss >> m_range >> n_range >> k_range;

            auto m = parse_range(m_range);
            auto n = parse_range(n_range);
            auto k = parse_range(k_range);

            switch (dt)
            {
                case 's':
                    if      (algo ==      "blis")    gemm_experiment<float,     BLIS>(R, m, n, k);
                    else if (algo == "blis+copy")    gemm_experiment<float,BLIS_COPY>(R, m, n, k);
                    else if (algo ==      "blas")    gemm_experiment<float,     BLAS>(R, m, n, k);
                    else if (algo == "blas+copy")    gemm_experiment<float,BLAS_COPY>(R, m, n, k);
                    else if (algo == "rand_blis") random_contraction<float,     BLIS>(R, m, n, k);
                    else if (algo == "rand_blas") random_contraction<float,     BLAS>(R, m, n, k);
                    break;
                case 'd':
                    if      (algo ==      "blis")    gemm_experiment<double,     BLIS>(R, m, n, k);
                    else if (algo == "blis+copy")    gemm_experiment<double,BLIS_COPY>(R, m, n, k);
                    else if (algo ==      "blas")    gemm_experiment<double,     BLAS>(R, m, n, k);
                    else if (algo == "blas+copy")    gemm_experiment<double,BLAS_COPY>(R, m, n, k);
                    else if (algo == "rand_blis") random_contraction<double,     BLIS>(R, m, n, k);
                    else if (algo == "rand_blas") random_contraction<double,     BLAS>(R, m, n, k);
                    break;
                case 'c':
                    if      (algo ==      "blis")    gemm_experiment<scomplex,     BLIS>(R, m, n, k);
                    else if (algo == "blis+copy")    gemm_experiment<scomplex,BLIS_COPY>(R, m, n, k);
                    else if (algo ==      "blas")    gemm_experiment<scomplex,     BLAS>(R, m, n, k);
                    else if (algo == "blas+copy")    gemm_experiment<scomplex,BLAS_COPY>(R, m, n, k);
                    else if (algo == "rand_blis") random_contraction<scomplex,     BLIS>(R, m, n, k);
                    else if (algo == "rand_blas") random_contraction<scomplex,     BLAS>(R, m, n, k);
                    break;
                case 'z':
                    if      (algo ==      "blis")    gemm_experiment<dcomplex,     BLIS>(R, m, n, k);
                    else if (algo == "blis+copy")    gemm_experiment<dcomplex,BLIS_COPY>(R, m, n, k);
                    else if (algo ==      "blas")    gemm_experiment<dcomplex,     BLAS>(R, m, n, k);
                    else if (algo == "blas+copy")    gemm_experiment<dcomplex,BLAS_COPY>(R, m, n, k);
                    else if (algo == "rand_blis") random_contraction<dcomplex,     BLIS>(R, m, n, k);
                    else if (algo == "rand_blas") random_contraction<dcomplex,     BLAS>(R, m, n, k);
                    break;
            }
        }
        else if (algo == "reg_blas" || algo == "reg_blis")
        {
            string idx_A, idx_B, idx_C;
            iss >> idx_A >> idx_B >> idx_C;

            set<char> labels;
            for (char c : idx_A) labels.insert(c);
            for (char c : idx_B) labels.insert(c);
            for (char c : idx_C) labels.insert(c);

            map<char,range_t<stride_type>> ranges;
            for (char c : labels)
            {
                string range;
                iss >> range;
                ranges[c] = parse_range(range);
            }

            switch (dt)
            {
                case 's':
                    if      (algo == "reg_blis") regular_contraction<float,BLIS>(R, idx_A, idx_B, idx_C, ranges);
                    else if (algo == "reg_blas") regular_contraction<float,BLAS>(R, idx_A, idx_B, idx_C, ranges);
                    break;
                case 'd':
                    if      (algo == "reg_blis") regular_contraction<double,BLIS>(R, idx_A, idx_B, idx_C, ranges);
                    else if (algo == "reg_blas") regular_contraction<double,BLAS>(R, idx_A, idx_B, idx_C, ranges);
                    break;
                case 'c':
                    if      (algo == "reg_blis") regular_contraction<scomplex,BLIS>(R, idx_A, idx_B, idx_C, ranges);
                    else if (algo == "reg_blas") regular_contraction<scomplex,BLAS>(R, idx_A, idx_B, idx_C, ranges);
                    break;
                case 'z':
                    if      (algo == "reg_blis") regular_contraction<dcomplex,BLIS>(R, idx_A, idx_B, idx_C, ranges);
                    else if (algo == "reg_blas") regular_contraction<dcomplex,BLAS>(R, idx_A, idx_B, idx_C, ranges);
                    break;
            }
        }
        else
        {
            cerr << "Unknown algorithm: " << algo << endl;
            exit(1);
        }
    }

    return 0;
}
