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

using namespace std;
using namespace tblis;
using namespace stl_ext;

template <typename T>
using gemm_p = void (*)(const char* transa, const char* transb,
                        const int* m, const int* n, const int* k,
                        const T* alpha, const T* A, const int* lda,
                                        const T* B, const int* ldb,
                        const T*  beta,       T* C, const int* ldc);

template <typename T>
using copy_p = void (*)(const int* n, const T* A, const int* inca,
                                            T* B, const int* incb);

template <typename T>
using gemm_f = typename remove_pointer<gemm_p<T>>::type;

template <typename T>
using copy_f = typename remove_pointer<copy_p<T>>::type;

extern "C"
{

gemm_f<   float> sgemm_;
gemm_f<  double> dgemm_;
gemm_f<scomplex> cgemm_;
gemm_f<dcomplex> zgemm_;

copy_f<   float> scopy_;
copy_f<  double> dcopy_;
copy_f<scomplex> ccopy_;
copy_f<dcomplex> zcopy_;

}

template<typename T> struct gemm_ptr;
template <> struct gemm_ptr<   float> { constexpr static gemm_p<   float> value = &sgemm_; };
template <> struct gemm_ptr<  double> { constexpr static gemm_p<  double> value = &dgemm_; };
template <> struct gemm_ptr<scomplex> { constexpr static gemm_p<scomplex> value = &cgemm_; };
template <> struct gemm_ptr<dcomplex> { constexpr static gemm_p<dcomplex> value = &zgemm_; };

template<typename T> struct copy_ptr;
template <> struct copy_ptr<   float> { constexpr static copy_p<   float> value = &scopy_; };
template <> struct copy_ptr<  double> { constexpr static copy_p<  double> value = &dcopy_; };
template <> struct copy_ptr<scomplex> { constexpr static copy_p<scomplex> value = &ccopy_; };
template <> struct copy_ptr<dcomplex> { constexpr static copy_p<dcomplex> value = &zcopy_; };

template<typename T>
void gemm(char transa, char transb,
          int m, int n, int k,
          T alpha, const T* A, int lda,
                   const T* B, int ldb,
          T  beta,       T* C, int ldc)
{
    gemm_ptr<T>::value(&transa, &transb, &m, &n, &k,
                       &alpha, A, &lda,
                               B, &ldb,
                        &beta, C, &ldc);
}

template<typename T>
void copy(int n, const T* A, int lda, T* B, int ldb)
{
    copy_ptr<T>::value(&n, A, &lda, B, &ldb);
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

enum algo_t { BLIS, BLIS_COPY, BLAS, BLAS_COPY };

template<typename T> struct type_char;
template <> struct type_char<   float> { static constexpr char value = 's'; };
template <> struct type_char<  double> { static constexpr char value = 'd'; };
template <> struct type_char<scomplex> { static constexpr char value = 'c'; };
template <> struct type_char<dcomplex> { static constexpr char value = 'z'; };

template<typename T, algo_t Algorithm>
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
        matrix<T> A_copy({m, k});
        matrix<T> B_copy({k, n});
        matrix<T> C_copy({m, n});

        double dt = run_kernel(R,
        [&]
        {
            if (Algorithm == BLIS)
            {
                mult(T(1), A, B, T(0), C);
            }
            else if (Algorithm == BLIS_COPY)
            {
                add(T(1), A, T(0), A_copy);
                add(T(1), B, T(0), B_copy);
                mult(T(1), A_copy, B_copy, T(0), C_copy);
                add(T(1), C_copy, T(0), C);
            }
            else if (Algorithm == BLAS)
            {
                gemm('N', 'N', m, n, k,
                     T(1), A.data(), m,
                           B.data(), k,
                     T(0), C.data(), m);
            }
            else if (Algorithm == BLAS_COPY)
            {
                copy(m*k, A.data(), 1, A_copy.data(), 1);
                copy(k*n, B.data(), 1, B_copy.data(), 1);
                gemm('N', 'N', m, n, k,
                     T(1), A_copy.data(), m,
                           B_copy.data(), k,
                     T(0), C_copy.data(), m);
                copy(m*n, C_copy.data(), 1, C.data(), 1);
            }
        });
        double gflops = 2*m*n*k*1e-9;

        printf("%e %e -- %s %c %ld %ld %ld\n", gflops, gflops / dt,
            (Algorithm == BLIS      ? "blis" :
             Algorithm == BLIS_COPY ? "blis+copy" :
             Algorithm == BLAS      ? "blas" :
                                      "blas+copy"),
            type_char<T>::value, m, n, k);
        fflush(stdout);
    }
};

template<typename T, algo_t Implementation, int N=3>
struct random_contraction
{
    len_type R;

    random_contraction(len_type R, const range_t<stride_type> &m_range,
                       const range_t<stride_type> &n_range,
                       const range_t<stride_type> &k_range)
    : R(R)
    {
        iterate_over_ranges(*this, {{'m', m_range}, {'n', n_range}, {'k', k_range}});
    }

    void operator()(const map<char, len_type> &values) const
    {
        len_type m = values.at('m');
        len_type n = values.at('n');
        len_type k = values.at('k');

        for (int i = 0;i < N;i++)
        {
            vector<len_type> len_m =
                random_product_constrained_sequence<len_type, ROUND_NEAREST>(random_number(1, 3), m);
            vector<len_type> len_n =
                random_product_constrained_sequence<len_type, ROUND_NEAREST>(random_number(1, 3), n);
            vector<len_type> len_k =
                random_product_constrained_sequence<len_type, ROUND_NEAREST>(random_number(1, 3), k);

            vector<label_type> idx_A, idx_B, idx_C;
            vector<len_type> len_A, len_B, len_C;
            char idx = 'a';

            map<char,len_type> lengths;

            stride_type tm = 1;
            for (len_type len : len_m)
            {
                idx_A.push_back(idx);
                len_A.push_back(len);
                idx_C.push_back(idx);
                len_C.push_back(len);
                lengths[idx] = len;
                idx++;
                tm *= len;
            }

            stride_type tn = 1;
            for (len_type len : len_n)
            {
                idx_B.push_back(idx);
                len_B.push_back(len);
                idx_C.push_back(idx);
                len_C.push_back(len);
                lengths[idx] = len;
                idx++;
                tn *= len;
            }

            stride_type tk = 1;
            for (len_type len : len_k)
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

            permute(idx_A, reorder_A);
            permute(len_A, reorder_A);
            permute(idx_B, reorder_B);
            permute(len_B, reorder_B);
            permute(idx_C, reorder_C);
            permute(len_C, reorder_C);

            tensor<T> A(len_A);
            tensor<T> B(len_B);
            tensor<T> C(len_C);

            double gflops = 2*tm*tn*tk*1e-9;
            tblis::internal::impl = (Implementation == BLAS ? tblis::internal::BLAS_BASED
                                                            : tblis::internal::BLIS_BASED);
            double dt = run_kernel(R,
            [&]
            {   mult(T(1), A, idx_A.data(),
                           B, idx_B.data(),
                     T(0), C, idx_C.data());
            });

            printf("%e %e -- %s %c %*s %*s %*s", gflops, gflops / dt,
                (Implementation == BLIS ? "rand_blis" : "rand_blas"), type_char<T>::value,
                static_cast<int>(idx_A.size()), idx_A.data(),
                static_cast<int>(idx_B.size()), idx_B.data(),
                static_cast<int>(idx_C.size()), idx_C.data());

            for (auto & l : lengths) printf(" %ld", l.second);
            printf("\n");
            fflush(stdout);
        }
    }
};

template<typename T, algo_t Implementation>
struct regular_contraction
{
    len_type R;
    vector<label_type> idx_A, idx_B, idx_C;

    regular_contraction(len_type R, const vector<label_type> &idx_A,
                        const vector<label_type> &idx_B,
                        const vector<label_type> &idx_C,
                        const map<char,range_t<stride_type>> &ranges)
    : R(R), idx_A(idx_A), idx_B(idx_B), idx_C(idx_C)
    {
        iterate_over_ranges(*this, ranges);
    }

    void operator()(const map<char, len_type> &lengths) const
    {
        vector<len_type> len_A, len_B, len_C;

        stride_type ntot = 1;
        for (auto & p : lengths) ntot *= p.second;

        for (char c : idx_A)
        {
            len_A.push_back(lengths.at(c));
        }

        for (char c : idx_B)
        {
            len_B.push_back(lengths.at(c));
        }

        for (char c : idx_C)
        {
            len_C.push_back(lengths.at(c));
        }

        tensor<T> A(len_A);
        tensor<T> B(len_B);
        tensor<T> C(len_C);

        double gflops = 2*ntot*1e-9;
        tblis::internal::impl = (Implementation == BLAS ? tblis::internal::BLAS_BASED
                                                        : tblis::internal::BLIS_BASED);
        double dt = run_kernel(R,
        [&]
        {
            mult(T(1), A, idx_A.data(),
                       B, idx_B.data(),
                 T(0), C, idx_C.data());
        });

        printf("%e %e -- %s %c %*s %*s %*s", gflops, gflops / dt,
            (Implementation == BLIS ? "reg_blis" : "reg_blas"), type_char<T>::value,
            static_cast<int>(idx_A.size()), idx_A.data(),
            static_cast<int>(idx_B.size()), idx_B.data(),
            static_cast<int>(idx_C.size()), idx_C.data());

        for (auto & l : lengths) printf(" %ld", l.second);
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
    rand_engine.seed(seed);

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
        if (algo == "blis" || algo == "blis+copy" || algo == "blas" ||
            algo == "blas+copy" || algo == "rand_blas" || algo == "rand_blis")
        {
            string m_range, n_range, k_range;
            iss >> m_range >> n_range >> k_range;

            auto m = parse_range(m_range);
            auto n = parse_range(n_range);
            auto k = parse_range(k_range);

            switch (dt)
            {
                case 's':
                    if      (algo == "blis")      gemm_experiment<float, BLIS>(R, m, n, k);
                    else if (algo == "blis+copy") gemm_experiment<float, BLIS_COPY>(R, m, n, k);
                    else if (algo == "blas")      gemm_experiment<float, BLAS>(R, m, n, k);
                    else if (algo == "blas+copy") gemm_experiment<float, BLAS_COPY>(R, m, n, k);
                    else if (algo == "rand_blis") random_contraction<float, BLIS>(R, m, n, k);
                    else if (algo == "rand_blas") random_contraction<float, BLAS>(R, m, n, k);
                    break;
                case 'd':
                    if      (algo == "blis")      gemm_experiment<double, BLIS>(R, m, n, k);
                    else if (algo == "blis+copy") gemm_experiment<double, BLIS_COPY>(R, m, n, k);
                    else if (algo == "blas")      gemm_experiment<double, BLAS>(R, m, n, k);
                    else if (algo == "blas+copy") gemm_experiment<double, BLAS_COPY>(R, m, n, k);
                    else if (algo == "rand_blis") random_contraction<double, BLIS>(R, m, n, k);
                    else if (algo == "rand_blas") random_contraction<double, BLAS>(R, m, n, k);
                    break;
                case 'c':
                    if      (algo == "blis")      gemm_experiment<scomplex, BLIS>(R, m, n, k);
                    else if (algo == "blis+copy") gemm_experiment<scomplex, BLIS_COPY>(R, m, n, k);
                    else if (algo == "blas")      gemm_experiment<scomplex, BLAS>(R, m, n, k);
                    else if (algo == "blas+copy") gemm_experiment<scomplex, BLAS_COPY>(R, m, n, k);
                    else if (algo == "rand_blis") random_contraction<scomplex, BLIS>(R, m, n, k);
                    else if (algo == "rand_blas") random_contraction<scomplex, BLAS>(R, m, n, k);
                    break;
                case 'z':
                    if      (algo == "blis")      gemm_experiment<dcomplex, BLIS>(R, m, n, k);
                    else if (algo == "blis+copy") gemm_experiment<dcomplex, BLIS_COPY>(R, m, n, k);
                    else if (algo == "blas")      gemm_experiment<dcomplex, BLAS>(R, m, n, k);
                    else if (algo == "blas+copy") gemm_experiment<dcomplex, BLAS_COPY>(R, m, n, k);
                    else if (algo == "rand_blis") random_contraction<dcomplex, BLIS>(R, m, n, k);
                    else if (algo == "rand_blas") random_contraction<dcomplex, BLAS>(R, m, n, k);
                    break;
            }
        }
        else if (algo == "reg_blas" || algo == "reg_blis")
        {
            string idx_A_, idx_B_, idx_C_;
            iss >> idx_A_ >> idx_B_ >> idx_C_;

            std::set<char>labels;
            for (char c : idx_A_) labels.insert(c);
            for (char c : idx_B_) labels.insert(c);
            for (char c : idx_C_) labels.insert(c);

            map<char,range_t<stride_type>> ranges;
            for (char c : labels)
            {
                string range;
                iss >> range;
                ranges[c] = parse_range(range);
            }

            vector<label_type> idx_A(idx_A_.begin(), idx_A_.end());
            vector<label_type> idx_B(idx_B_.begin(), idx_B_.end());
            vector<label_type> idx_C(idx_C_.begin(), idx_C_.end());

            switch (dt)
            {
                case 's':
                    if      (algo == "reg_blis") regular_contraction<float, BLIS>(R, idx_A, idx_B, idx_C, ranges);
                    else if (algo == "reg_blas") regular_contraction<float, BLAS>(R, idx_A, idx_B, idx_C, ranges);
                    break;
                case 'd':
                    if      (algo == "reg_blis") regular_contraction<double, BLIS>(R, idx_A, idx_B, idx_C, ranges);
                    else if (algo == "reg_blas") regular_contraction<double, BLAS>(R, idx_A, idx_B, idx_C, ranges);
                    break;
                case 'c':
                    if      (algo == "reg_blis") regular_contraction<scomplex, BLIS>(R, idx_A, idx_B, idx_C, ranges);
                    else if (algo == "reg_blas") regular_contraction<scomplex, BLAS>(R, idx_A, idx_B, idx_C, ranges);
                    break;
                case 'z':
                    if      (algo == "reg_blis") regular_contraction<dcomplex, BLIS>(R, idx_A, idx_B, idx_C, ranges);
                    else if (algo == "reg_blas") regular_contraction<dcomplex, BLAS>(R, idx_A, idx_B, idx_C, ranges);
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
