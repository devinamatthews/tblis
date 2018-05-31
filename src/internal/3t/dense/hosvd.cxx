#include "cp_gradient.hpp"
#include "khatri_rao.hpp"
#include "hosvd.hpp"

#include "util/tensor.hpp"

#include "memory/memory_pool.hpp"

#include "iface/1t/add.h"

#include "external/lawrap/lapack.h"

namespace tblis
{
namespace internal
{

template <typename T>
void hosvd_once(len_vector& len_A,
                T* A, const stride_vector& stride_A, T* tmp,
                const ptr_vector<T>& U,
                const stride_vector& ld_U,
                double tol)
{
    std::vector<T> s(stl_ext::max(len_A));

    label_vector idx = range<label_type>(len_A.size());
    add<T>(T(1), {len_A, A, stride_A}, idx.data(),
           T(0), {len_A, tmp, COLUMN_MAJOR}, idx.data());

    len_type m = stl_ext::prod(len_A);
    for (unsigned i = len_A.size();i --> 0;)
    {
        memcpy(A, tmp, m*sizeof(T));

        len_type n = len_A[i];
        m /= n;

        LAWrap::gesvd('N', 'A', m, n, tmp, m, s.data(),
                      nullptr, 0, U[i], ld_U[i]);

        len_type k = 0;
        for (;k < n;n++) if (std::abs(s[k])) break;
        len_A[i] = k;

        LAWrap::gemm('T', 'T', k, m, n,
                     T(1), U[i], ld_U[i],
                              A,       m,
                     T(0),  tmp,       k);

        m *= k;
    }

    add<T>(T(1), {len_A, tmp, COLUMN_MAJOR}, idx.data(),
           T(0), {len_A, A, stride_A}, idx.data());
}

template <typename T>
void hosvd(len_vector& len_A_, T* A, const stride_vector& stride_A_,
           const ptr_vector<T>& U_, const stride_vector& ld_U,
           double tol, bool iterate)
{
    auto reorder_A = detail::sort_by_stride(stride_A_);

    auto len_A = stl_ext::permuted(len_A_, reorder_A);
    auto stride_A = stl_ext::permuted(stride_A_, reorder_A);
    auto U = stl_ext::permuted(U_, reorder_A);

    T* tmp = new T[stl_ext::prod(len_A)];

    len_vector old_len_A = len_A;
    hosvd_once(len_A, A, stride_A, tmp, U, ld_U, tol);

    if (iterate)
    {
        ptr_vector<T> tmp_U;
        len_type max_size = 0;
        for (unsigned i = 0;i < len_A.size();i++)
        {
            tmp_U.push_back(new T[ld_U[i]*len_A[i]]);
            max_size = std::max(max_size, ld_U[i]*len_A[i]);
        }
        T* tmp2 = new T[max_size];

        while (old_len_A != len_A)
        {
            old_len_A = len_A;

            hosvd_once(len_A, A, stride_A, tmp, tmp_U, ld_U, tol);

            for (unsigned i = 0;i < len_A.size();i++)
            {
                LAWrap::gemm('N', 'N', ld_U[i], len_A[i], old_len_A[i],
                             T(1),     U[i], ld_U[i],
                                   tmp_U[i], ld_U[i],
                             T(0),     tmp2, ld_U[i]);
                memcpy(U[i], tmp2, ld_U[i]*len_A[i]*sizeof(T));
            }
        }

        delete[] tmp2;
        for (unsigned i = 0;i < len_A.size();i++)
            delete[] tmp_U[i];
    }

    delete[] tmp;
    len_A_ = stl_ext::unpermuted(len_A, reorder_A);
}

#define FOREACH_TYPE(T) \
template void hosvd(len_vector& len_A, T* A, const stride_vector& stride_A, \
                    const ptr_vector<T>& U, const stride_vector& ld_U, \
                    double tol, bool iterate)
#include "configs/foreach_type.h"

}
}
