#include "../test.hpp"

#include "nodes/gemm.hpp"
#include "matrix/normal_matrix.hpp"
#include "matrix/tensor_matrix.hpp"

/*
 * Creates a random matrix multiplication operation, where each matrix
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_gemm(stride_type N, matrix<T>& A,
                                matrix<T>& B,
                                matrix<T>& C)
{
    len_type m = random_number<len_type>(1, lrint(floor(sqrt(N/sizeof(T)))));
    len_type n = random_number<len_type>(1, lrint(floor(sqrt(N/sizeof(T)))));
    len_type k = random_number<len_type>(1, lrint(floor(sqrt(N/sizeof(T)))));

    random_matrix(N, m, k, A);
    random_matrix(N, k, n, B);
    random_matrix(N, m, n, C);
}

REPLICATED_TEMPLATED_TEST_CASE(gemm, R, T, all_types)
{
    matrix<T> A, B, C, D, E;

    random_gemm(N/10, A, B, C);

    T scale(10.0*random_unit<T>());

    len_type m = C.length(0);
    len_type n = C.length(1);
    len_type k = A.length(1);

    INFO_OR_PRINT("m, n, k    = " << m << ", " << n << ", " << k);
    INFO_OR_PRINT("rs_a, cs_a = " << A.stride(0) << ", " << A.stride(1));
    INFO_OR_PRINT("rs_b, cs_b = " << B.stride(0) << ", " << B.stride(1));
    INFO_OR_PRINT("rs_c, cs_c = " << C.stride(0) << ", " << C.stride(1));

    D.reset(C);
    gemm_ref<T>(scale, A, B, scale, D);

    E.reset(C);
    mult(scale, A, B, scale, E);

    add(-1, D, 1, E);
    T error = reduce<T>(REDUCE_NORM_2, E);

    check("REF", error, scale*m*n*k);
}

/*
REPLICATED_TEMPLATED_TEST_CASE(gemm_diag, R, T, all_types)
{
    matrix<T> A, B, C, E, F;

    random_gemm(N/10, A, B, C);

    T scale(10.0*random_unit<T>());

    len_type m = C.length(0);
    len_type n = C.length(1);
    len_type k = A.length(1);

    row<T> D({k}, uninitialized);
    for (len_type i = 0;i < k;i++) D[i] = random_unit<T>();

    INFO_OR_PRINT("m, n, k    = " << m << ", " << n << ", " << k);
    INFO_OR_PRINT("rs_a, cs_a = " << A.stride(0) << ", " << A.stride(1));
    INFO_OR_PRINT("rs_b, cs_b = " << B.stride(0) << ", " << B.stride(1));
    INFO_OR_PRINT("rs_c, cs_c = " << C.stride(0) << ", " << C.stride(1));

    E.reset(C);
    gemm_ref<T>(scale, A, D, B, scale, E);

    F.reset(C);
    mult(scale, A, D, B, scale, F);

    add(-1, E, 1, F);
    T error = reduce<T>(REDUCE_NORM_2, F);

    check("REF", error, scale*m*n*k);
}
*/
