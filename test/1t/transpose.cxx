#include "../test.hpp"

/*
 * Creates a random tensor transpose operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_transpose(stride_type N, T&& A, label_vector& idx_A,
                                     T&& B, label_vector& idx_B)
{
    auto ndim_A = random_number(1,8);

    random_tensors(N,
                   0, 0,
                   ndim_A,
                   A, idx_A,
                   B, idx_B);
}

REPLICATED_TEMPLATED_TEST_CASE(transpose, R, T, all_types)
{
    marray<T> A, B, C;
    label_vector idx_A, idx_B;

    random_transpose(1000, A, idx_A, B, idx_B);

    auto ndim = A.dimension();
    auto perm = relative_permutation(idx_A, idx_B);

    TENSOR_INFO(A);
    TENSOR_INFO(B);
    INFO_OR_PRINT("perm = " << perm);

    auto neps = prod(A.lengths());

    T scale(10.0*random_unit<T>());

    C.reset(A);
    add(scale, A, idx_A, 1, C, idx_A);
    T norm1 = reduce<T>(REDUCE_NORM_2, A, idx_A);
    T norm2 = reduce<T>(REDUCE_NORM_2, C, idx_A);
    check("COPY", std::abs(1+scale)*norm1, norm2, std::abs(1+scale)*neps);

    C.reset(A);
    add(A, idx_A, B, idx_B);
    add(scale, B, idx_B, scale, C, idx_A);

    add(-2*scale, A, idx_A, 1, C, idx_A);
    T error = reduce<T>(REDUCE_NORM_2, C, idx_A);
    check("INVERSE", error, 2*scale*neps);

    B.reset(A);
    idx_B = idx_A;
    label_vector idx_C(ndim, 0);
    len_vector len_C(ndim);
    do
    {
        for (auto i : range(ndim))
        {
            auto j = 0;
            while (j < ndim && idx_A[j] != static_cast<label_type>(perm[i]+'a'))
                j++;
            idx_C[i] = idx_B[j];
            len_C[i] = B.length(j);
        }
        C.reset(len_C);
        add(B, idx_B, C, idx_C);
        B.reset(C);
        idx_B = idx_C;
    }
    while (idx_C != idx_A);

    add(-1, A, idx_A, 1, C, idx_A);
    error = reduce<T>(REDUCE_NORM_2, C, idx_A);
    check("CYCLE", error, neps);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_transpose, R, T, all_types)
{
    dpd_marray<T> A, B, C, D;
    label_vector idx_A, idx_B;

    random_transpose(1000, A, idx_A, B, idx_B);

    DPD_TENSOR_INFO(A);
    DPD_TENSOR_INFO(B);

    auto neps = dpd_marray<T>::size(B.irrep(), B.lengths());

    T scale(10.0*random_unit<T>());

    dpd_impl = dpd_impl_t::FULL;
    C.reset(B);
    add<T>(scale, A, idx_A, scale, C, idx_B);

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(B);
    add<T>(scale, A, idx_A, scale, D, idx_B);

    add<T>(T(-1), C, idx_B, T(1), D, idx_B);
    T error = reduce<T>(REDUCE_NORM_2, D, idx_B);

    check("BLOCKED", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_transpose, R, T, all_types)
{
    indexed_marray<T> A, B, C, D;
    label_vector idx_A, idx_B;

    random_transpose(1000, A, idx_A, B, idx_B);

    INDEXED_TENSOR_INFO(A);
    INDEXED_TENSOR_INFO(B);

    auto neps = prod(B.lengths());

    T scale(10.0*random_unit<T>());

    dpd_impl = dpd_impl_t::FULL;
    C.reset(B);
    add<T>(scale, A, idx_A, scale, C, idx_B);

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(B);
    add<T>(scale, A, idx_A, scale, D, idx_B);

    for (auto& f : C.factors()) f = T(1);
    for (auto& f : D.factors()) f = T(1);
    add<T>(T(-1), C, idx_B, T(1), D, idx_B);
    T error = reduce<T>(REDUCE_NORM_2, D, idx_B);

    check("BLOCKED", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_dpd_transpose, R, T, all_types)
{
    indexed_dpd_marray<T> A, B, C, D;
    label_vector idx_A, idx_B;

    random_transpose(1000, A, idx_A, B, idx_B);

    INDEXED_DPD_TENSOR_INFO(A);
    INDEXED_DPD_TENSOR_INFO(B);

    auto neps = dpd_marray<T>::size(B.irrep(), B.lengths());

    T scale(10.0*random_unit<T>());

    dpd_impl = dpd_impl_t::FULL;
    C.reset(B);
    add<T>(scale, A, idx_A, scale, C, idx_B);

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(B);
    add<T>(scale, A, idx_A, scale, D, idx_B);

    for (auto& f : C.factors()) f = T(1);
    for (auto& f : D.factors()) f = T(1);
    add<T>(T(-1), C, idx_B, T(1), D, idx_B);
    T error = reduce<T>(REDUCE_NORM_2, D, idx_B);

    check("BLOCKED", error, scale*neps);
}
