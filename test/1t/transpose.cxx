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
    unsigned ndim_A = random_number(1,8);

    random_tensors(N,
                   0, 0,
                   ndim_A,
                   A, idx_A,
                   B, idx_B);
}

REPLICATED_TEMPLATED_TEST_CASE(transpose, R, T, all_types)
{
    varray<T> A, B, C;
    label_vector idx_A, idx_B;

    random_transpose(1000, A, idx_A, B, idx_B);

    unsigned ndim = A.dimension();
    auto perm = relative_permutation(idx_A, idx_B);

    TENSOR_INFO(A);
    TENSOR_INFO(B);
    INFO_OR_PRINT("perm = " << perm);

    auto neps = prod(A.lengths());

    T scale(10.0*random_unit<T>());

    C.reset(A);
    add<T>(T(1), A, idx_A.data(), T(0), B, idx_B.data());
    add<T>(scale, B, idx_B.data(), scale, C, idx_A.data());

    add<T>(-2*scale, A, idx_A.data(), T(1), C, idx_A.data());
    T error = reduce<T>(REDUCE_NORM_2, C, idx_A.data()).first;
    check("INVERSE", error, 2*scale*neps);

    B.reset(A);
    idx_B = idx_A;
    label_vector idx_C(ndim, 0);
    len_vector len_C(ndim);
    do
    {
        for (unsigned i = 0;i < ndim;i++)
        {
            unsigned j; for (j = 0;j < ndim && idx_A[j] != static_cast<label_type>(perm[i]+'a');j++) continue;
            idx_C[i] = idx_B[j];
            len_C[i] = B.length(j);
        }
        C.reset(len_C);
        add<T>(T(1), B, idx_B.data(), T(0), C, idx_C.data());
        B.reset(C);
        idx_B = idx_C;
    }
    while (idx_C != idx_A);

    add<T>(T(-1), A, idx_A.data(), T(1), C, idx_A.data());
    error = reduce<T>(REDUCE_NORM_2, C, idx_A.data()).first;
    check("CYCLE", error, neps);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_transpose, R, T, all_types)
{
    dpd_varray<T> A, B, C, D;
    label_vector idx_A, idx_B;

    random_transpose(1000, A, idx_A, B, idx_B);

    DPD_TENSOR_INFO(A);
    DPD_TENSOR_INFO(B);

    auto neps = dpd_varray<T>::size(B.irrep(), B.lengths());

    T scale(10.0*random_unit<T>());

    dpd_impl = dpd_impl_t::FULL;
    C.reset(B);
    add<T>(scale, A, idx_A.data(), scale, C, idx_B.data());

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(B);
    add<T>(scale, A, idx_A.data(), scale, D, idx_B.data());

    add<T>(T(-1), C, idx_B.data(), T(1), D, idx_B.data());
    T error = reduce<T>(REDUCE_NORM_2, D, idx_B.data()).first;

    check("BLOCKED", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_transpose, R, T, all_types)
{
    indexed_varray<T> A, B, C, D;
    label_vector idx_A, idx_B;

    random_transpose(1000, A, idx_A, B, idx_B);

    INDEXED_TENSOR_INFO(A);
    INDEXED_TENSOR_INFO(B);

    auto neps = prod(B.lengths());

    T scale(10.0*random_unit<T>());

    dpd_impl = dpd_impl_t::FULL;
    C.reset(B);
    add<T>(scale, A, idx_A.data(), scale, C, idx_B.data());

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(B);
    add<T>(scale, A, idx_A.data(), scale, D, idx_B.data());

    add<T>(T(-1), C, idx_B.data(), T(1), D, idx_B.data());
    T error = reduce<T>(REDUCE_NORM_2, D, idx_B.data()).first;

    check("BLOCKED", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_dpd_transpose, R, T, all_types)
{
    indexed_dpd_varray<T> A, B, C, D;
    label_vector idx_A, idx_B;

    random_transpose(1000, A, idx_A, B, idx_B);

    INDEXED_DPD_TENSOR_INFO(A);
    INDEXED_DPD_TENSOR_INFO(B);

    auto neps = dpd_varray<T>::size(B.irrep(), B.lengths());

    T scale(10.0*random_unit<T>());

    dpd_impl = dpd_impl_t::FULL;
    C.reset(B);
    add<T>(scale, A, idx_A.data(), scale, C, idx_B.data());

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(B);
    add<T>(scale, A, idx_A.data(), scale, D, idx_B.data());

    add<T>(T(-1), C, idx_B.data(), T(1), D, idx_B.data());
    T error = reduce<T>(REDUCE_NORM_2, D, idx_B.data()).first;

    check("BLOCKED", error, scale*neps);
}
