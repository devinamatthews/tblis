#include "../test.hpp"

/*
 * Creates a random tensor weighting operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_weight(stride_type N, T&& A, label_vector& idx_A,
                                  T&& B, label_vector& idx_B,
                                  T&& C, label_vector& idx_C)
{
    int ndim_A, ndim_B, ndim_C;
    int ndim_AC, ndim_BC;
    int ndim_ABC;

    do
    {
        ndim_A = random_number(1,8);
        ndim_B = random_number(1,8);
        ndim_C = random_number(1,8);
        ndim_ABC = ndim_A+ndim_B-ndim_C;
        ndim_AC = ndim_A-ndim_ABC;
        ndim_BC = ndim_B-ndim_ABC;
    }
    while (ndim_AC  < 0 ||
           ndim_BC  < 0 ||
           ndim_ABC < 0);

    random_tensors(N,
                   0, 0, 0,
                   0, ndim_AC, ndim_BC,
                   ndim_ABC,
                   A, idx_A,
                   B, idx_B,
                   C, idx_C);
}

REPLICATED_TEMPLATED_TEST_CASE(weight, R, T, all_types)
{
    marray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    random_weight(N, A, idx_A, B, idx_B, C, idx_C);

    TENSOR_INFO(A);
    TENSOR_INFO(B);
    TENSOR_INFO(C);

    auto neps = prod(C.lengths());

    T scale(10.0*random_unit<T>());

    impl = BLAS_BASED;
    D.reset(C);
    mult(scale, A, idx_A, B, idx_B, scale, D, idx_C);

    impl = REFERENCE;
    E.reset(C);
    mult(scale, A, idx_A, B, idx_B, scale, E, idx_C);

    add(-1, D, 1, E);
    T error = reduce<T>(REDUCE_NORM_2, E);

    check("BLAS", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_weight, R, T, all_types)
{
    dpd_marray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    random_weight(N, A, idx_A, B, idx_B, C, idx_C);

    DPD_TENSOR_INFO(A);
    DPD_TENSOR_INFO(B);
    DPD_TENSOR_INFO(C);

    auto neps = dpd_marray<T>::size(C.irrep(), C.lengths());

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(C);
    mult<T>(scale, A, idx_A, B, idx_B, scale, D, idx_C);

    dpd_impl = dpd_impl_t::FULL;
    E.reset(C);
    mult<T>(scale, A, idx_A, B, idx_B, scale, E, idx_C);

    add<T>(T(-1), D, idx_C, T(1), E, idx_C);
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C);

    check("BLOCKED", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_weight, R, T, all_types)
{
    indexed_marray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    random_weight(N, A, idx_A, B, idx_B, C, idx_C);

    INDEXED_TENSOR_INFO(A);
    INDEXED_TENSOR_INFO(B);
    INDEXED_TENSOR_INFO(C);

    auto neps = prod(C.lengths());

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(C);
    mult<T>(scale, A, idx_A, B, idx_B, scale, D, idx_C);

    dpd_impl = dpd_impl_t::FULL;
    E.reset(C);
    mult<T>(scale, A, idx_A, B, idx_B, scale, E, idx_C);

    for (auto& f : E.factors()) f = T(1);
    for (auto& f : D.factors()) f = T(1);
    add<T>(T(-1), D, idx_C, T(1), E, idx_C);
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C);

    check("BLOCKED", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_dpd_weight, R, T, all_types)
{
    indexed_dpd_marray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    random_weight(N, A, idx_A, B, idx_B, C, idx_C);

    INDEXED_DPD_TENSOR_INFO(A);
    INDEXED_DPD_TENSOR_INFO(B);
    INDEXED_DPD_TENSOR_INFO(C);

    auto neps = dpd_marray<T>::size(C.irrep(), C.lengths());

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(C);
    mult<T>(scale, A, idx_A, B, idx_B, scale, D, idx_C);

    dpd_impl = dpd_impl_t::FULL;
    E.reset(C);
    mult<T>(scale, A, idx_A, B, idx_B, scale, E, idx_C);

    for (auto& f : E.factors()) f = T(1);
    for (auto& f : D.factors()) f = T(1);
    add<T>(T(-1), D, idx_C, T(1), E, idx_C);
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C);

    check("BLOCKED", error, scale*neps);
}
