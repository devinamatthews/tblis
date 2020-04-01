#include "../test.hpp"

/*
 * Creates a random tensor replication operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_replicate(stride_type N, T&& A, label_vector& idx_A,
                                     T&& B, label_vector& idx_B)
{
    unsigned ndim_A, ndim_B;

    do
    {
        ndim_A = random_number(1,8);
        ndim_B = random_number(1,8);
        if (ndim_B < ndim_A) swap(ndim_A, ndim_B);
    }
    while (ndim_A == ndim_B);

    random_tensors(N,
                   0, ndim_B-ndim_A,
                   ndim_A,
                   A, idx_A,
                   B, idx_B);
}

REPLICATED_TEMPLATED_TEST_CASE(replicate, R, T, all_types)
{
    varray<T> A, B;
    label_vector idx_A, idx_B;

    random_replicate(1000, A, idx_A, B, idx_B);

    TENSOR_INFO(A);
    TENSOR_INFO(B);

    auto idx_B_only = exclusion(idx_B, idx_A);
    stride_type NB = prod(select_from(B.lengths(), idx_B, idx_B_only));
    auto neps = prod(B.lengths());

    T scale(10.0*random_unit<T>());

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A);
    T add_b = reduce<T>(REDUCE_SUM, B, idx_B);
    add(scale, A, idx_A, scale, B, idx_B);
    T calc_val = reduce<T>(REDUCE_SUM, B, idx_B);
    check("SUM", scale*(NB*ref_val+add_b), calc_val, neps*scale);

    ref_val = reduce<T>(REDUCE_NORM_1, A, idx_A);
    add(scale, A, idx_A, B, idx_B);
    calc_val = reduce<T>(REDUCE_NORM_1, B, idx_B);
    check("NRM1", std::abs(scale)*NB*ref_val, calc_val, neps*scale);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_replicate, R, T, all_types)
{
    dpd_varray<T> A, B, C, D;
    label_vector idx_A, idx_B;

    random_replicate(1000, A, idx_A, B, idx_B);

    DPD_TENSOR_INFO(A);
    DPD_TENSOR_INFO(B);

    auto neps = dpd_varray<T>::size(B.irrep(), B.lengths());

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

REPLICATED_TEMPLATED_TEST_CASE(indexed_replicate, R, T, all_types)
{
    indexed_varray<T> A, B, C, D;
    label_vector idx_A, idx_B;

    random_replicate(1000, A, idx_A, B, idx_B);

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

    add<T>(T(-1), C, idx_B, T(1), D, idx_B);
    T error = reduce<T>(REDUCE_NORM_2, D, idx_B);

    check("BLOCKED", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_dpd_replicate, R, T, all_types)
{
    indexed_dpd_varray<T> A, B, C, D;
    label_vector idx_A, idx_B;

    random_replicate(1000, A, idx_A, B, idx_B);

    INDEXED_DPD_TENSOR_INFO(A);
    INDEXED_DPD_TENSOR_INFO(B);

    auto neps = dpd_varray<T>::size(B.irrep(), B.lengths());

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
