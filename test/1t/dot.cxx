#include "../test.hpp"

/*
 * Creates a random tensor dot product operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_dot(stride_type N, T&& A, label_vector& idx_A,
                               T&& B, label_vector& idx_B)
{
    auto ndim_A = random_number(1,8);

    random_tensors(N,
                   0, 0,
                   ndim_A,
                   A, idx_A,
                   B, idx_B);
}

REPLICATED_TEMPLATED_TEST_CASE(dot, R, T, all_types)
{
    marray<T> A, B;
    label_vector idx_A, idx_B;

    random_dot(1000, A, idx_A, B, idx_B);

    TENSOR_INFO(A);
    TENSOR_INFO(B);

    auto neps = prod(A.lengths());

    add(A, idx_A, B, idx_B);
    B.for_each_element([](T& e) { e = tblis::conj(e); });
    T ref_val = reduce<T>(REDUCE_NORM_2, A, idx_A);
    T calc_val = dot<T>(A, idx_A, B, idx_B);
    check("NRM2", ref_val*ref_val, calc_val, neps);

    B = T(1);
    ref_val = reduce<T>(REDUCE_SUM, A, idx_A);
    calc_val = dot<T>(A, idx_A, B, idx_B);
    check("UNIT", ref_val, calc_val, neps);

    B = T(0);
    calc_val = dot<T>(A, idx_A, B, idx_B);
    check("ZERO", calc_val, neps);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_dot, R, T, all_types)
{
    dpd_marray<T> A, B;
    label_vector idx_A, idx_B;

    random_dot(1000, A, idx_A, B, idx_B);

    DPD_TENSOR_INFO(A);
    DPD_TENSOR_INFO(B);

    auto neps = dpd_marray<T>::size(A.irrep(), A.lengths());

    dpd_impl = dpd_impl_t::FULL;
    T ref_val = dot<T>(A, idx_A, B, idx_B);

    dpd_impl = dpd_impl_t::BLOCKED;
    T calc_val = dot<T>(A, idx_A, B, idx_B);

    check("BLOCKED", calc_val, ref_val, neps);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_dot, R, T, all_types)
{
    indexed_marray<T> A, B;
    label_vector idx_A, idx_B;

    random_dot(1000, A, idx_A, B, idx_B);

    INDEXED_TENSOR_INFO(A);
    INDEXED_TENSOR_INFO(B);

    auto neps = prod(A.lengths());

    dpd_impl = dpd_impl_t::FULL;
    T ref_val = dot<T>(A, idx_A, B, idx_B);

    dpd_impl = dpd_impl_t::BLOCKED;
    T calc_val = dot<T>(A, idx_A, B, idx_B);

    check("BLOCKED", calc_val, ref_val, neps);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_dpd_dot, R, T, all_types)
{
    indexed_dpd_marray<T> A, B;
    label_vector idx_A, idx_B;

    random_dot(1000, A, idx_A, B, idx_B);

    INDEXED_DPD_TENSOR_INFO(A);
    INDEXED_DPD_TENSOR_INFO(B);

    auto neps = dpd_marray<T>::size(A.irrep(), A.lengths());

    dpd_impl = dpd_impl_t::FULL;
    T ref_val = dot<T>(A, idx_A, B, idx_B);

    dpd_impl = dpd_impl_t::BLOCKED;
    T calc_val = dot<T>(A, idx_A, B, idx_B);

    check("BLOCKED", calc_val, ref_val, neps);
}
