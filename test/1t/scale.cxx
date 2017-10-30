#include "../test.hpp"

REPLICATED_TEMPLATED_TEST_CASE(scale, R, T, all_types)
{
    varray<T> A;

    random_tensor(100, A);
    label_vector idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    TENSOR_INFO(A);

    auto neps = prod(A.lengths());

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;

    T scale(10.0*random_unit<T>());

    tblis::scale<T>(scale, A, idx_A.data());
    T calc_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    check("RANDOM", ref_val, calc_val/scale, neps);

    tblis::scale<T>(T(1), A, idx_A.data());
    calc_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    check("UNIT", ref_val, calc_val/scale, neps);

    tblis::scale<T>(T(0), A, idx_A.data());
    calc_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;
    check("ZERO", calc_val, neps);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_scale, R, T, all_types)
{
    dpd_varray<T> A, B, C;

    random_tensor(100, A);
    label_vector idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    DPD_TENSOR_INFO(A);

    auto NA = dpd_varray<T>::size(A.irrep(), A.lengths());

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;

    T scale(10.0*random_unit<T>());

    dpd_impl = dpd_impl_t::FULL;
    B.reset(A);
    auto vB = A.view(); vB.data(B.data());
    tblis::scale<T>(scale, vB, idx_A.data());

    dpd_impl = dpd_impl_t::BLOCKED;
    C.reset(A);
    auto vC = A.view(); vC.data(C.data());
    tblis::scale<T>(scale, vC, idx_A.data());

    add<T>(T(-1), B, idx_A.data(), T(1), C, idx_A.data());
    T error = reduce<T>(REDUCE_NORM_2, C, idx_A.data()).first;

    check("BLOCKED", error, scale*NA);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_scale, R, T, all_types)
{
    indexed_varray<T> A, B, C;

    random_tensor(100, A);
    label_vector idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    INDEXED_TENSOR_INFO(A);

    auto NA = prod(A.lengths());

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;

    T scale(10.0*random_unit<T>());

    dpd_impl = dpd_impl_t::FULL;
    B.reset(A);
    tblis::scale<T>(scale, B, idx_A.data());

    dpd_impl = dpd_impl_t::BLOCKED;
    C.reset(A);
    tblis::scale<T>(scale, C, idx_A.data());

    add<T>(T(-1), B, idx_A.data(), T(1), C, idx_A.data());
    T error = reduce<T>(REDUCE_NORM_2, C, idx_A.data()).first;

    check("BLOCKED", error, scale*NA);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_dpd_scale, R, T, all_types)
{
    indexed_dpd_varray<T> A, B, C;

    random_tensor(100, A);
    label_vector idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    INDEXED_DPD_TENSOR_INFO(A);

    auto NA = dpd_varray<T>::size(A.irrep(), A.lengths());

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A.data()).first;

    T scale(10.0*random_unit<T>());

    dpd_impl = dpd_impl_t::FULL;
    B.reset(A);
    tblis::scale<T>(scale, B, idx_A.data());

    dpd_impl = dpd_impl_t::BLOCKED;
    C.reset(A);
    tblis::scale<T>(scale, C, idx_A.data());

    add<T>(T(-1), B, idx_A.data(), T(1), C, idx_A.data());
    T error = reduce<T>(REDUCE_NORM_2, C, idx_A.data()).first;

    check("BLOCKED", error, scale*NA);
}
