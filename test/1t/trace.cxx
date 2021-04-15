#include "../test.hpp"

/*
 * Creates a random tensor trace operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_trace(stride_type N, T&& A, string& idx_A,
                                 T&& B, string& idx_B)
{
    int ndim_A, ndim_B;

    do
    {
        ndim_A = random_number(1,8);
        ndim_B = random_number(1,8);
        if (ndim_A < ndim_B) swap(ndim_A, ndim_B);
    }
    while (ndim_A == ndim_B);

    random_tensors(N,
                   ndim_A-ndim_B, 0,
                   ndim_B,
                   A, idx_A,
                   B, idx_B);
}

REPLICATED_TEMPLATED_TEST_CASE(trace, R, T, all_types)
{
    varray<T> A, B;
    string idx_A, idx_B;

    random_trace(1000, A, idx_A, B, idx_B);

    TENSOR_INFO(A);
    TENSOR_INFO(B);

    auto neps = prod(A.lengths());

    T scale(10.0*random_unit<T>());

    T ref_val = reduce<T>(REDUCE_SUM, A, idx_A);
    T add_b = reduce<T>(REDUCE_SUM, B, idx_B);
    add(scale, A, idx_A, scale, B, idx_B);
    T calc_val = reduce<T>(REDUCE_SUM, B, idx_B);
    check("SUM", scale*(ref_val+add_b), calc_val, neps*scale);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_trace, R, T, all_types)
{
    dpd_varray<T> A, B, C, D;
    string idx_A, idx_B;

    random_trace(1000, A, idx_A, B, idx_B);

    DPD_TENSOR_INFO(A);
    DPD_TENSOR_INFO(B);

    auto neps = A.size();

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

REPLICATED_TEMPLATED_TEST_CASE(indexed_trace, R, T, all_types)
{
    indexed_varray<T> A, B, C, D;
    string idx_A, idx_B;

    random_trace(1000, A, idx_A, B, idx_B);

    INDEXED_TENSOR_INFO(A);
    INDEXED_TENSOR_INFO(B);

    auto neps = prod(A.lengths())*A.num_indices();

    T scale(10.0*random_unit<T>());
    scale = 1.0;

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

REPLICATED_TEMPLATED_TEST_CASE(indexed_dpd_trace, R, T, all_types)
{
    indexed_dpd_varray<T> A, B, C, D;
    string idx_A, idx_B;

    random_trace(1000, A, idx_A, B, idx_B);

    INDEXED_DPD_TENSOR_INFO(A);
    INDEXED_DPD_TENSOR_INFO(B);

    auto neps = A.size();

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
