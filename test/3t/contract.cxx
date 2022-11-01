#include "../test.hpp"

/*
 * Creates a random tensor contraction operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_contract(stride_type N, T&& A, string& idx_A,
                                    T&& B, string& idx_B,
                                    T&& C, string& idx_C)
{
    int ndim_A, ndim_B, ndim_C;
    int ndim_AB, ndim_AC, ndim_BC;

    do
    {
        ndim_A = random_number(1,8);
        ndim_B = random_number(1,8);
        ndim_C = random_number(1,8);
        ndim_AB = (ndim_A+ndim_B-ndim_C)/2;
        ndim_AC = ndim_A-ndim_AB;
        ndim_BC = ndim_B-ndim_AB;
    }
    while (ndim_AB < 0 ||
           ndim_AC < 0 ||
           ndim_BC < 0 ||
           (ndim_A+ndim_B+ndim_C)%2 != 0);

    random_tensors(N,
                   0, 0, 0,
                   ndim_AB, ndim_AC, ndim_BC,
                   0,
                   A, idx_A,
                   B, idx_B,
                   C, idx_C);
}

REPLICATED_TEMPLATED_TEST_CASE(contract, R, T, all_types)
{
    marray<T> A, B, C, D, E;
    string idx_A, idx_B, idx_C;

    random_contract(N, A, idx_A, B, idx_B, C, idx_C);

    T scale(10.0*random_unit<T>());

    TENSOR_INFO(A);
    TENSOR_INFO(B);
    TENSOR_INFO(C);

    auto idx_AB = intersection(idx_A, idx_B);
    auto neps = (prod(select_from(A.lengths(), idx_A, idx_AB))+1)*prod(C.lengths());

    impl = BLAS_BASED;
    D.reset(C);
    mult(scale, A, idx_A, B, idx_B, scale, D, idx_C);

    impl = REFERENCE;
    E.reset(C);
    mult(scale, A, idx_A, B, idx_B, scale, E, idx_C);

    add(-1, D, 1, E);
    T error = reduce<T>(REDUCE_NORM_2, E);

    check("BLAS", error, scale*neps);

    impl = BLIS_BASED;
    E.reset(C);
    mult(scale, A, idx_A, B, idx_B, scale, E, idx_C);

    add(-1, D, 1, E);
    error = reduce<T>(REDUCE_NORM_2, E);

    check("BLIS", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_contract, R, T, all_types)
{
    dpd_marray<T> A, B, C, D, E;
    string idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    random_contract(N, A, idx_A, B, idx_B, C, idx_C);

    DPD_TENSOR_INFO(A);
    DPD_TENSOR_INFO(B);
    DPD_TENSOR_INFO(C);

    auto idx_AB = intersection(idx_A, idx_B);
    auto idx_AC = intersection(idx_A, idx_C);
    auto idx_BC = intersection(idx_B, idx_C);

    auto size_AB = group_size(A.lengths(), idx_A, idx_AB);
    auto size_AC = group_size(A.lengths(), idx_A, idx_AC);
    auto size_BC = group_size(B.lengths(), idx_B, idx_BC);

    auto nirrep = A.num_irreps();
    stride_type neps = 0;
    for (auto irrep_AB : range(nirrep))
    {
        auto irrep_AC = A.irrep()^irrep_AB;
        auto irrep_BC = B.irrep()^irrep_AB;

        neps += (size_AB[irrep_AB]+1)*
                size_AC[irrep_AC]*
                size_BC[irrep_BC];
    }

    dpd_impl = dpd_impl_t::FULL;
    D.reset(C);
    mult<T>(scale, A, idx_A, B, idx_B, scale, D, idx_C);

    dpd_impl = dpd_impl_t::BLOCKED;
    E.reset(C);
    mult<T>(scale, A, idx_A, B, idx_B, scale, E, idx_C);

    add<T>(T(-1), D, idx_C, T(1), E, idx_C);
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C);

    check("BLOCKED", error, scale*neps);

    dpd_impl = dpd_impl_t::BLIS;
    E.reset(C);
    mult<T>(scale, A, idx_A, B, idx_B, scale, E, idx_C);

    add<T>(T(-1), D, idx_C, T(1), E, idx_C);
    error = reduce<T>(REDUCE_NORM_2, E, idx_C);

    check("BLIS", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_contract, R, T, all_types)
{
    indexed_marray<T> A, B, C, D, E;
    string idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    random_contract(N, A, idx_A, B, idx_B, C, idx_C);

    INDEXED_TENSOR_INFO(A);
    INDEXED_TENSOR_INFO(B);
    INDEXED_TENSOR_INFO(C);

    auto idx_AB = intersection(idx_A, idx_B);
    auto neps = (prod(select_from(A.lengths(), idx_A, idx_AB))+1)*prod(C.lengths());

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

REPLICATED_TEMPLATED_TEST_CASE(indexed_dpd_contract, R, T, all_types)
{
    indexed_dpd_marray<T> A, B, C, D, E;
    string idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    random_contract(N, A, idx_A, B, idx_B, C, idx_C);

    INDEXED_DPD_TENSOR_INFO(A);
    INDEXED_DPD_TENSOR_INFO(B);
    INDEXED_DPD_TENSOR_INFO(C);

    auto idx_AB = intersection(idx_A, idx_B);
    auto idx_AC = intersection(idx_A, idx_C);
    auto idx_BC = intersection(idx_B, idx_C);

    auto size_AB = group_size(A.lengths(), idx_A, idx_AB);
    auto size_AC = group_size(A.lengths(), idx_A, idx_AC);
    auto size_BC = group_size(B.lengths(), idx_B, idx_BC);

    auto nirrep = A.num_irreps();
    stride_type neps = 0;
    for (auto irrep_AB : range(nirrep))
    {
        auto irrep_AC = A.irrep()^irrep_AB;
        auto irrep_BC = B.irrep()^irrep_AB;

        neps += (size_AB[irrep_AB]+1)*
                size_AC[irrep_AC]*
                size_BC[irrep_BC];
    }

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
