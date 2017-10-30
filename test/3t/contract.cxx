#include "../test.hpp"

/*
 * Creates a random tensor contraction operation, where each tensor
 * has a storage size of N or fewer elements. All possibilities are sampled
 * uniformly.
 */
template <typename T>
void random_contract(stride_type N, T&& A, label_vector& idx_A,
                                    T&& B, label_vector& idx_B,
                                    T&& C, label_vector& idx_C)
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
    varray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    random_contract(N, A, idx_A, B, idx_B, C, idx_C);

    T scale(10.0*random_unit<T>());

    TENSOR_INFO(A);
    TENSOR_INFO(B);
    TENSOR_INFO(C);

    auto idx_AB = intersection(idx_A, idx_B);
    auto neps = prod(select_from(A.lengths(), idx_A, idx_AB))*prod(C.lengths());

    impl = BLAS_BASED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    impl = REFERENCE;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLAS", error, scale*neps);

    impl = BLIS_BASED;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLIS", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_contract, R, T, all_types)
{
    dpd_varray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

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

    unsigned nirrep = A.num_irreps();
    stride_type neps = 0;
    for (unsigned irrep_AB = 0;irrep_AB < nirrep;irrep_AB++)
    {
        unsigned irrep_AC = A.irrep()^irrep_AB;
        unsigned irrep_BC = B.irrep()^irrep_AB;

        neps += size_AB[irrep_AB]*
                size_AC[irrep_AC]*
                size_BC[irrep_BC];
    }

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    dpd_impl = dpd_impl_t::FULL;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLOCKED", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_contract, R, T, all_types)
{
    indexed_varray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

    T scale(10.0*random_unit<T>());

    random_contract(N, A, idx_A, B, idx_B, C, idx_C);

    INDEXED_TENSOR_INFO(A);
    INDEXED_TENSOR_INFO(B);
    INDEXED_TENSOR_INFO(C);

    auto idx_AB = intersection(idx_A, idx_B);
    auto neps = prod(select_from(A.lengths(), idx_A, idx_AB))*prod(C.lengths());

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    dpd_impl = dpd_impl_t::FULL;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLOCKED", error, scale*neps);
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_dpd_contract, R, T, all_types)
{
    indexed_dpd_varray<T> A, B, C, D, E;
    label_vector idx_A, idx_B, idx_C;

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

    unsigned nirrep = A.num_irreps();
    stride_type neps = 0;
    for (unsigned irrep_AB = 0;irrep_AB < nirrep;irrep_AB++)
    {
        unsigned irrep_AC = A.irrep()^irrep_AB;
        unsigned irrep_BC = B.irrep()^irrep_AB;

        neps += size_AB[irrep_AB]*
                size_AC[irrep_AC]*
                size_BC[irrep_BC];
    }

    dpd_impl = dpd_impl_t::BLOCKED;
    D.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, D, idx_C.data());

    dpd_impl = dpd_impl_t::FULL;
    E.reset(C);
    mult<T>(scale, A, idx_A.data(), B, idx_B.data(), scale, E, idx_C.data());

    add<T>(T(-1), D, idx_C.data(), T(1), E, idx_C.data());
    T error = reduce<T>(REDUCE_NORM_2, E, idx_C.data()).first;

    check("BLOCKED", error, scale*neps);
}
