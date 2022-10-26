#include "../test.hpp"

static std::map<reduce_t, string> ops =
{
 {REDUCE_SUM, "REDUCE_SUM"},
 {REDUCE_SUM_ABS, "REDUCE_SUM_ABS"},
 {REDUCE_MAX, "REDUCE_MAX"},
 {REDUCE_MAX_ABS, "REDUCE_MAX_ABS"},
 {REDUCE_MIN, "REDUCE_MIN"},
 {REDUCE_MIN_ABS, "REDUCE_MIN_ABS"},
 {REDUCE_NORM_2, "REDUCE_NORM_2"}
};

template <typename T>
void reduce_ref(reduce_t op, len_type n, const T* A, T& value, len_type& idx)
{
    reduce_init(op, value, idx);

    if (op == REDUCE_MIN ||
        op == REDUCE_MIN_ABS)
        value = -value;

    for (stride_type i = 0;i < n;i++)
    {
        auto tmp = A[i];

        if (op == REDUCE_SUM_ABS ||
            op == REDUCE_MAX_ABS ||
            op == REDUCE_MIN_ABS)
            tmp = std::abs(tmp);

        if (op == REDUCE_NORM_2)
            tmp = norm2(tmp);

        if (op == REDUCE_MIN ||
            op == REDUCE_MIN_ABS)
            tmp = -tmp;

        if (op == REDUCE_SUM ||
            op ==  REDUCE_SUM_ABS ||
            op ==  REDUCE_NORM_2)
            value += tmp;

        if ((op ==  REDUCE_MAX ||
             op ==  REDUCE_MAX_ABS ||
             op ==  REDUCE_MIN ||
             op ==  REDUCE_MIN_ABS) &&
            tmp > value)
        {
            value = tmp;
            idx = i;
        }
    }

    if (op == REDUCE_MIN ||
        op == REDUCE_MIN_ABS)
        value = -value;

    if (op == REDUCE_NORM_2)
        value = sqrt(value);
}

REPLICATED_TEMPLATED_TEST_CASE(reduce, R, T, all_types)
{
    marray<T> A;

    random_tensor(100, A);
    label_vector idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    TENSOR_INFO(A);

    auto NA = prod(A.lengths());

    T ref_val{}, blas_val{};
    stride_type ref_idx{}, blas_idx{};

    T* data = A.data();

    for (auto op : ops)
    {
        reduce(op.first, A, idx_A, ref_val, ref_idx);

        reduce_ref(op.first, NA, data, blas_val, blas_idx);

        check(op.second, ref_idx, blas_idx, ref_val, blas_val, NA);
    }

    A = T(1);
    reduce(REDUCE_SUM, A, idx_A, ref_val, ref_idx);
    check("COUNT", ref_val, NA, NA);
}

REPLICATED_TEMPLATED_TEST_CASE(dpd_reduce, R, T, all_types)
{
    dpd_marray<T> A;

    random_tensor(100, A);
    label_vector idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    DPD_TENSOR_INFO(A);

    auto NA = dpd_marray<T>::size(A.irrep(), A.lengths());

    T ref_val, calc_val;
    stride_type ref_idx, calc_idx;

    for (auto op : ops)
    {
        dpd_impl = dpd_impl_t::FULL;
        reduce<T>(op.first, A, idx_A, ref_val, ref_idx);

        dpd_impl = dpd_impl_t::BLOCKED;
        reduce<T>(op.first, A, idx_A, calc_val, calc_idx);

        check(op.second, ref_idx, calc_idx, ref_val, calc_val, NA);
    }
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_reduce, R, T, all_types)
{
    indexed_marray<T> A;

    random_tensor(100, A);
    label_vector idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    INDEXED_TENSOR_INFO(A);

    auto NA = prod(A.dense_lengths())*A.num_indices();

    T ref_val, calc_val;
    stride_type ref_idx, calc_idx;

    for (auto op : ops)
    {
        dpd_impl = dpd_impl_t::FULL;
        reduce<T>(op.first, A, idx_A, ref_val, ref_idx);

        dpd_impl = dpd_impl_t::BLOCKED;
        reduce<T>(op.first, A, idx_A, calc_val, calc_idx);

        check(op.second, ref_idx, calc_idx, ref_val, calc_val, NA);
    }
}

REPLICATED_TEMPLATED_TEST_CASE(indexed_dpd_reduce, R, T, all_types)
{
    indexed_dpd_marray<T> A;

    random_tensor(100, A);
    label_vector idx_A = range<label_type>('a', static_cast<label_type>('a'+A.dimension()));

    INDEXED_DPD_TENSOR_INFO(A);

    auto NA = dpd_marray<T>::size(A.dense_irrep(), A.dense_lengths())*A.num_indices();

    T ref_val, calc_val;
    stride_type ref_idx, calc_idx;

    for (auto op : ops)
    {
        dpd_impl = dpd_impl_t::FULL;
        reduce<T>(op.first, A, idx_A, ref_val, ref_idx);

        dpd_impl = dpd_impl_t::BLOCKED;
        reduce<T>(op.first, A, idx_A, calc_val, calc_idx);

        check(op.second, ref_idx, calc_idx, ref_val, calc_val, NA);
    }
}
