#ifndef _TBLIS_IFACE_3T_KHATRI_RAO_H_
#define _TBLIS_IFACE_3T_KHATRI_RAO_H_

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_tensor_khatri_rao(const tblis_comm* comm, const tblis_config* cfg,
                             const tblis_matrix* const * U, const label_type* const * idx_U,
                             tblis_tensor* A, const label_type* idx_A);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

namespace internal
{

template <typename T, size_t N>
std::vector<tblis_matrix> convert(T alpha, const std::array<matrix_view<const T>,N>& U)
{
    std::vector<tblis_matrix> U_s;
    U_s.emplace_back(alpha, U[0]);
    for (unsigned i = 1;i < U.size();i++)
        U_s.emplace_back(U[i]);
    return U_s;
}

template <typename T, size_t N>
std::vector<tblis_matrix> convert(T alpha, const std::array<matrix_view<T>,N>& U)
{
    std::vector<tblis_matrix> U_s;
    U_s.emplace_back(alpha, U[0]);
    for (unsigned i = 1;i < U.size();i++)
        U_s.emplace_back(U[i]);
    return U_s;
}

template <typename T, size_t N>
std::vector<tblis_matrix> convert(T alpha, const std::array<matrix<T>,N>& U)
{
    std::vector<tblis_matrix> U_s;
    U_s.emplace_back(alpha, U[0].view());
    for (unsigned i = 1;i < U.size();i++)
        U_s.emplace_back(U[i].view());
    return U_s;
}

template <typename T>
std::vector<tblis_matrix> convert(T alpha, const std::vector<matrix_view<const T>>& U)
{
    std::vector<tblis_matrix> U_s;
    U_s.emplace_back(alpha, U[0]);
    for (unsigned i = 1;i < U.size();i++)
        U_s.emplace_back(U[i]);
    return U_s;
}

template <typename T>
std::vector<tblis_matrix> convert(T alpha, const std::vector<matrix_view<T>>& U)
{
    std::vector<tblis_matrix> U_s;
    U_s.emplace_back(alpha, U[0]);
    for (unsigned i = 1;i < U.size();i++)
        U_s.emplace_back(U[i]);
    return U_s;
}

template <typename T>
std::vector<tblis_matrix> convert(T alpha, const std::vector<matrix<T>>& U)
{
    std::vector<tblis_matrix> U_s;
    U_s.emplace_back(alpha, U[0].view());
    for (unsigned i = 1;i < U.size();i++)
        U_s.emplace_back(U[i].view());
    return U_s;
}

template <typename T>
std::vector<tblis_matrix> convert(T alpha, std::initializer_list<matrix_view<const T>> U)
{
    std::vector<tblis_matrix> U_s;
    U_s.emplace_back(alpha, U.begin()[0]);
    for (unsigned i = 1;i < U.size();i++)
        U_s.emplace_back(U.begin()[i]);
    return U_s;
}

template <typename T>
std::vector<tblis_matrix> convert(T alpha, std::initializer_list<matrix_view<T>> U)
{
    std::vector<tblis_matrix> U_s;
    U_s.emplace_back(alpha, U.begin()[0]);
    for (unsigned i = 1;i < U.size();i++)
        U_s.emplace_back(U.begin()[i]);
    return U_s;
}

template <typename T>
std::vector<tblis_matrix> convert(T alpha, std::initializer_list<matrix<T>> U)
{
    std::vector<tblis_matrix> U_s;
    U_s.emplace_back(alpha, U.begin()[0].view());
    for (unsigned i = 1;i < U.size();i++)
        U_s.emplace_back(U.begin()[i].view());
    return U_s;
}

}

template <typename T, typename Container, typename IdxContainer>
void khatri_rao(T alpha, const Container& U,
                         const IdxContainer& idx_U,
                T  beta, varray_view<T> A, const label_type* idx_A)
{
    TBLIS_ASSERT(A.dimension() == U.size()+1);

    auto U_s_ = internal::convert(alpha, U);
    tblis_tensor A_s(beta, A);
    std::vector<const tblis_matrix*> U_s(U.size());

    for (unsigned i = 0;i < U.size();i++)
        U_s[i] = &U_s_[i];

    tblis_tensor_khatri_rao(nullptr, nullptr,
                            U_s.data(), idx_U.data(), &A_s, idx_A);
}

template <typename T, typename Container>
void khatri_rao(T alpha, const Container& U,
                T  beta, varray_view<T> A)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(alpha, U);
    tblis_tensor A_s(beta, A);
    std::vector<const tblis_matrix*> U_s(U.size());
    std::vector<label_vector> idx_U_(U.size());
    std::vector<const label_type*> idx_U(U.size());
    label_vector idx_A(U.size()+1, 0);

    for (unsigned i = 0;i < U.size();i++)
    {
        U_s[i] = &U_s_[i];
        idx_U_[i].resize(2, 0);
        idx_U[i] = idx_U_[i].data();
        idx_A[i] = idx_U_[i][0] = i+1;
    }

    tblis_tensor_khatri_rao(nullptr, nullptr,
                            U_s.data(), idx_U.data(), &A_s, idx_A.data());
}

template <typename T, typename Container, typename IdxContainer>
void khatri_rao(const communicator& comm,
                T alpha, const Container& U,
                         const IdxContainer& idx_U,
                T  beta, varray_view<T> A, const label_type* idx_A)
{
    TBLIS_ASSERT(A.dimension() == U.size()+1);

    auto U_s_ = internal::convert(alpha, U);
    tblis_tensor A_s(beta, A);
    std::vector<const tblis_matrix*> U_s(U.size());

    for (unsigned i = 0;i < U.size();i++)
        U_s[i] = &U_s_[i];

    tblis_tensor_khatri_rao(comm, nullptr,
                            U_s.data(), idx_U.data(), &A_s, idx_A);
}

template <typename T, typename Container>
void khatri_rao(const communicator& comm,
                T alpha, const Container& U,
                T  beta, varray_view<T> A)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(alpha, U);
    tblis_tensor A_s(beta, A);
    std::vector<const tblis_matrix*> U_s(U.size());
    std::vector<label_vector> idx_U_(U.size());
    std::vector<const label_type*> idx_U(U.size());
    label_vector idx_A(U.size()+1, 0);

    for (unsigned i = 0;i < U.size();i++)
    {
        U_s[i] = &U_s_[i];
        idx_U_[i].resize(2, 0);
        idx_U[i] = idx_U_[i].data();
        idx_A[i] = idx_U_[i][0] = i+1;
    }

    tblis_tensor_khatri_rao(comm, nullptr,
                            U_s.data(), idx_U.data(), &A_s, idx_A.data());
}

template <typename T>
void khatri_rao(T alpha, std::initializer_list<matrix_view<const T>> U,
                         std::initializer_list<const label_type*> idx_U,
                T  beta, varray_view<T> A, const label_type* idx_A)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(alpha, U);
    tblis_tensor A_s(beta, A);
    std::vector<const tblis_matrix*> U_s(U.size());

    for (unsigned i = 0;i < U.size();i++)
        U_s[i] = &U_s_[i];

    tblis_tensor_khatri_rao(nullptr, nullptr,
                            U_s.data(), idx_U.begin(), &A_s, idx_A);
}

template <typename T>
void khatri_rao(T alpha, std::initializer_list<matrix_view<const T>> U,
                T  beta, varray_view<T> A)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(alpha, U);
    tblis_tensor A_s(beta, A);
    std::vector<const tblis_matrix*> U_s(U.size());
    std::vector<label_vector> idx_U_(U.size());
    std::vector<const label_type*> idx_U(U.size());
    label_vector idx_A(U.size()+1, 0);

    for (unsigned i = 0;i < U.size();i++)
    {
        U_s[i] = &U_s_[i];
        idx_U_[i].resize(2, 0);
        idx_U[i] = idx_U_[i].data();
        idx_A[i] = idx_U_[i][0] = i+1;
    }

    tblis_tensor_khatri_rao(nullptr, nullptr,
                            U_s.data(), idx_U.data(), &A_s, idx_A.data());
}

template <typename T>
void khatri_rao(const communicator& comm,
                T alpha, std::initializer_list<matrix_view<const T>> U,
                         std::initializer_list<const label_type*> idx_U,
                T  beta, varray_view<T> A, const label_type* idx_A)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(alpha, U);
    tblis_tensor A_s(beta, A);
    std::vector<const tblis_matrix*> U_s(U.size());

    for (unsigned i = 0;i < U.size();i++)
        U_s[i] = &U_s_[i];

    tblis_tensor_khatri_rao(comm, nullptr,
                            U_s.data(), idx_U.begin(), &A_s, idx_A);
}

template <typename T>
void khatri_rao(const communicator& comm,
                T alpha, std::initializer_list<matrix_view<const T>> U,
                T  beta, varray_view<T> A)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(alpha, U);
    tblis_tensor A_s(beta, A);
    std::vector<const tblis_matrix*> U_s(U.size());
    std::vector<label_vector> idx_U_(U.size());
    std::vector<const label_type*> idx_U(U.size());
    label_vector idx_A(U.size()+1, 0);

    for (unsigned i = 0;i < U.size();i++)
    {
        U_s[i] = &U_s_[i];
        idx_U_[i].resize(2, 0);
        idx_U[i] = idx_U_[i].data();
        idx_A[i] = idx_U_[i][0] = i+1;
    }

    tblis_tensor_khatri_rao(comm, nullptr,
                            U_s.data(), idx_U.data(), &A_s, idx_A.data());
}

#endif

#ifdef __cplusplus
}
#endif

#endif
