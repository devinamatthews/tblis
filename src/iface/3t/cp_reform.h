#ifndef _TBLIS_IFACE_3T_CP_REFORM_H_
#define _TBLIS_IFACE_3T_CP_REFORM_H_

#include "khatri_rao.h"

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_tensor_cp_reform(const tblis_comm* comm, const tblis_config* cfg,
                            const tblis_matrix* const * U, const label_type* const * idx_U,
                            tblis_tensor* A, const label_type* idx_A);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <typename T, typename Container, typename IdxContainer>
void cp_reform(T alpha, const Container& U,
                        const IdxContainer& idx_U,
               T  beta, varray_view<T> A, const label_type* idx_A)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(alpha, U);
    tblis_tensor A_s(beta, A);
    std::vector<const tblis_matrix*> U_s(U.size());

    for (unsigned i = 0;i < U.size();i++)
        U_s[i] = &U_s_[i];

    tblis_tensor_cp_reform(nullptr, nullptr,
                           U_s.data(), idx_U.data(), &A_s, idx_A);
}

template <typename T, typename Container>
void cp_reform(T alpha, const Container& U,
               T  beta, varray_view<T> A)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(alpha, U);
    tblis_tensor A_s(beta, A);
    std::vector<const tblis_matrix*> U_s(U.size());
    std::vector<label_vector> idx_U_(U.size());
    std::vector<const label_type*> idx_U(U.size());
    label_vector idx_A(U.size(), 0);

    for (unsigned i = 0;i < U.size();i++)
    {
        U_s[i] = &U_s_[i];
        idx_U_[i].resize(2, 0);
        idx_U[i] = idx_U_[i].data();
        idx_A[i] = idx_U_[i][0] = i;
    }

    tblis_tensor_cp_reform(nullptr, nullptr,
                           U_s.data(), idx_U.data(), &A_s, idx_A.data());
}

template <typename T, typename Container, typename IdxContainer>
void cp_reform(const communicator& comm,
               T alpha, const Container& U,
                        const IdxContainer& idx_U,
               T  beta, varray_view<T> A, const label_type* idx_A)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(alpha, U);
    tblis_tensor A_s(beta, A);
    std::vector<const tblis_matrix*> U_s(U.size());

    for (unsigned i = 0;i < U.size();i++)
        U_s[i] = &U_s_[i];

    tblis_tensor_cp_reform(comm, nullptr,
                           U_s.data(), idx_U.data(), &A_s, idx_A);
}

template <typename T, typename Container>
void cp_reform(const communicator& comm,
               T alpha, const Container& U,
               T  beta, varray_view<T> A)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(alpha, U);
    tblis_tensor A_s(beta, A);
    std::vector<const tblis_matrix*> U_s(U.size());
    std::vector<label_vector> idx_U_(U.size());
    std::vector<const label_type*> idx_U(U.size());
    label_vector idx_A(U.size(), 0);

    for (unsigned i = 0;i < U.size();i++)
    {
        U_s[i] = &U_s_[i];
        idx_U_[i].resize(2, 0);
        idx_U[i] = idx_U_[i].data();
        idx_A[i] = idx_U_[i][0] = i;
    }

    tblis_tensor_cp_reform(comm, nullptr,
                           U_s.data(), idx_U.data(), &A_s, idx_A.data());
}

template <typename T>
void cp_reform(T alpha, std::initializer_list<matrix_view<const T>> U,
                        std::initializer_list<const label_type*> idx_U,
               T  beta, varray_view<T> A, const label_type* idx_A)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(alpha, U);
    tblis_tensor A_s(beta, A);
    std::vector<const tblis_matrix*> U_s(U.size());

    for (unsigned i = 0;i < U.size();i++)
        U_s[i] = &U_s_[i];

    tblis_tensor_cp_reform(nullptr, nullptr,
                           U_s.data(), idx_U.begin(), &A_s, idx_A);
}

template <typename T>
void cp_reform(T alpha, std::initializer_list<matrix_view<const T>> U,
               T  beta, varray_view<T> A)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(alpha, U);
    tblis_tensor A_s(beta, A);
    std::vector<const tblis_matrix*> U_s(U.size());
    std::vector<label_vector> idx_U_(U.size());
    std::vector<const label_type*> idx_U(U.size());
    label_vector idx_A(U.size(), 0);

    for (unsigned i = 0;i < U.size();i++)
    {
        U_s[i] = &U_s_[i];
        idx_U_[i].resize(2, 0);
        idx_U[i] = idx_U_[i].data();
        idx_A[i] = idx_U_[i][0] = i;
    }

    tblis_tensor_cp_reform(nullptr, nullptr,
                           U_s.data(), idx_U.data(), &A_s, idx_A.data());
}

template <typename T>
void cp_reform(const communicator& comm,
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

    tblis_tensor_cp_reform(comm, nullptr,
                           U_s.data(), idx_U.begin(), &A_s, idx_A);
}

template <typename T>
void cp_reform(const communicator& comm,
               T alpha, std::initializer_list<matrix_view<const T>> U,
               T  beta, varray_view<T> A)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(alpha, U);
    tblis_tensor A_s(beta, A);
    std::vector<const tblis_matrix*> U_s(U.size());
    std::vector<label_vector> idx_U_(U.size());
    std::vector<const label_type*> idx_U(U.size());
    label_vector idx_A(U.size(), 0);

    for (unsigned i = 0;i < U.size();i++)
    {
        U_s[i] = &U_s_[i];
        idx_U_[i].resize(2, 0);
        idx_U[i] = idx_U_[i].data();
        idx_A[i] = idx_U_[i][0] = i;
    }

    tblis_tensor_cp_reform(comm, nullptr,
                           U_s.data(), idx_U.data(), &A_s, idx_A.data());
}

#endif

#ifdef __cplusplus
}
#endif

#endif
