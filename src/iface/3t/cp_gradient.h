#ifndef _TBLIS_IFACE_3T_CP_GRADIENT_H_
#define _TBLIS_IFACE_3T_CP_GRADIENT_H_

#include "khatri_rao.h"

#include "../../util/thread.h"
#include "../../util/basic_types.h"

#ifdef __cplusplus

namespace tblis
{

extern "C"
{

#endif

void tblis_tensor_cp_gradient(const tblis_comm* comm, const tblis_config* cfg,
                              tblis_tensor* A, const label_type* idx_A,
                              const tblis_matrix* const * U, const label_type* const * idx_U,
                              tblis_matrix* G, const label_type* idx_G);

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && !defined(TBLIS_DONT_USE_CXX11)

template <typename T, typename Container, typename IdxContainer>
void cp_gradient(varray_view<const T> A, const label_type* idx_A,
                 const Container& U, const IdxContainer& idx_U,
                 matrix_view<T> G, const label_type* idx_G)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(T(1), U);
    tblis_tensor A_s(A);
    tblis_matrix G_s(G);
    std::vector<const tblis_matrix*> U_s(U.size());

    for (unsigned i = 0;i < U.size();i++)
        U_s[i] = &U_s_[i];

    tblis_tensor_cp_gradient(nullptr, nullptr,
                             &A_s, idx_A,
                             U_s.data(), &*idx_U.begin(),
                             &G_s, idx_G);
}

template <typename T, typename Container>
void cp_gradient(varray_view<const T> A,
                 const Container& U,
                 unsigned dim, matrix_view<T> G)
{
    TBLIS_ASSERT(A.dimension() == U.size());
    TBLIS_ASSERT(dim < U.size());

    auto U_s_ = internal::convert(T(1), U);
    tblis_tensor A_s(A);
    tblis_matrix G_s(G);
    std::vector<const tblis_matrix*> U_s(U.size());
    std::vector<label_vector> idx_U_(U.size());
    std::vector<const label_type*> idx_U(U.size());
    label_vector idx_A(U.size(), 0);
    label_vector idx_G;

    for (unsigned i = 0;i < U.size();i++)
    {
        U_s[i] = &U_s_[i];
        idx_U_[i].resize(2, 0);
        idx_U[i] = idx_U_[i].data();
        idx_A[i] = idx_U_[i][0] = i;

        if (i == dim)
            idx_G = idx_U_[i];
    }

    tblis_tensor_cp_gradient(nullptr, nullptr,
                             &A_s, idx_A.data(),
                             U_s.data(), idx_U.data(),
                             &G_s, idx_G.data());
}

template <typename T, typename Container, typename IdxContainer>
void cp_gradient(const communicator& comm,
                 varray_view<const T> A, const label_type* idx_A,
                 const Container& U, const IdxContainer& idx_U,
                 matrix_view<T> G, const label_type* idx_G)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(T(1), U);
    tblis_tensor A_s(A);
    tblis_matrix G_s(G);
    std::vector<const tblis_matrix*> U_s(U.size());

    for (unsigned i = 0;i < U.size();i++)
        U_s[i] = &U_s_[i];

    tblis_tensor_cp_gradient(comm, nullptr,
                             &A_s, idx_A,
                             U_s.data(), &*idx_U.begin(),
                             &G_s, idx_G);
}

template <typename T, typename Container>
void cp_gradient(const communicator& comm,
                 varray_view<const T> A,
                 const Container& U,
                 unsigned dim, matrix_view<T> G)
{
    TBLIS_ASSERT(A.dimension() == U.size());
    TBLIS_ASSERT(dim < U.size());

    auto U_s_ = internal::convert(T(1), U);
    tblis_tensor A_s(A);
    tblis_matrix G_s(G);
    std::vector<const tblis_matrix*> U_s(U.size());
    std::vector<label_vector> idx_U_(U.size());
    std::vector<const label_type*> idx_U(U.size());
    label_vector idx_A(U.size(), 0);
    label_vector idx_G;

    for (unsigned i = 0;i < U.size();i++)
    {
        U_s[i] = &U_s_[i];
        idx_U_[i].resize(2, 0);
        idx_U[i] = idx_U_[i].data();
        idx_A[i] = idx_U_[i][0] = i;

        if (i == dim)
            idx_G = idx_U_[i];
    }

    tblis_tensor_cp_gradient(comm, nullptr,
                             &A_s, idx_A.data(),
                             U_s.data(), idx_U.data(),
                             &G_s, idx_G.data());
}

template <typename T>
void cp_gradient(varray_view<const T> A, const label_type* idx_A,
                 std::initializer_list<matrix_view<const T>> U,
                 std::initializer_list<const label_type*> idx_U,
                 matrix_view<T> G, const label_type* idx_G)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(T(1), U);
    tblis_tensor A_s(A);
    tblis_matrix G_s(G);
    std::vector<const tblis_matrix*> U_s(U.size());

    for (unsigned i = 0;i < U.size();i++)
        U_s[i] = &U_s_[i];

    tblis_tensor_cp_gradient(nullptr, nullptr,
                             &A_s, idx_A,
                             U_s.data(), &*idx_U.begin(),
                             &G_s, idx_G);
}

template <typename T>
void cp_gradient(varray_view<const T> A,
                 std::initializer_list<matrix_view<const T>> U,
                 unsigned dim, matrix_view<T> G)
{
    TBLIS_ASSERT(A.dimension() == U.size());
    TBLIS_ASSERT(dim < U.size());

    auto U_s_ = internal::convert(T(1), U);
    tblis_tensor A_s(A);
    tblis_matrix G_s(G);
    std::vector<const tblis_matrix*> U_s(U.size());
    std::vector<label_vector> idx_U_(U.size());
    std::vector<const label_type*> idx_U(U.size());
    label_vector idx_A(U.size(), 0);
    label_vector idx_G;

    for (unsigned i = 0;i < U.size();i++)
    {
        U_s[i] = &U_s_[i];
        idx_U_[i].resize(2, 0);
        idx_U[i] = idx_U_[i].data();
        idx_A[i] = idx_U_[i][0] = i;

        if (i == dim)
            idx_G = idx_U_[i];
    }

    tblis_tensor_cp_gradient(nullptr, nullptr,
                             &A_s, idx_A.data(),
                             U_s.data(), idx_U.data(),
                             &G_s, idx_G.data());
}

template <typename T>
void cp_gradient(const communicator& comm,
                 varray_view<const T> A, const label_type* idx_A,
                 std::initializer_list<matrix_view<const T>> U,
                 std::initializer_list<const label_type*> idx_U,
                 matrix_view<T> G, const label_type* idx_G)
{
    TBLIS_ASSERT(A.dimension() == U.size());

    auto U_s_ = internal::convert(T(1), U);
    tblis_tensor A_s(A);
    tblis_matrix G_s(G);
    std::vector<const tblis_matrix*> U_s(U.size());

    for (unsigned i = 0;i < U.size();i++)
        U_s[i] = &U_s_[i];

    tblis_tensor_cp_gradient(comm, nullptr,
                             &A_s, idx_A,
                             U_s.data(), &*idx_U.begin(),
                             &G_s, idx_G);
}

template <typename T>
void cp_gradient(const communicator& comm,
                 varray_view<const T> A,
                 std::initializer_list<matrix_view<const T>> U,
                 unsigned dim, matrix_view<T> G)
{
    TBLIS_ASSERT(A.dimension() == U.size());
    TBLIS_ASSERT(dim < U.size());

    auto U_s_ = internal::convert(T(1), U);
    tblis_tensor A_s(A);
    tblis_matrix G_s(G);
    std::vector<const tblis_matrix*> U_s(U.size());
    std::vector<label_vector> idx_U_(U.size());
    std::vector<const label_type*> idx_U(U.size());
    label_vector idx_A(U.size(), 0);
    label_vector idx_G;

    for (unsigned i = 0;i < U.size();i++)
    {
        U_s[i] = &U_s_[i];
        idx_U_[i].resize(2, 0);
        idx_U[i] = idx_U_[i].data();
        idx_A[i] = idx_U_[i][0] = i;

        if (i == dim)
            idx_G = idx_U_[i];
    }

    tblis_tensor_cp_gradient(comm, nullptr,
                             &A_s, idx_A.data(),
                             U_s.data(), idx_U.data(),
                             &G_s, idx_G.data());
}

#endif

#ifdef __cplusplus
}
#endif

#endif
