#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T>
static void tblis_axpbyv_ref(bool conj_A, dim_t n,
                             T alpha, const T* restrict A, inc_t inc_A,
                             T  beta,       T* restrict B, inc_t inc_B)
{
    if (inc_A == 1 && inc_B == 1)
    {
        if (conj_A)
        {
            for (dim_t i = 0;i < n;i++)
            {
                B[i] = alpha*conj(A[i]) + beta*B[i];
            }
        }
        else
        {
            for (dim_t i = 0;i < n;i++)
            {
                B[i] = alpha*A[i] + beta*B[i];
            }
        }
    }
    else
    {
        if (conj_A)
        {
            for (dim_t i = 0;i < n;i++)
            {
                (*B) = alpha*conj(*A) + beta*(*B);
                A += inc_A;
                B += inc_B;
            }
        }
        else
        {
            for (dim_t i = 0;i < n;i++)
            {
                (*B) = alpha*(*A) + beta*(*B);
                A += inc_A;
                B += inc_B;
            }
        }
    }
}

template <typename T>
static void tblis_axpbyv_int(T alpha, const Matrix<T>& A, T beta, Matrix<T>& B)
{
    if (A.length() > 1)
    {
        tblis_axpbyv_ref(A.is_conjugated(), A.length(), alpha, A.data(), A.row_stride(),
                                                         beta, B.data(), B.row_stride());
    }
    else
    {
        tblis_axpbyv_ref(A.is_conjugated(), A.width(), alpha, A.data(), A.col_stride(),
                                                        beta, B.data(), B.col_stride());
    }
}
template <typename T>
void tblis_axpbyv(bool conj_A, dim_t n,
                  T alpha, const T* A, dim_t inc_A,
                  T  beta,       T* B, dim_t inc_B)
{
    if (alpha == 0)
    {
        tblis_scalv(n, beta, B, inc_B);
    }
    else if (alpha == 1)
    {
        tblis_xpbyv(conj_A, n, A, inc_A, beta, B, inc_B);
    }
    else if (beta == 0)
    {
        tblis_scal2v(conj_A, n, alpha, A, inc_A, B, inc_B);
    }
    else if (beta == 1)
    {
        tblis_axpyv(conj_A, n, alpha, A, inc_A, B, inc_B);
    }

    if (n == 0) return;

    tblis_axpbyv_ref(conj_A, n, alpha, A, inc_A, beta, B, inc_B);
}

template <typename T>
void tblis_axpbyv(T alpha, const Matrix<T>& A, T beta, Matrix<T>& B)
{
    if (alpha == 0)
    {
        tblis_scalv(beta, B);
    }
    else if (alpha == 1)
    {
        tblis_xpbyv(A, beta, B);
    }
    else if (beta == 0)
    {
        tblis_scal2v(alpha, A, B);
    }
    else if (beta == 1)
    {
        tblis_axpyv(alpha, A, B);
    }

    Matrix<T> Av;
    Matrix<T> Bv;

    ViewNoTranspose(const_cast<Matrix<T>&>(A), Av);
    ViewNoTranspose(                       B , Bv);

    ASSERT(A.length() == B.length());
    ASSERT(A.width() == B.width());

    if (A.length() == 0 || A.width() == 0) return;

    tblis_axpbyv_int(alpha, A, beta, B);
}

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void tblis_axpbyv(bool conj_A, dim_t n, T alpha, const T* A, dim_t inc_A, T beta, T* B, dim_t inc_B); \
template void tblis_axpbyv(T alpha, const Matrix<T>& A, T beta, Matrix<T>& B);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

}
}
