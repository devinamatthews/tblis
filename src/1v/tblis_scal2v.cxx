#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T>
static void tblis_scal2v_ref(bool conj_A, dim_t n,
                             T alpha, const T* restrict A, inc_t inc_A,
                                            T* restrict B, inc_t inc_B)
{
    if (inc_A == 1 && inc_B == 1)
    {
        if (conj_A)
        {
            for (dim_t i = 0;i < n;i++)
            {
                B[i] = alpha*conj(A[i]);
            }
        }
        else
        {
            for (dim_t i = 0;i < n;i++)
            {
                B[i] = alpha*A[i];
            }
        }
    }
    else
    {
        if (conj_A)
        {
            for (dim_t i = 0;i < n;i++)
            {
                (*B) = alpha*conj(*A);
                A += inc_A;
                B += inc_B;
            }
        }
        else
        {
            for (dim_t i = 0;i < n;i++)
            {
                (*B) = alpha*(*A);
                A += inc_A;
                B += inc_B;
            }
        }
    }
}

template <typename T>
static void tblis_scal2v_int(T alpha, const Matrix<T>& A, Matrix<T>& B)
{
    if (A.length() > 1)
    {
        tblis_scal2v_ref(A.is_conjugated(), A.length(), alpha, A.data(), A.row_stride(),
                                                               B.data(), B.row_stride());
    }
    else
    {
        tblis_scal2v_ref(A.is_conjugated(), A.width(), alpha, A.data(), A.col_stride(),
                                                              B.data(), B.col_stride());
    }
}

template <typename T>
void tblis_scal2v(bool conj_A, dim_t n,
                  T alpha, const T* A, inc_t inc_A,
                                 T* B, inc_t inc_B)
{
    if (alpha == 0)
    {
        tblis_zerov(n, B, inc_B);
    }
    else if (alpha == 1)
    {
        tblis_copyv(conj_A, n, A, inc_A, B, inc_B);
    }

    if (n == 0) return;

    tblis_scal2v_ref(conj_A, n, alpha, A, inc_A, B, inc_B);
}

template <typename T>
void tblis_scal2v(T alpha, const Matrix<T>& A, Matrix<T>& B)
{
    if (alpha == 0)
    {
        tblis_zerov(B);
    }
    else if (alpha == 1)
    {
        tblis_copyv(A, B);
    }

    Matrix<T> Av;
    Matrix<T> Bv;

    ViewNoTranspose(const_cast<Matrix<T>&>(A), Av);
    ViewNoTranspose(                       B , Bv);

    ASSERT(A.length() == B.length());
    ASSERT(A.width() == B.width());

    if (A.length() == 0 || A.width() == 0) return;

    tblis_scal2v_int(alpha, A, B);
}

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void tblis_scal2v(bool conj_A, dim_t n, T alpha, const T* A, inc_t inc_A, T* B, inc_t inc_B); \
template void tblis_scal2v(T alpha, const Matrix<T>& A, Matrix<T>& B);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

}
}
