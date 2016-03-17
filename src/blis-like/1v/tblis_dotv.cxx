#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T>
static void tblis_dotv_ref(bool conj_A, bool conj_B, dim_t n,
                           const T* restrict A, inc_t inc_A,
                           const T* restrict B, inc_t inc_B, T& restrict sum)
{
    sum = T();

    conj_B = conj_A ^ conj_B;

    if (inc_A == 1)
    {
        if (conj_B)
        {
            for (dim_t i = 0;i < n;i++)
            {
                sum += A[i]*conj(B[i]);
            }
        }
        else
        {
            for (dim_t i = 0;i < n;i++)
            {
                sum += A[i]*B[i];
            }
        }
    }
    else
    {
        if (conj_B)
        {
            for (dim_t i = 0;i < n;i++)
            {
                sum += (*A)*conj(*B);
                A += inc_A;
                B += inc_B;
            }
        }
        else
        {
            for (dim_t i = 0;i < n;i++)
            {
                sum += (*A)*(*B);
                A += inc_A;
                B += inc_B;
            }
        }
    }

    if (conj_A) sum = conj(sum);
}

template <typename T>
static void tblis_dotv_int(const Matrix<T>& A, const Matrix<T>& B, T& sum)
{
    if (A.length() > 1)
    {
        tblis_dotv_ref(A.is_conjugated(), B.is_conjugated(),
                       A.length(), A.data(), A.row_stride(),
                                   B.data(), B.row_stride(), sum);
    }
    else
    {
        tblis_dotv_ref(A.is_conjugated(), B.is_conjugated(),
                       A.width(), A.data(), A.col_stride(),
                                  B.data(), B.col_stride(), sum);
    }
}

template <typename T>
void tblis_dotv(bool conj_A, bool conj_B, dim_t n,
                const T* A, inc_t inc_A,
                const T* B, inc_t inc_B, T& sum)
{
    if (n == 0) return;
    tblis_dotv_ref(conj_A, conj_B, n, A, inc_A, B, inc_B, sum);
}

template <typename T>
T tblis_dotv(bool conj_A, bool conj_B, dim_t n,
             const T* A, inc_t inc_A,
             const T* B, inc_t inc_B)
{
    T sum;
    tblis_dotv(conj_A, conj_B, n, A, inc_A, B, inc_B, sum);
    return sum;
}

template <typename T>
void tblis_dotv(const Matrix<T>& A, const Matrix<T>& B, T& sum)
{
    Matrix<T> Av;
    Matrix<T> Bv;

    ViewNoTranspose(const_cast<Matrix<T>&>(A), Av);
    ViewNoTranspose(const_cast<Matrix<T>&>(B), Bv);

    ASSERT(A.length() == B.length());
    ASSERT(A.width() == B.width());

    if (A.length() == 0 || A.width() == 0) return;

    tblis_dotv_int(A, B, sum);
}

template <typename T>
T tblis_dotv(const Matrix<T>& A, const Matrix<T>& B)
{
    T sum;
    tblis_dotv(A, B, sum);
    return sum;
}

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void tblis_dotv(bool conj_A, bool conj_B, dim_t n, const T* A, inc_t inc_A, const T* B, inc_t inc_B, T& sum); \
template    T tblis_dotv(bool conj_A, bool conj_B, dim_t n, const T* A, inc_t inc_A, const T* B, inc_t inc_B); \
template void tblis_dotv(const Matrix<T>& A, const Matrix<T>& B, T& sum); \
template    T tblis_dotv(const Matrix<T>& A, const Matrix<T>& B);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

}
}
