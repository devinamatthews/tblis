#include "tblis.hpp"

namespace tblis
{
namespace blis_like
{

template <typename T>
static void tblis_asumv_ref(dim_t n, const T* restrict A, inc_t inc_A, T& restrict sum)
{
    sum = T();

    if (inc_A == 1)
    {
        for (dim_t i = 0;i < n;i++)
        {
            sum += std::abs(A[i]);
        }
    }
    else
    {
        for (dim_t i = 0;i < n;i++)
        {
            sum += std::abs(*A);
            A += inc_A;
        }
    }
}

template <typename T>
static void tblis_asumv_int(const Matrix<T>& A, T& sum)
{
    if (A.length() > 1)
    {
        tblis_asumv_ref(A.length(), A.data(), A.row_stride(), sum);
    }
    else
    {
        tblis_asumv_ref(A.width(), A.data(), A.col_stride(), sum);
    }
}

template <typename T>
void tblis_asumv(dim_t n, const T* A, inc_t inc_A, T& sum)
{
    if (n == 0) return;
    tblis_asumv_ref(n, A, inc_A, sum);
}

template <typename T>
T tblis_asumv(dim_t n, const T* A, inc_t inc_A)
{
    T sum;
    tblis_asumv(n, A, inc_A, sum);
    return sum;
}

template <typename T>
void tblis_asumv(const Matrix<T>& A, T& sum)
{
    Matrix<T> Av;

    ViewNoTranspose(const_cast<Matrix<T>&>(A), Av);

    if (A.length() == 0 || A.width() == 0) return;

    tblis_asumv_int(A, sum);
}

template <typename T>
T tblis_asumv(const Matrix<T>& A)
{
    T sum;
    tblis_asumv(A, sum);
    return sum;
}

#define INSTANTIATION(T,MT,NT,KT,MR,NR,KR) \
template void tblis_asumv(dim_t n, const T* A, inc_t inc_A, T& sum); \
template    T tblis_asumv(dim_t n, const T* A, inc_t inc_A); \
template void tblis_asumv(const Matrix<T>& A, T& sum); \
template    T tblis_asumv(const Matrix<T>& A);
DEFINE_INSTANTIATIONS()
#undef INSTANTIATION

}
}
