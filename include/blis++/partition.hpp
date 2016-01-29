#ifndef _TENSOR_BLIS___PARTITION_HPP_
#define _TENSOR_BLIS___PARTITION_HPP_

#include "matrix.hpp"
#include "scatter_matrix.hpp"

#include <stdexcept>

namespace blis
{

namespace detail
{
    template <typename AbstractMatrix>
    void AssertNotSelfView(const AbstractMatrix& A, const AbstractMatrix& V)
    {
        if (&A == &V && !A.is_view())
            throw std::logic_error("a non-view cannot view itself");
    }

    inline
    void AssertNonNegative(dim_t k)
    {
        if (k < 0)
            throw std::logic_error("parameter must be non-negative");
    }

    template <typename T>
    void AssertLengthCompatible(const Matrix<T>& AL, const Matrix<T>& AR)
    {
        if (AL.length() != AR.length())
            throw std::logic_error("number of rows must match");

        if (AL.row_stride() != AR.row_stride())
            throw std::logic_error("row stride must match");

        if (AL.col_stride() != AR.col_stride())
            throw std::logic_error("column stride must match");

        if ((T*)AL + AL.width()*AL.col_stride() != (T*)AR)
            throw std::logic_error("submatrices must be contiguous");
    }

    template <typename T>
    void AssertWidthCompatible(const Matrix<T>& AT, const Matrix<T>& AB)
    {
        if (AT.width() != AB.width())
            throw std::logic_error("number of columns must match");

        if (AT.row_stride() != AB.row_stride())
            throw std::logic_error("row stride must match");

        if (AT.col_stride() != AB.col_stride())
            throw std::logic_error("column stride must match");

        if ((T*)AT + AT.length()*AT.row_stride() != (T*)AB)
            throw std::logic_error("submatrices must be contiguous");
    }
}

template <typename T>
void View(Matrix<T>& A, Matrix<T>& V)
{
    detail::AssertNotSelfView(A, V);

    V.reset(A.length(), A.width(), A,
            A.row_stride(), A.col_stride());

    if (A.is_transposed()) V.transpose();
    if (A.is_conjugated()) V.conjugate();
}

template <typename T>
void ViewNoTranspose(Matrix<T>& A, Matrix<T>& V)
{
    detail::AssertNotSelfView(A, V);

    if (A.is_conjugated()) V.conjugate();

    if (A.is_transposed())
    {
        V.reset(A.width(), A.length(), A,
                A.col_stride(), A.row_stride());
    }
    else
    {
        V.reset(A.length(), A.width(), A,
                A.row_stride(), A.col_stride());
    }
}

template <typename T>
void ViewNoTranspose(ScatterMatrix<T>& A, ScatterMatrix<T>& V)
{
    detail::AssertNotSelfView(A, V);

    if (A.is_conjugated()) V.conjugate();

    if (A.is_transposed())
    {
        V.reset(A.width(), A.length(), A,
                A.col_stride(), A.row_stride(),
                A.col_scatter(), A.row_scatter());
    }
    else
    {
        V.reset(A.length(), A.width(), A,
                A.row_stride(), A.col_stride(),
                A.row_scatter(), A.col_scatter());
    }
}

template <typename T>
void PartitionTop(dim_t k,               Matrix<T>& AT,
                           /***********/ /************/
                           Matrix<T>& A, Matrix<T>& AB )
{
    detail::AssertNonNegative(k);
    detail::AssertNotSelfView(A, AT);
    detail::AssertNotSelfView(A, AB);

    dim_t m = A.length();
    dim_t n = A.width();
    inc_t rs = A.row_stride();
    inc_t cs = A.col_stride();
    T* p = A;

    k = std::min(m,k);
    m -= k;

    AT.reset(k, n, p     , rs, cs);
    AB.reset(m, n, p+rs*k, rs, cs);
}

template <typename T>
void PartitionBottom(dim_t k, Matrix<T>& A, Matrix<T>& AT,
                              /***********/ /************/
                                            Matrix<T>& AB )
{
    detail::AssertNonNegative(k);
    detail::AssertNotSelfView(A, AT);
    detail::AssertNotSelfView(A, AB);

    dim_t m = A.length();
    dim_t n = A.width();
    inc_t rs = A.row_stride();
    inc_t cs = A.col_stride();
    T* p = A;

    k = std::min(m,k);
    m -= k;

    AT.reset(m, n, p     , rs, cs);
    AB.reset(k, n, p+rs*k, rs, cs);
}

template <typename T>
void PartitionLeft(dim_t k,                /**/ Matrix<T>&  A,
                            Matrix<T>& AL, /**/ Matrix<T>& AR)
{
    detail::AssertNonNegative(k);
    detail::AssertNotSelfView(A, AL);
    detail::AssertNotSelfView(A, AR);

    dim_t m = A.length();
    dim_t n = A.width();
    inc_t rs = A.row_stride();
    inc_t cs = A.col_stride();
    T* p = A;

    k = std::min(n,k);
    n -= k;

    AL.reset(m, k, p     , rs, cs);
    AR.reset(m, n, p+cs*k, rs, cs);
}

template <typename T>
void PartitionRight(dim_t k, Matrix<T>&  A, /**/
                             Matrix<T>& AL, /**/ Matrix<T>& AR)
{
    detail::AssertNonNegative(k);
    detail::AssertNotSelfView(A, AL);
    detail::AssertNotSelfView(A, AR);

    dim_t m = A.length();
    dim_t n = A.width();
    inc_t rs = A.row_stride();
    inc_t cs = A.col_stride();
    T* p = A;

    k = std::min(n,k);
    n -= k;

    AL.reset(m, n, p     , rs, cs);
    AR.reset(m, k, p+cs*n, rs, cs);
}

template <typename T>
void UnpartitionTop(Matrix<T>& AT,
                    /************/ /***********/
                    Matrix<T>& AB, Matrix<T>& A )
{
    detail::AssertWidthCompatible(AT, AB);
    detail::AssertNotSelfView(A, AT);
    detail::AssertNotSelfView(A, AB);

    dim_t k = AT.length();
    dim_t m = AB.length();
    dim_t n = AT.width();
    inc_t rs = AT.row_stride();
    inc_t cs = AT.col_stride();
    T* p = AT;

    A.reset(m+k, n, p, rs, cs);
}

template <typename T>
void UnpartitionBottom(Matrix<T>& AT, Matrix<T>& A,
                       /************/ /***********/
                       Matrix<T>& AB               )
{
    detail::AssertWidthCompatible(AT, AB);
    detail::AssertNotSelfView(A, AT);
    detail::AssertNotSelfView(A, AB);

    dim_t m = AT.length();
    dim_t k = AB.length();
    dim_t n = AT.width();
    inc_t rs = AT.row_stride();
    inc_t cs = AT.col_stride();
    T* p = AT;

    A.reset(m+k, n, p, rs, cs);
}

template <typename T>
void UnpartitionLeft(Matrix<T>& AL, /**/ Matrix<T>& AR,
                                    /**/ Matrix<T>&  A)
{
    detail::AssertLengthCompatible(AL, AR);
    detail::AssertNotSelfView(A, AL);
    detail::AssertNotSelfView(A, AR);

    dim_t m = AL.length();
    dim_t k = AL.width();
    dim_t n = AR.width();
    inc_t rs = AL.row_stride();
    inc_t cs = AL.col_stride();
    T* p = AL;

    A.reset(m, n+k, p, rs, cs);
}

template <typename T>
void UnpartitionRight(Matrix<T>& AL, /**/ Matrix<T>& AR,
                      Matrix<T>&  A  /**/              )
{
    detail::AssertLengthCompatible(AL, AR);
    detail::AssertNotSelfView(A, AL);
    detail::AssertNotSelfView(A, AR);

    dim_t m = AL.length();
    dim_t n = AL.width();
    dim_t k = AR.width();
    inc_t rs = AL.row_stride();
    inc_t cs = AL.col_stride();
    T* p = AL;

    A.reset(m, n+k, p, rs, cs);
}

template <typename T>
void PartitionDown(dim_t k,               Matrix<T>& A0,
                                          Matrix<T>& A1,
                            /***********/ /************/
                            Matrix<T>& A, Matrix<T>& A2 )
{
    detail::AssertNonNegative(k);
    detail::AssertNotSelfView(A, A0);
    detail::AssertNotSelfView(A, A1);
    detail::AssertNotSelfView(A, A2);

    dim_t m = A.length();
    dim_t n = A.width();
    inc_t rs = A.row_stride();
    inc_t cs = A.col_stride();
    T* p = A;

    k = std::min(m,k);
    m -= k;

    A0.reset(0, n, p     , rs, cs);
    A1.reset(k, n, p     , rs, cs);
    A2.reset(m, n, p+rs*k, rs, cs);
}

template <typename T>
void PartitionAcross(dim_t k,                               /**/ Matrix<T>&  A,
                              Matrix<T>& A0, Matrix<T>& A1, /**/ Matrix<T>& A2)
{
    detail::AssertNonNegative(k);
    detail::AssertNotSelfView(A, A0);
    detail::AssertNotSelfView(A, A1);
    detail::AssertNotSelfView(A, A2);

    dim_t m = A.length();
    dim_t n = A.width();
    inc_t rs = A.row_stride();
    inc_t cs = A.col_stride();
    T* p = A;

    k = std::min(n,k);
    n -= k;

    A0.reset(m, 0, p     , rs, cs);
    A1.reset(m, k, p     , rs, cs);
    A2.reset(m, n, p+cs*k, rs, cs);
}

template <typename T>
void SlidePartitionDown(Matrix<T>& A0,
                        Matrix<T>& A1,
                        Matrix<T>& A2)
{
    dim_t k = A1.length();
    UnpartitionBottom(A0, A0,
                      A1    );
    PartitionTop  (k,     A1,
                      A2, A2);
}

template <typename T>
void SlidePartitionAcross(Matrix<T>& A0, Matrix<T>& A1, Matrix<T>& A2)
{
    dim_t k = A1.width();
    UnpartitionRight(A0, A1,
                     A0    );
    PartitionLeft(k,     A2,
                     A1, A2);
}

}

#endif
