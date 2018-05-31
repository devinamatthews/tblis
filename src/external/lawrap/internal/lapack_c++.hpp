#ifndef _LAWRAP_INTERNAL_LAPACK_CPP_HPP_
#define _LAWRAP_INTERNAL_LAPACK_CPP_HPP_

#ifndef _LAWRAP_LAPACK_H_
#error "This file must only be included through lapack.h."
#endif

/*
 * C++ wrappers
 */
namespace LAWrap
{

inline integer getrf( integer m, integer n, float* a, integer lda,
                    integer* ipiv )
{
    return c_sgetrf(m, n, a, lda, ipiv);
}

inline integer getrf( integer m, integer n, double* a, integer lda,
                    integer* ipiv )
{
    return c_dgetrf(m, n, a, lda, ipiv);
}

inline integer getrf( integer m, integer n, scomplex* a, integer lda,
                    integer* ipiv )
{
    return c_cgetrf(m, n, a, lda, ipiv);
}

inline integer getrf( integer m, integer n, dcomplex* a, integer lda,
                    integer* ipiv )
{
    return c_zgetrf(m, n, a, lda, ipiv);
}

inline integer gbtrf( integer m, integer n, integer kl,
                    integer ku, float* ab, integer ldab,
                    integer* ipiv )
{
    return c_sgbtrf(m, n, kl, ku, ab, ldab, ipiv);
}

inline integer gbtrf( integer m, integer n, integer kl,
                    integer ku, double* ab, integer ldab,
                    integer* ipiv )
{
    return c_dgbtrf(m, n, kl, ku, ab, ldab, ipiv);
}

inline integer gbtrf( integer m, integer n, integer kl,
                    integer ku, scomplex* ab, integer ldab,
                    integer* ipiv )
{
    return c_cgbtrf(m, n, kl, ku, ab, ldab, ipiv);
}

inline integer gbtrf( integer m, integer n, integer kl,
                    integer ku, dcomplex* ab, integer ldab,
                    integer* ipiv )
{
    return c_zgbtrf(m, n, kl, ku, ab, ldab, ipiv);
}

inline integer gttrf( integer n, float* dl, float* d, float* du,
                    float* du2, integer* ipiv )
{
    return c_sgttrf(n, dl, d, du, du2, ipiv);
}

inline integer gttrf( integer n, double* dl, double* d, double* du,
                    double* du2, integer* ipiv )
{
    return c_dgttrf(n, dl, d, du, du2, ipiv);
}

inline integer gttrf( integer n, scomplex* dl, scomplex* d, scomplex* du,
                    scomplex* du2, integer* ipiv )
{
    return c_cgttrf(n, dl, d, du, du2, ipiv);
}

inline integer gttrf( integer n, dcomplex* dl, dcomplex* d, dcomplex* du,
                    dcomplex* du2, integer* ipiv )
{
    return c_zgttrf(n, dl, d, du, du2, ipiv);
}

inline integer potrf( char uplo, integer n, float* a, integer lda )
{
    return c_spotrf(uplo, n, a, lda);
}

inline integer potrf( char uplo, integer n, double* a, integer lda )
{
    return c_dpotrf(uplo, n, a, lda);
}

inline integer potrf( char uplo, integer n, scomplex* a, integer lda )
{
    return c_cpotrf(uplo, n, a, lda);
}

inline integer potrf( char uplo, integer n, dcomplex* a, integer lda )
{
    return c_zpotrf(uplo, n, a, lda);
}

inline integer pstrf( char uplo, integer n, float* a, integer lda,
                    integer* piv, integer& rank, float tol)
{
    return c_spstrf(uplo, n, a, lda, piv, &rank, tol);
}

inline integer pstrf( char uplo, integer n, double* a, integer lda,
                    integer* piv, integer& rank, double tol)
{
    return c_dpstrf(uplo, n, a, lda, piv, &rank, tol);
}

inline integer pstrf( char uplo, integer n, scomplex* a, integer lda,
                    integer* piv, integer& rank, float tol)
{
    return c_cpstrf(uplo, n, a, lda, piv, &rank, tol);
}

inline integer pstrf( char uplo, integer n, dcomplex* a, integer lda,
                    integer* piv, integer& rank, double tol)
{
    return c_zpstrf(uplo, n, a, lda, piv, &rank, tol);
}

inline integer pftrf( char transr, char uplo, integer n, float* a )
{
    return c_spftrf(transr, uplo, n, a);
}

inline integer pftrf( char transr, char uplo, integer n, double* a )
{
    return c_dpftrf(transr, uplo, n, a);
}

inline integer pftrf( char transr, char uplo, integer n, scomplex* a )
{
    return c_cpftrf(transr, uplo, n, a);
}

inline integer pftrf( char transr, char uplo, integer n, dcomplex* a )
{
    return c_zpftrf(transr, uplo, n, a);
}

inline integer pptrf( char uplo, integer n, float* ap )
{
    return c_spptrf(uplo, n, ap);
}

inline integer pptrf( char uplo, integer n, double* ap )
{
    return c_dpptrf(uplo, n, ap);
}

inline integer pptrf( char uplo, integer n, scomplex* ap )
{
    return c_cpptrf(uplo, n, ap);
}

inline integer pptrf( char uplo, integer n, dcomplex* ap )
{
    return c_zpptrf(uplo, n, ap);
}

inline integer pbtrf( char uplo, integer n, integer kd, float* ab,
                    integer ldab )
{
    return c_spbtrf(uplo, n, kd, ab, ldab);
}

inline integer pbtrf( char uplo, integer n, integer kd, double* ab,
                    integer ldab )
{
    return c_dpbtrf(uplo, n, kd, ab, ldab);
}

inline integer pbtrf( char uplo, integer n, integer kd, scomplex* ab,
                    integer ldab )
{
    return c_cpbtrf(uplo, n, kd, ab, ldab);
}

inline integer pbtrf( char uplo, integer n, integer kd, dcomplex* ab,
                    integer ldab )
{
    return c_zpbtrf(uplo, n, kd, ab, ldab);
}

inline integer pttrf( integer n, float* d, float* e )
{
    return c_spttrf(n, d, e);
}

inline integer pttrf( integer n, double* d, double* e )
{
    return c_dpttrf(n, d, e);
}

inline integer pttrf( integer n, float* d, scomplex* e )
{
    return c_cpttrf(n, d, e);
}

inline integer pttrf( integer n, double* d, dcomplex* e )
{
    return c_zpttrf(n, d, e);
}

inline integer sytrf( char uplo, integer n, float* a, integer lda,
                    integer* ipiv)
{
    return c_ssytrf(uplo, n, a, lda, ipiv);
}

inline integer sytrf( char uplo, integer n, double* a, integer lda,
                    integer* ipiv)
{
    return c_dsytrf(uplo, n, a, lda, ipiv);
}

inline integer sytrf( char uplo, integer n, scomplex* a, integer lda,
                    integer* ipiv)
{
    return c_csytrf(uplo, n, a, lda, ipiv);
}

inline integer sytrf( char uplo, integer n, dcomplex* a, integer lda,
                    integer* ipiv)
{
    return c_zsytrf(uplo, n, a, lda, ipiv);
}

inline integer hetrf( char uplo, integer n, float* a, integer lda,
                    integer* ipiv)
{
    return c_ssytrf(uplo, n, a, lda, ipiv);
}

inline integer hetrf( char uplo, integer n, double* a, integer lda,
                    integer* ipiv)
{
    return c_dsytrf(uplo, n, a, lda, ipiv);
}

inline integer hetrf( char uplo, integer n, scomplex* a, integer lda,
                    integer* ipiv)
{
    return c_chetrf(uplo, n, a, lda, ipiv);
}

inline integer hetrf( char uplo, integer n, dcomplex* a, integer lda,
                    integer* ipiv)
{
    return c_zhetrf(uplo, n, a, lda, ipiv);
}

inline integer sptrf( char uplo, integer n, float* ap, integer* ipiv )
{
    return c_ssptrf(uplo, n, ap, ipiv);
}

inline integer sptrf( char uplo, integer n, double* ap, integer* ipiv )
{
    return c_dsptrf(uplo, n, ap, ipiv);
}

inline integer sptrf( char uplo, integer n, scomplex* ap, integer* ipiv )
{
    return c_csptrf(uplo, n, ap, ipiv);
}

inline integer sptrf( char uplo, integer n, dcomplex* ap, integer* ipiv )
{
    return c_zsptrf(uplo, n, ap, ipiv);
}

inline integer hptrf( char uplo, integer n, float* ap, integer* ipiv )
{
    return c_ssptrf(uplo, n, ap, ipiv);
}

inline integer hptrf( char uplo, integer n, double* ap, integer* ipiv )
{
    return c_dsptrf(uplo, n, ap, ipiv);
}

inline integer hptrf( char uplo, integer n, scomplex* ap, integer* ipiv )
{
    return c_chptrf(uplo, n, ap, ipiv);
}

inline integer hptrf( char uplo, integer n, dcomplex* ap, integer* ipiv )
{
    return c_zhptrf(uplo, n, ap, ipiv);
}

inline integer getrs( char trans, integer n, integer nrhs,
                    const float* a, integer lda, const integer* ipiv,
                    float* b, integer ldb )
{
    return c_sgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer getrs( char trans, integer n, integer nrhs,
                    const double* a, integer lda, const integer* ipiv,
                    double* b, integer ldb )
{
    return c_dgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer getrs( char trans, integer n, integer nrhs,
                    const scomplex* a, integer lda, const integer* ipiv,
                    scomplex* b, integer ldb )
{
    return c_cgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer getrs( char trans, integer n, integer nrhs,
                    const dcomplex* a, integer lda, const integer* ipiv,
                    dcomplex* b, integer ldb )
{
    return c_zgetrs(trans, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer gbtrs( char trans, integer n, integer kl, integer ku,
                    integer nrhs, const float* ab, integer ldab,
                    const integer* ipiv, float* b, integer ldb )
{
    return c_sgbtrs(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
}

inline integer gbtrs( char trans, integer n, integer kl, integer ku,
                    integer nrhs, const double* ab, integer ldab,
                    const integer* ipiv, double* b, integer ldb )
{
    return c_dgbtrs(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
}

inline integer gbtrs( char trans, integer n, integer kl, integer ku,
                    integer nrhs, const scomplex* ab, integer ldab,
                    const integer* ipiv, scomplex* b, integer ldb )
{
    return c_cgbtrs(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
}

inline integer gbtrs( char trans, integer n, integer kl, integer ku,
                    integer nrhs, const dcomplex* ab, integer ldab,
                    const integer* ipiv, dcomplex* b, integer ldb )
{
    return c_zgbtrs(trans, n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
}

inline integer gttrs( char trans, integer n, integer nrhs,
                    const float* dl, const float* d, const float* du,
                    const float* du2, const integer* ipiv, float* b,
                    integer ldb )
{
    return c_sgttrs(trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb);
}

inline integer gttrs( char trans, integer n, integer nrhs,
                    const double* dl, const double* d, const double* du,
                    const double* du2, const integer* ipiv, double* b,
                    integer ldb )
{
    return c_dgttrs(trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb);
}

inline integer gttrs( char trans, integer n, integer nrhs,
                    const scomplex* dl, const scomplex* d, const scomplex* du,
                    const scomplex* du2, const integer* ipiv, scomplex* b,
                    integer ldb )
{
    return c_cgttrs(trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb);
}

inline integer gttrs( char trans, integer n, integer nrhs,
                    const dcomplex* dl, const dcomplex* d, const dcomplex* du,
                    const dcomplex* du2, const integer* ipiv, dcomplex* b,
                    integer ldb )
{
    return c_zgttrs(trans, n, nrhs, dl, d, du, du2, ipiv, b, ldb);
}

inline integer potrs( char uplo, integer n, integer nrhs,
                    const float* a, integer lda, float* b,
                    integer ldb )
{
    return c_spotrs(uplo, n, nrhs, a, lda, b, ldb);
}

inline integer potrs( char uplo, integer n, integer nrhs,
                    const double* a, integer lda, double* b,
                    integer ldb )
{
    return c_dpotrs(uplo, n, nrhs, a, lda, b, ldb);
}

inline integer potrs( char uplo, integer n, integer nrhs,
                    const scomplex* a, integer lda, scomplex* b,
                    integer ldb )
{
    return c_cpotrs(uplo, n, nrhs, a, lda, b, ldb);
}

inline integer potrs( char uplo, integer n, integer nrhs,
                    const dcomplex* a, integer lda, dcomplex* b,
                    integer ldb )
{
    return c_zpotrs(uplo, n, nrhs, a, lda, b, ldb);
}

inline integer pftrs( char transr, char uplo, integer n, integer nrhs,
                    const float* a, float* b, integer ldb )
{
    return c_spftrs(transr, uplo, n, nrhs, a, b, ldb);
}

inline integer pftrs( char transr, char uplo, integer n, integer nrhs,
                    const double* a, double* b, integer ldb )
{
    return c_dpftrs(transr, uplo, n, nrhs, a, b, ldb);
}

inline integer pftrs( char transr, char uplo, integer n, integer nrhs,
                    const scomplex* a, scomplex* b, integer ldb )
{
    return c_cpftrs(transr, uplo, n, nrhs, a, b, ldb);
}

inline integer pftrs( char transr, char uplo, integer n, integer nrhs,
                    const dcomplex* a, dcomplex* b, integer ldb )
{
    return c_zpftrs(transr, uplo, n, nrhs, a, b, ldb);
}

inline integer pptrs( char uplo, integer n, integer nrhs,
                    const float* ap, float* b, integer ldb )
{
    return c_spptrs(uplo, n, nrhs, ap, b, ldb);
}

inline integer pptrs( char uplo, integer n, integer nrhs,
                    const double* ap, double* b, integer ldb )
{
    return c_dpptrs(uplo, n, nrhs, ap, b, ldb);
}

inline integer pptrs( char uplo, integer n, integer nrhs,
                    const scomplex* ap, scomplex* b, integer ldb )
{
    return c_cpptrs(uplo, n, nrhs, ap, b, ldb);
}

inline integer pptrs( char uplo, integer n, integer nrhs,
                    const dcomplex* ap, dcomplex* b, integer ldb )
{
    return c_zpptrs(uplo, n, nrhs, ap, b, ldb);
}

inline integer pbtrs( char uplo, integer n, integer kd, integer nrhs,
                    const float* ab, integer ldab, float* b,
                    integer ldb )
{
    return c_spbtrs(uplo, n, kd, nrhs, ab, ldab, b, ldb);
}

inline integer pbtrs( char uplo, integer n, integer kd, integer nrhs,
                    const double* ab, integer ldab, double* b,
                    integer ldb )
{
    return c_dpbtrs(uplo, n, kd, nrhs, ab, ldab, b, ldb);
}

inline integer pbtrs( char uplo, integer n, integer kd, integer nrhs,
                    const scomplex* ab, integer ldab, scomplex* b,
                    integer ldb )
{
    return c_cpbtrs(uplo, n, kd, nrhs, ab, ldab, b, ldb);
}

inline integer pbtrs( char uplo, integer n, integer kd, integer nrhs,
                    const dcomplex* ab, integer ldab, dcomplex* b,
                    integer ldb )
{
    return c_zpbtrs(uplo, n, kd, nrhs, ab, ldab, b, ldb);
}

inline integer pttrs( integer n, integer nrhs, const float* d,
                    const float* e, float* b, integer ldb )
{
    return c_spttrs(n, nrhs, d, e, b, ldb);
}

inline integer pttrs( integer n, integer nrhs, const double* d,
                    const double* e, double* b, integer ldb )
{
    return c_dpttrs(n, nrhs, d, e, b, ldb);
}

inline integer pttrs( integer n, integer nrhs, const float* d,
                    const scomplex* e, scomplex* b, integer ldb )
{
    return c_cpttrs(n, nrhs, d, e, b, ldb);
}

inline integer pttrs( integer n, integer nrhs, const double* d,
                    const dcomplex* e, dcomplex* b, integer ldb )
{
    return c_zpttrs(n, nrhs, d, e, b, ldb);
}

inline integer sytrs( char uplo, integer n, integer nrhs,
                    const float* a, integer lda, const integer* ipiv,
                    float* b, integer ldb )
{
    return c_ssytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer sytrs( char uplo, integer n, integer nrhs,
                    const double* a, integer lda, const integer* ipiv,
                    double* b, integer ldb )
{
    return c_dsytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer sytrs( char uplo, integer n, integer nrhs,
                    const scomplex* a, integer lda, const integer* ipiv,
                    scomplex* b, integer ldb )
{
    return c_csytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer sytrs( char uplo, integer n, integer nrhs,
                    const dcomplex* a, integer lda, const integer* ipiv,
                    dcomplex* b, integer ldb )
{
    return c_zsytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer hetrs( char uplo, integer n, integer nrhs,
                    const float* a, integer lda, const integer* ipiv,
                    float* b, integer ldb )
{
    return c_ssytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer hetrs( char uplo, integer n, integer nrhs,
                    const double* a, integer lda, const integer* ipiv,
                    double* b, integer ldb )
{
    return c_dsytrs(uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer hetrs( char uplo, integer n, integer nrhs,
                    const scomplex* a, integer lda, const integer* ipiv,
                    scomplex* b, integer ldb )
{
    return c_chetrs(uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer hetrs( char uplo, integer n, integer nrhs,
                    const dcomplex* a, integer lda, const integer* ipiv,
                    dcomplex* b, integer ldb )
{
    return c_zhetrs(uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer sptrs( char uplo, integer n, integer nrhs,
                    const float* ap, const integer* ipiv, float* b,
                    integer ldb )
{
    return c_ssptrs(uplo, n, nrhs, ap, ipiv, b, ldb);
}

inline integer sptrs( char uplo, integer n, integer nrhs,
                    const double* ap, const integer* ipiv, double* b,
                    integer ldb )
{
    return c_dsptrs(uplo, n, nrhs, ap, ipiv, b, ldb);
}

inline integer sptrs( char uplo, integer n, integer nrhs,
                    const scomplex* ap, const integer* ipiv, scomplex* b,
                    integer ldb )
{
    return c_csptrs(uplo, n, nrhs, ap, ipiv, b, ldb);
}

inline integer sptrs( char uplo, integer n, integer nrhs,
                    const dcomplex* ap, const integer* ipiv, dcomplex* b,
                    integer ldb )
{
    return c_zsptrs(uplo, n, nrhs, ap, ipiv, b, ldb);
}

inline integer hptrs( char uplo, integer n, integer nrhs,
                    const float* ap, const integer* ipiv, float* b,
                    integer ldb )
{
    return c_ssptrs(uplo, n, nrhs, ap, ipiv, b, ldb);
}

inline integer hptrs( char uplo, integer n, integer nrhs,
                    const double* ap, const integer* ipiv, double* b,
                    integer ldb )
{
    return c_dsptrs(uplo, n, nrhs, ap, ipiv, b, ldb);
}

inline integer hptrs( char uplo, integer n, integer nrhs,
                    const scomplex* ap, const integer* ipiv, scomplex* b,
                    integer ldb )
{
    return c_chptrs(uplo, n, nrhs, ap, ipiv, b, ldb);
}

inline integer hptrs( char uplo, integer n, integer nrhs,
                    const dcomplex* ap, const integer* ipiv, dcomplex* b,
                    integer ldb )
{
    return c_zhptrs(uplo, n, nrhs, ap, ipiv, b, ldb);
}

inline integer trtrs( char uplo, char trans, char diag, integer n,
                    integer nrhs, const float* a, integer lda,
                    float* b, integer ldb )
{
    return c_strtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb);
}

inline integer trtrs( char uplo, char trans, char diag, integer n,
                    integer nrhs, const double* a, integer lda,
                    double* b, integer ldb )
{
    return c_dtrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb);
}

inline integer trtrs( char uplo, char trans, char diag, integer n,
                    integer nrhs, const scomplex* a, integer lda,
                    scomplex* b, integer ldb )
{
    return c_ctrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb);
}

inline integer trtrs( char uplo, char trans, char diag, integer n,
                    integer nrhs, const dcomplex* a, integer lda,
                    dcomplex* b, integer ldb )
{
    return c_ztrtrs(uplo, trans, diag, n, nrhs, a, lda, b, ldb);
}

inline integer tptrs( char uplo, char trans, char diag, integer n,
                    integer nrhs, const float* ap, float* b,
                    integer ldb )
{
    return c_stptrs(uplo, trans, diag, n, nrhs, ap, b, ldb);
}

inline integer tptrs( char uplo, char trans, char diag, integer n,
                    integer nrhs, const double* ap, double* b,
                    integer ldb )
{
    return c_dtptrs(uplo, trans, diag, n, nrhs, ap, b, ldb);
}

inline integer tptrs( char uplo, char trans, char diag, integer n,
                    integer nrhs, const scomplex* ap, scomplex* b,
                    integer ldb )
{
    return c_ctptrs(uplo, trans, diag, n, nrhs, ap, b, ldb);
}

inline integer tptrs( char uplo, char trans, char diag, integer n,
                    integer nrhs, const dcomplex* ap, dcomplex* b,
                    integer ldb )
{
    return c_ztptrs(uplo, trans, diag, n, nrhs, ap, b, ldb);
}

inline integer tbtrs( char uplo, char trans, char diag, integer n,
                    integer kd, integer nrhs, const float* ab,
                    integer ldab, float* b, integer ldb )
{
    return c_stbtrs(uplo, trans, diag, n, kd, nrhs, ab, ldab, b, ldb);
}

inline integer tbtrs( char uplo, char trans, char diag, integer n,
                    integer kd, integer nrhs, const double* ab,
                    integer ldab, double* b, integer ldb )
{
    return c_dtbtrs(uplo, trans, diag, n, kd, nrhs, ab, ldab, b, ldb);
}

inline integer tbtrs( char uplo, char trans, char diag, integer n,
                    integer kd, integer nrhs, const scomplex* ab,
                    integer ldab, scomplex* b, integer ldb )
{
    return c_ctbtrs(uplo, trans, diag, n, kd, nrhs, ab, ldab, b, ldb);
}

inline integer tbtrs( char uplo, char trans, char diag, integer n,
                    integer kd, integer nrhs, const dcomplex* ab,
                    integer ldab, dcomplex* b, integer ldb )
{
    return c_ztbtrs(uplo, trans, diag, n, kd, nrhs, ab, ldab, b, ldb);
}

inline integer gecon( char norm, integer n, const float* a, integer lda,
                    float anorm, float& rcond)
{
    return c_sgecon(norm, n, a, lda, anorm, &rcond);
}

inline integer gecon( char norm, integer n, const double* a, integer lda,
                    double anorm, double& rcond)
{
    return c_dgecon(norm, n, a, lda, anorm, &rcond);
}

inline integer gecon( char norm, integer n, const scomplex* a, integer lda,
                    float anorm, float& rcond)
{
    return c_cgecon(norm, n, a, lda, anorm, &rcond);
}

inline integer gecon( char norm, integer n, const dcomplex* a, integer lda,
                    double anorm, double& rcond)
{
    return c_zgecon(norm, n, a, lda, anorm, &rcond);
}

inline integer gbcon( char norm, integer n, integer kl, integer ku,
                    const float* ab, integer ldab, const integer* ipiv,
                    float anorm, float& rcond)
{
    return c_sgbcon(norm, n, kl, ku, ab, ldab, ipiv, anorm, &rcond);
}

inline integer gbcon( char norm, integer n, integer kl, integer ku,
                    const double* ab, integer ldab, const integer* ipiv,
                    double anorm, double& rcond)
{
    return c_dgbcon(norm, n, kl, ku, ab, ldab, ipiv, anorm, &rcond);
}

inline integer gbcon( char norm, integer n, integer kl, integer ku,
                    const scomplex* ab, integer ldab, const integer* ipiv,
                    float anorm, float& rcond)
{
    return c_cgbcon(norm, n, kl, ku, ab, ldab, ipiv, anorm, &rcond);
}

inline integer gbcon( char norm, integer n, integer kl, integer ku,
                    const dcomplex* ab, integer ldab, const integer* ipiv,
                    double anorm, double& rcond)
{
    return c_zgbcon(norm, n, kl, ku, ab, ldab, ipiv, anorm, &rcond);
}

inline integer gtcon( char norm, integer n, const float* dl,
                    const float* d, const float* du, const float* du2,
                    const integer* ipiv, float anorm, float& rcond)
{
    return c_sgtcon(norm, n, dl, d, du, du2, ipiv, anorm, &rcond);
}

inline integer gtcon( char norm, integer n, const double* dl,
                    const double* d, const double* du, const double* du2,
                    const integer* ipiv, double anorm, double& rcond)
{
    return c_dgtcon(norm, n, dl, d, du, du2, ipiv, anorm, &rcond);
}

inline integer gtcon( char norm, integer n, const scomplex* dl,
                    const scomplex* d, const scomplex* du, const scomplex* du2,
                    const integer* ipiv, float anorm, float& rcond)
{
    return c_cgtcon(norm, n, dl, d, du, du2, ipiv, anorm, &rcond);
}

inline integer gtcon( char norm, integer n, const dcomplex* dl,
                    const dcomplex* d, const dcomplex* du, const dcomplex* du2,
                    const integer* ipiv, double anorm, double& rcond)
{
    return c_zgtcon(norm, n, dl, d, du, du2, ipiv, anorm, &rcond);
}

inline integer pocon( char uplo, integer n, const float* a, integer lda,
                    float anorm, float& rcond)
{
    return c_spocon(uplo, n, a, lda, anorm, &rcond);
}

inline integer pocon( char uplo, integer n, const double* a, integer lda,
                    double anorm, double& rcond)
{
    return c_dpocon(uplo, n, a, lda, anorm, &rcond);
}

inline integer pocon( char uplo, integer n, const scomplex* a, integer lda,
                    float anorm, float& rcond)
{
    return c_cpocon(uplo, n, a, lda, anorm, &rcond);
}

inline integer pocon( char uplo, integer n, const dcomplex* a, integer lda,
                    double anorm, double& rcond)
{
    return c_zpocon(uplo, n, a, lda, anorm, &rcond);
}

inline integer ppcon( char uplo, integer n, const float* ap, float anorm, float& rcond)
{
    return c_sppcon(uplo, n, ap, anorm, &rcond);
}

inline integer ppcon( char uplo, integer n, const double* ap, double anorm, double& rcond)
{
    return c_dppcon(uplo, n, ap, anorm, &rcond);
}

inline integer ppcon( char uplo, integer n, const scomplex* ap, float anorm, float& rcond)
{
    return c_cppcon(uplo, n, ap, anorm, &rcond);
}

inline integer ppcon( char uplo, integer n, const dcomplex* ap, double anorm, double& rcond)
{
    return c_zppcon(uplo, n, ap, anorm, &rcond);
}

inline integer pbcon( char uplo, integer n, integer kd, const float* ab,
                    integer ldab, float anorm, float& rcond)
{
    return c_spbcon(uplo, n, kd, ab, ldab, anorm, &rcond);
}

inline integer pbcon( char uplo, integer n, integer kd, const double* ab,
                    integer ldab, double anorm, double& rcond)
{
    return c_dpbcon(uplo, n, kd, ab, ldab, anorm, &rcond);
}

inline integer pbcon( char uplo, integer n, integer kd, const scomplex* ab,
                    integer ldab, float anorm, float& rcond)
{
    return c_cpbcon(uplo, n, kd, ab, ldab, anorm, &rcond);
}

inline integer pbcon( char uplo, integer n, integer kd, const dcomplex* ab,
                    integer ldab, double anorm, double& rcond)
{
    return c_zpbcon(uplo, n, kd, ab, ldab, anorm, &rcond);
}

inline integer ptcon( integer n, const float* d, const float* e,
                    float anorm, float& rcond)
{
    return c_sptcon(n, d, e, anorm, &rcond);
}

inline integer ptcon( integer n, const double* d, const double* e,
                    double anorm, double& rcond)
{
    return c_dptcon(n, d, e, anorm, &rcond);
}

inline integer ptcon( integer n, const float* d, const scomplex* e,
                    float anorm, float& rcond)
{
    return c_cptcon(n, d, e, anorm, &rcond);
}

inline integer ptcon( integer n, const double* d, const dcomplex* e,
                    double anorm, double& rcond)
{
    return c_zptcon(n, d, e, anorm, &rcond);
}

inline integer sycon( char uplo, integer n, const float* a, integer lda,
                    const integer* ipiv, float anorm, float& rcond)
{
    return c_ssycon(uplo, n, a, lda, ipiv, anorm, &rcond);
}

inline integer sycon( char uplo, integer n, const double* a, integer lda,
                    const integer* ipiv, double anorm, double& rcond)
{
    return c_dsycon(uplo, n, a, lda, ipiv, anorm, &rcond);
}

inline integer sycon( char uplo, integer n, const scomplex* a, integer lda,
                    const integer* ipiv, float anorm, float& rcond)
{
    return c_csycon(uplo, n, a, lda, ipiv, anorm, &rcond);
}

inline integer sycon( char uplo, integer n, const dcomplex* a, integer lda,
                    const integer* ipiv, double anorm, double& rcond)
{
    return c_zsycon(uplo, n, a, lda, ipiv, anorm, &rcond);
}

inline integer hecon( char uplo, integer n, const float* a, integer lda,
                    const integer* ipiv, float anorm, float& rcond)
{
    return c_ssycon(uplo, n, a, lda, ipiv, anorm, &rcond);
}

inline integer hecon( char uplo, integer n, const double* a, integer lda,
                    const integer* ipiv, double anorm, double& rcond)
{
    return c_dsycon(uplo, n, a, lda, ipiv, anorm, &rcond);
}

inline integer hecon( char uplo, integer n, const scomplex* a, integer lda,
                    const integer* ipiv, float anorm, float& rcond)
{
    return c_checon(uplo, n, a, lda, ipiv, anorm, &rcond);
}

inline integer hecon( char uplo, integer n, const dcomplex* a, integer lda,
                    const integer* ipiv, double anorm, double& rcond)
{
    return c_zhecon(uplo, n, a, lda, ipiv, anorm, &rcond);
}

inline integer spcon( char uplo, integer n, const float* ap,
                    const integer* ipiv, float anorm, float& rcond)
{
    return c_sspcon(uplo, n, ap, ipiv, anorm, &rcond);
}

inline integer spcon( char uplo, integer n, const double* ap,
                    const integer* ipiv, double anorm, double& rcond)
{
    return c_dspcon(uplo, n, ap, ipiv, anorm, &rcond);
}

inline integer spcon( char uplo, integer n, const scomplex* ap,
                    const integer* ipiv, float anorm, float& rcond)
{
    return c_cspcon(uplo, n, ap, ipiv, anorm, &rcond);
}

inline integer spcon( char uplo, integer n, const dcomplex* ap,
                    const integer* ipiv, double anorm, double& rcond)
{
    return c_zspcon(uplo, n, ap, ipiv, anorm, &rcond);
}

inline integer hpcon( char uplo, integer n, const float* ap,
                    const integer* ipiv, float anorm, float& rcond)
{
    return c_sspcon(uplo, n, ap, ipiv, anorm, &rcond);
}

inline integer hpcon( char uplo, integer n, const double* ap,
                    const integer* ipiv, double anorm, double& rcond)
{
    return c_dspcon(uplo, n, ap, ipiv, anorm, &rcond);
}

inline integer hpcon( char uplo, integer n, const scomplex* ap,
                    const integer* ipiv, float anorm, float& rcond)
{
    return c_chpcon(uplo, n, ap, ipiv, anorm, &rcond);
}

inline integer hpcon( char uplo, integer n, const dcomplex* ap,
                    const integer* ipiv, double anorm, double& rcond)
{
    return c_zhpcon(uplo, n, ap, ipiv, anorm, &rcond);
}

inline integer trcon( char norm, char uplo, char diag, integer n,
                    const float* a, integer lda, float& rcond)
{
    return c_strcon(norm, uplo, diag, n, a, lda, &rcond);
}

inline integer trcon( char norm, char uplo, char diag, integer n,
                    const double* a, integer lda, double& rcond)
{
    return c_dtrcon(norm, uplo, diag, n, a, lda, &rcond);
}

inline integer trcon( char norm, char uplo, char diag, integer n,
                    const scomplex* a, integer lda, float& rcond)
{
    return c_ctrcon(norm, uplo, diag, n, a, lda, &rcond);
}

inline integer trcon( char norm, char uplo, char diag, integer n,
                    const dcomplex* a, integer lda, double& rcond)
{
    return c_ztrcon(norm, uplo, diag, n, a, lda, &rcond);
}

inline integer tpcon( char norm, char uplo, char diag, integer n,
                    const float* ap, float& rcond)
{
    return c_stpcon(norm, uplo, diag, n, ap, &rcond);
}

inline integer tpcon( char norm, char uplo, char diag, integer n,
                    const double* ap, double& rcond)
{
    return c_dtpcon(norm, uplo, diag, n, ap, &rcond);
}

inline integer tpcon( char norm, char uplo, char diag, integer n,
                    const scomplex* ap, float& rcond)
{
    return c_ctpcon(norm, uplo, diag, n, ap, &rcond);
}

inline integer tpcon( char norm, char uplo, char diag, integer n,
                    const dcomplex* ap, double& rcond)
{
    return c_ztpcon(norm, uplo, diag, n, ap, &rcond);
}

inline integer tbcon( char norm, char uplo, char diag, integer n,
                    integer kd, const float* ab, integer ldab, float& rcond)
{
    return c_stbcon(norm, uplo, diag, n, kd, ab, ldab, &rcond);
}

inline integer tbcon( char norm, char uplo, char diag, integer n,
                    integer kd, const double* ab, integer ldab, double& rcond)
{
    return c_dtbcon(norm, uplo, diag, n, kd, ab, ldab, &rcond);
}

inline integer tbcon( char norm, char uplo, char diag, integer n,
                    integer kd, const scomplex* ab, integer ldab, float& rcond)
{
    return c_ctbcon(norm, uplo, diag, n, kd, ab, ldab, &rcond);
}

inline integer tbcon( char norm, char uplo, char diag, integer n,
                    integer kd, const dcomplex* ab, integer ldab, double& rcond)
{
    return c_ztbcon(norm, uplo, diag, n, kd, ab, ldab, &rcond);
}

inline integer gerfs( char trans, integer n, integer nrhs,
                    const float* a, integer lda, const float* af,
                    integer ldaf, const integer* ipiv, const float* b,
                    integer ldb, float* x, integer ldx, float* ferr,
                    float* berr)
{
    return c_sgerfs(trans, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer gerfs( char trans, integer n, integer nrhs,
                    const double* a, integer lda, const double* af,
                    integer ldaf, const integer* ipiv, const double* b,
                    integer ldb, double* x, integer ldx, double* ferr,
                    double* berr)
{
    return c_dgerfs(trans, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer gerfs( char trans, integer n, integer nrhs,
                    const scomplex* a, integer lda, const scomplex* af,
                    integer ldaf, const integer* ipiv, const scomplex* b,
                    integer ldb, scomplex* x, integer ldx, float* ferr,
                    float* berr)
{
    return c_cgerfs(trans, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer gerfs( char trans, integer n, integer nrhs,
                    const dcomplex* a, integer lda, const dcomplex* af,
                    integer ldaf, const integer* ipiv, const dcomplex* b,
                    integer ldb, dcomplex* x, integer ldx, double* ferr,
                    double* berr)
{
    return c_zgerfs(trans, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer gerfsx( char trans, char equed, integer n, integer nrhs,
                     const float* a, integer lda, const float* af,
                     integer ldaf, const integer* ipiv, const float* r,
                     const float* c, const float* b, integer ldb,
                     float* x, integer ldx, float& rcond, float* berr,
                     integer n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, integer nparams, float* params)
{
    return c_sgerfsx(trans, equed, n, nrhs, a, lda, af, ldaf, ipiv, r, c, b, ldb, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer gerfsx( char trans, char equed, integer n, integer nrhs,
                     const double* a, integer lda, const double* af,
                     integer ldaf, const integer* ipiv, const double* r,
                     const double* c, const double* b, integer ldb,
                     double* x, integer ldx, double& rcond, double* berr,
                     integer n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, integer nparams, double* params)
{
    return c_dgerfsx(trans, equed, n, nrhs, a, lda, af, ldaf, ipiv, r, c, b, ldb, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer gerfsx( char trans, char equed, integer n, integer nrhs,
                     const scomplex* a, integer lda, const scomplex* af,
                     integer ldaf, const integer* ipiv, const float* r,
                     const float* c, const scomplex* b, integer ldb,
                     scomplex* x, integer ldx, float& rcond, float* berr,
                     integer n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, integer nparams, float* params)
{
    return c_cgerfsx(trans, equed, n, nrhs, a, lda, af, ldaf, ipiv, r, c, b, ldb, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer gerfsx( char trans, char equed, integer n, integer nrhs,
                     const dcomplex* a, integer lda, const dcomplex* af,
                     integer ldaf, const integer* ipiv, const double* r,
                     const double* c, const dcomplex* b, integer ldb,
                     dcomplex* x, integer ldx, double& rcond, double* berr,
                     integer n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, integer nparams, double* params)
{
    return c_zgerfsx(trans, equed, n, nrhs, a, lda, af, ldaf, ipiv, r, c, b, ldb, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer gbrfs( char trans, integer n, integer kl, integer ku,
                    integer nrhs, const float* ab, integer ldab,
                    const float* afb, integer ldafb,
                    const integer* ipiv, const float* b, integer ldb,
                    float* x, integer ldx, float* ferr, float* berr)
{
    return c_sgbrfs(trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer gbrfs( char trans, integer n, integer kl, integer ku,
                    integer nrhs, const double* ab, integer ldab,
                    const double* afb, integer ldafb,
                    const integer* ipiv, const double* b, integer ldb,
                    double* x, integer ldx, double* ferr, double* berr)
{
    return c_dgbrfs(trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer gbrfs( char trans, integer n, integer kl, integer ku,
                    integer nrhs, const scomplex* ab, integer ldab,
                    const scomplex* afb, integer ldafb,
                    const integer* ipiv, const scomplex* b, integer ldb,
                    scomplex* x, integer ldx, float* ferr, float* berr)
{
    return c_cgbrfs(trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer gbrfs( char trans, integer n, integer kl, integer ku,
                    integer nrhs, const dcomplex* ab, integer ldab,
                    const dcomplex* afb, integer ldafb,
                    const integer* ipiv, const dcomplex* b, integer ldb,
                    dcomplex* x, integer ldx, double* ferr, double* berr)
{
    return c_zgbrfs(trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer gbrfsx( char trans, char equed, integer n, integer kl,
                     integer ku, integer nrhs, const float* ab,
                     integer ldab, const float* afb, integer ldafb,
                     const integer* ipiv, const float* r, const float* c,
                     const float* b, integer ldb, float* x,
                     integer ldx, float& rcond, float* berr,
                     integer n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, integer nparams, float* params)
{
    return c_sgbrfsx(trans, equed, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, r, c, b, ldb, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer gbrfsx( char trans, char equed, integer n, integer kl,
                     integer ku, integer nrhs, const double* ab,
                     integer ldab, const double* afb, integer ldafb,
                     const integer* ipiv, const double* r, const double* c,
                     const double* b, integer ldb, double* x,
                     integer ldx, double& rcond, double* berr,
                     integer n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, integer nparams, double* params)
{
    return c_dgbrfsx(trans, equed, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, r, c, b, ldb, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer gbrfsx( char trans, char equed, integer n, integer kl,
                     integer ku, integer nrhs, const scomplex* ab,
                     integer ldab, const scomplex* afb, integer ldafb,
                     const integer* ipiv, const float* r, const float* c,
                     const scomplex* b, integer ldb, scomplex* x,
                     integer ldx, float& rcond, float* berr,
                     integer n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, integer nparams, float* params)
{
    return c_cgbrfsx(trans, equed, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, r, c, b, ldb, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer gbrfsx( char trans, char equed, integer n, integer kl,
                     integer ku, integer nrhs, const dcomplex* ab,
                     integer ldab, const dcomplex* afb, integer ldafb,
                     const integer* ipiv, const double* r, const double* c,
                     const dcomplex* b, integer ldb, dcomplex* x,
                     integer ldx, double& rcond, double* berr,
                     integer n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, integer nparams, double* params)
{
    return c_zgbrfsx(trans, equed, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, r, c, b, ldb, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer gtrfs( char trans, integer n, integer nrhs,
                    const float* dl, const float* d, const float* du,
                    const float* dlf, const float* df, const float* duf,
                    const float* du2, const integer* ipiv, const float* b,
                    integer ldb, float* x, integer ldx, float* ferr,
                    float* berr)
{
    return c_sgtrfs(trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer gtrfs( char trans, integer n, integer nrhs,
                    const double* dl, const double* d, const double* du,
                    const double* dlf, const double* df, const double* duf,
                    const double* du2, const integer* ipiv, const double* b,
                    integer ldb, double* x, integer ldx, double* ferr,
                    double* berr)
{
    return c_dgtrfs(trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer gtrfs( char trans, integer n, integer nrhs,
                    const scomplex* dl, const scomplex* d, const scomplex* du,
                    const scomplex* dlf, const scomplex* df, const scomplex* duf,
                    const scomplex* du2, const integer* ipiv, const scomplex* b,
                    integer ldb, scomplex* x, integer ldx, float* ferr,
                    float* berr)
{
    return c_cgtrfs(trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer gtrfs( char trans, integer n, integer nrhs,
                    const dcomplex* dl, const dcomplex* d, const dcomplex* du,
                    const dcomplex* dlf, const dcomplex* df, const dcomplex* duf,
                    const dcomplex* du2, const integer* ipiv, const dcomplex* b,
                    integer ldb, dcomplex* x, integer ldx, double* ferr,
                    double* berr)
{
    return c_zgtrfs(trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer porfs( char uplo, integer n, integer nrhs,
                    const float* a, integer lda, const float* af,
                    integer ldaf, const float* b, integer ldb,
                    float* x, integer ldx, float* ferr, float* berr)
{
    return c_sporfs(uplo, n, nrhs, a, lda, af, ldaf, b, ldb, x, ldx, ferr, berr);
}

inline integer porfs( char uplo, integer n, integer nrhs,
                    const double* a, integer lda, const double* af,
                    integer ldaf, const double* b, integer ldb,
                    double* x, integer ldx, double* ferr, double* berr)
{
    return c_dporfs(uplo, n, nrhs, a, lda, af, ldaf, b, ldb, x, ldx, ferr, berr);
}

inline integer porfs( char uplo, integer n, integer nrhs,
                    const scomplex* a, integer lda, const scomplex* af,
                    integer ldaf, const scomplex* b, integer ldb,
                    scomplex* x, integer ldx, float* ferr, float* berr)
{
    return c_cporfs(uplo, n, nrhs, a, lda, af, ldaf, b, ldb, x, ldx, ferr, berr);
}

inline integer porfs( char uplo, integer n, integer nrhs,
                    const dcomplex* a, integer lda, const dcomplex* af,
                    integer ldaf, const dcomplex* b, integer ldb,
                    dcomplex* x, integer ldx, double* ferr, double* berr)
{
    return c_zporfs(uplo, n, nrhs, a, lda, af, ldaf, b, ldb, x, ldx, ferr, berr);
}

inline integer porfsx( char uplo, char equed, integer n, integer nrhs,
                     const float* a, integer lda, const float* af,
                     integer ldaf, const float* s, const float* b,
                     integer ldb, float* x, integer ldx, float& rcond,
                     float* berr, integer n_err_bnds,
                     float* err_bnds_norm, float* err_bnds_comp,
                     integer nparams, float* params)
{
    return c_sporfsx(uplo, equed, n, nrhs, a, lda, af, ldaf, s, b, ldx, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer porfsx( char uplo, char equed, integer n, integer nrhs,
                     const double* a, integer lda, const double* af,
                     integer ldaf, const double* s, const double* b,
                     integer ldb, double* x, integer ldx, double& rcond,
                     double* berr, integer n_err_bnds,
                     double* err_bnds_norm, double* err_bnds_comp,
                     integer nparams, double* params)
{
    return c_dporfsx(uplo, equed, n, nrhs, a, lda, af, ldaf, s, b, ldx, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer porfsx( char uplo, char equed, integer n, integer nrhs,
                     const scomplex* a, integer lda, const scomplex* af,
                     integer ldaf, const float* s, const scomplex* b,
                     integer ldb, scomplex* x, integer ldx, float& rcond,
                     float* berr, integer n_err_bnds,
                     float* err_bnds_norm, float* err_bnds_comp,
                     integer nparams, float* params)
{
    return c_cporfsx(uplo, equed, n, nrhs, a, lda, af, ldaf, s, b, ldx, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer porfsx( char uplo, char equed, integer n, integer nrhs,
                     const dcomplex* a, integer lda, const dcomplex* af,
                     integer ldaf, const double* s, const dcomplex* b,
                     integer ldb, dcomplex* x, integer ldx, double& rcond,
                     double* berr, integer n_err_bnds,
                     double* err_bnds_norm, double* err_bnds_comp,
                     integer nparams, double* params)
{
    return c_zporfsx(uplo, equed, n, nrhs, a, lda, af, ldaf, s, b, ldx, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer pprfs( char uplo, integer n, integer nrhs,
                    const float* ap, const float* afp, const float* b,
                    integer ldb, float* x, integer ldx, float* ferr,
                    float* berr)
{
    return c_spprfs(uplo, n, nrhs, ap, afp, b, ldb, x, ldx, ferr, berr);
}

inline integer pprfs( char uplo, integer n, integer nrhs,
                    const double* ap, const double* afp, const double* b,
                    integer ldb, double* x, integer ldx, double* ferr,
                    double* berr)
{
    return c_dpprfs(uplo, n, nrhs, ap, afp, b, ldb, x, ldx, ferr, berr);
}

inline integer pprfs( char uplo, integer n, integer nrhs,
                    const scomplex* ap, const scomplex* afp, const scomplex* b,
                    integer ldb, scomplex* x, integer ldx, float* ferr,
                    float* berr)
{
    return c_cpprfs(uplo, n, nrhs, ap, afp, b, ldb, x, ldx, ferr, berr);
}

inline integer pprfs( char uplo, integer n, integer nrhs,
                    const dcomplex* ap, const dcomplex* afp, const dcomplex* b,
                    integer ldb, dcomplex* x, integer ldx, double* ferr,
                    double* berr)
{
    return c_zpprfs(uplo, n, nrhs, ap, afp, b, ldb, x, ldx, ferr, berr);
}

inline integer pbrfs( char uplo, integer n, integer kd, integer nrhs,
                    const float* ab, integer ldab, const float* afb,
                    integer ldafb, const float* b, integer ldb,
                    float* x, integer ldx, float* ferr, float* berr)
{
    return c_spbrfs(uplo, n, kd, nrhs, ab, ldab, afb, ldafb, b, ldb, x, ldx, ferr, berr);
}

inline integer pbrfs( char uplo, integer n, integer kd, integer nrhs,
                    const double* ab, integer ldab, const double* afb,
                    integer ldafb, const double* b, integer ldb,
                    double* x, integer ldx, double* ferr, double* berr)
{
    return c_dpbrfs(uplo, n, kd, nrhs, ab, ldab, afb, ldafb, b, ldb, x, ldx, ferr, berr);
}

inline integer pbrfs( char uplo, integer n, integer kd, integer nrhs,
                    const scomplex* ab, integer ldab, const scomplex* afb,
                    integer ldafb, const scomplex* b, integer ldb,
                    scomplex* x, integer ldx, float* ferr, float* berr)
{
    return c_cpbrfs(uplo, n, kd, nrhs, ab, ldab, afb, ldafb, b, ldb, x, ldx, ferr, berr);
}

inline integer pbrfs( char uplo, integer n, integer kd, integer nrhs,
                    const dcomplex* ab, integer ldab, const dcomplex* afb,
                    integer ldafb, const dcomplex* b, integer ldb,
                    dcomplex* x, integer ldx, double* ferr, double* berr)
{
    return c_zpbrfs(uplo, n, kd, nrhs, ab, ldab, afb, ldafb, b, ldb, x, ldx, ferr, berr);
}

inline integer ptrfs( integer n, integer nrhs, const float* d,
                    const float* e, const float* df, const float* ef,
                    const float* b, integer ldb, float* x,
                    integer ldx, float* ferr, float* berr)
{
    return c_sptrfs(n, nrhs, d, e, df, ef, b, ldb, x, ldx, ferr, berr);
}

inline integer ptrfs( integer n, integer nrhs, const double* d,
                    const double* e, const double* df, const double* ef,
                    const double* b, integer ldb, double* x,
                    integer ldx, double* ferr, double* berr)
{
    return c_dptrfs(n, nrhs, d, e, df, ef, b, ldb, x, ldx, ferr, berr);
}

inline integer ptrfs( integer n, integer nrhs, const float* d,
                    const scomplex* e, const float* df, const scomplex* ef,
                    const scomplex* b, integer ldb, scomplex* x,
                    integer ldx, float* ferr, float* berr)
{
    return c_cptrfs(n, nrhs, d, e, df, ef, b, ldb, x, ldx, ferr, berr);
}

inline integer ptrfs( integer n, integer nrhs, const double* d,
                    const dcomplex* e, const double* df, const dcomplex* ef,
                    const dcomplex* b, integer ldb, dcomplex* x,
                    integer ldx, double* ferr, double* berr)
{
    return c_zptrfs(n, nrhs, d, e, df, ef, b, ldb, x, ldx, ferr, berr);
}

inline integer syrfs( char uplo, integer n, integer nrhs,
                    const float* a, integer lda, const float* af,
                    integer ldaf, const integer* ipiv, const float* b,
                    integer ldb, float* x, integer ldx, float* ferr,
                    float* berr)
{
    return c_ssyrfs(uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer syrfs( char uplo, integer n, integer nrhs,
                    const double* a, integer lda, const double* af,
                    integer ldaf, const integer* ipiv, const double* b,
                    integer ldb, double* x, integer ldx, double* ferr,
                    double* berr)
{
    return c_dsyrfs(uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer syrfs( char uplo, integer n, integer nrhs,
                    const scomplex* a, integer lda, const scomplex* af,
                    integer ldaf, const integer* ipiv, const scomplex* b,
                    integer ldb, scomplex* x, integer ldx, float* ferr,
                    float* berr)
{
    return c_csyrfs(uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer syrfs( char uplo, integer n, integer nrhs,
                    const dcomplex* a, integer lda, const dcomplex* af,
                    integer ldaf, const integer* ipiv, const dcomplex* b,
                    integer ldb, dcomplex* x, integer ldx, double* ferr,
                    double* berr)
{
    return c_zsyrfs(uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer herfs( char uplo, integer n, integer nrhs,
                    const float* a, integer lda, const float* af,
                    integer ldaf, const integer* ipiv, const float* b,
                    integer ldb, float* x, integer ldx, float* ferr,
                    float* berr)
{
    return c_ssyrfs(uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer herfs( char uplo, integer n, integer nrhs,
                    const double* a, integer lda, const double* af,
                    integer ldaf, const integer* ipiv, const double* b,
                    integer ldb, double* x, integer ldx, double* ferr,
                    double* berr)
{
    return c_dsyrfs(uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer herfs( char uplo, integer n, integer nrhs,
                    const scomplex* a, integer lda, const scomplex* af,
                    integer ldaf, const integer* ipiv, const scomplex* b,
                    integer ldb, scomplex* x, integer ldx, float* ferr,
                    float* berr)
{
    return c_cherfs(uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer herfs( char uplo, integer n, integer nrhs,
                    const dcomplex* a, integer lda, const dcomplex* af,
                    integer ldaf, const integer* ipiv, const dcomplex* b,
                    integer ldb, dcomplex* x, integer ldx, double* ferr,
                    double* berr)
{
    return c_zherfs(uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer syrfsx( char uplo, char equed, integer n, integer nrhs,
                     const float* a, integer lda, const float* af,
                     integer ldaf, const integer* ipiv, const float* s,
                     const float* b, integer ldb, float* x,
                     integer ldx, float& rcond, float* berr,
                     integer n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, integer nparams, float* params)
{
    return c_ssyrfsx(uplo, equed, n, nrhs, a, lda, af, ldaf, ipiv, s, b, ldb, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer syrfsx( char uplo, char equed, integer n, integer nrhs,
                     const double* a, integer lda, const double* af,
                     integer ldaf, const integer* ipiv, const double* s,
                     const double* b, integer ldb, double* x,
                     integer ldx, double& rcond, double* berr,
                     integer n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, integer nparams, double* params)
{
    return c_dsyrfsx(uplo, equed, n, nrhs, a, lda, af, ldaf, ipiv, s, b, ldb, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer syrfsx( char uplo, char equed, integer n, integer nrhs,
                     const scomplex* a, integer lda, const scomplex* af,
                     integer ldaf, const integer* ipiv, const float* s,
                     const scomplex* b, integer ldb, scomplex* x,
                     integer ldx, float& rcond, float* berr,
                     integer n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, integer nparams, float* params)
{
    return c_csyrfsx(uplo, equed, n, nrhs, a, lda, af, ldaf, ipiv, s, b, ldb, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer syrfsx( char uplo, char equed, integer n, integer nrhs,
                     const dcomplex* a, integer lda, const dcomplex* af,
                     integer ldaf, const integer* ipiv, const double* s,
                     const dcomplex* b, integer ldb, dcomplex* x,
                     integer ldx, double& rcond, double* berr,
                     integer n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, integer nparams, double* params)
{
    return c_zsyrfsx(uplo, equed, n, nrhs, a, lda, af, ldaf, ipiv, s, b, ldb, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer herfsx( char uplo, char equed, integer n, integer nrhs,
                     const float* a, integer lda, const float* af,
                     integer ldaf, const integer* ipiv, const float* s,
                     const float* b, integer ldb, float* x,
                     integer ldx, float& rcond, float* berr,
                     integer n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, integer nparams, float* params)
{
    return c_ssyrfsx(uplo, equed, n, nrhs, a, lda, af, ldaf, ipiv, s, b, ldb, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer herfsx( char uplo, char equed, integer n, integer nrhs,
                     const double* a, integer lda, const double* af,
                     integer ldaf, const integer* ipiv, const double* s,
                     const double* b, integer ldb, double* x,
                     integer ldx, double& rcond, double* berr,
                     integer n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, integer nparams, double* params)
{
    return c_dsyrfsx(uplo, equed, n, nrhs, a, lda, af, ldaf, ipiv, s, b, ldb, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer herfsx( char uplo, char equed, integer n, integer nrhs,
                     const scomplex* a, integer lda, const scomplex* af,
                     integer ldaf, const integer* ipiv, const float* s,
                     const scomplex* b, integer ldb, scomplex* x,
                     integer ldx, float& rcond, float* berr,
                     integer n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, integer nparams, float* params)
{
    return c_cherfsx(uplo, equed, n, nrhs, a, lda, af, ldaf, ipiv, s, b, ldb, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer herfsx( char uplo, char equed, integer n, integer nrhs,
                     const dcomplex* a, integer lda, const dcomplex* af,
                     integer ldaf, const integer* ipiv, const double* s,
                     const dcomplex* b, integer ldb, dcomplex* x,
                     integer ldx, double& rcond, double* berr,
                     integer n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, integer nparams, double* params)
{
    return c_zherfsx(uplo, equed, n, nrhs, a, lda, af, ldaf, ipiv, s, b, ldb, x, ldx,
                   &rcond, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer sprfs( char uplo, integer n, integer nrhs,
                    const float* ap, const float* afp, const integer* ipiv,
                    const float* b, integer ldb, float* x,
                    integer ldx, float* ferr, float* berr)
{
    return c_ssprfs(uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer sprfs( char uplo, integer n, integer nrhs,
                    const double* ap, const double* afp, const integer* ipiv,
                    const double* b, integer ldb, double* x,
                    integer ldx, double* ferr, double* berr)
{
    return c_dsprfs(uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer sprfs( char uplo, integer n, integer nrhs,
                    const scomplex* ap, const scomplex* afp, const integer* ipiv,
                    const scomplex* b, integer ldb, scomplex* x,
                    integer ldx, float* ferr, float* berr)
{
    return c_csprfs(uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer sprfs( char uplo, integer n, integer nrhs,
                    const dcomplex* ap, const dcomplex* afp, const integer* ipiv,
                    const dcomplex* b, integer ldb, dcomplex* x,
                    integer ldx, double* ferr, double* berr)
{
    return c_zsprfs(uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer hprfs( char uplo, integer n, integer nrhs,
                    const float* ap, const float* afp, const integer* ipiv,
                    const float* b, integer ldb, float* x,
                    integer ldx, float* ferr, float* berr)
{
    return c_ssprfs(uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer hprfs( char uplo, integer n, integer nrhs,
                    const double* ap, const double* afp, const integer* ipiv,
                    const double* b, integer ldb, double* x,
                    integer ldx, double* ferr, double* berr)
{
    return c_dsprfs(uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer hprfs( char uplo, integer n, integer nrhs,
                    const scomplex* ap, const scomplex* afp, const integer* ipiv,
                    const scomplex* b, integer ldb, scomplex* x,
                    integer ldx, float* ferr, float* berr)
{
    return c_chprfs(uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer hprfs( char uplo, integer n, integer nrhs,
                    const dcomplex* ap, const dcomplex* afp, const integer* ipiv,
                    const dcomplex* b, integer ldb, dcomplex* x,
                    integer ldx, double* ferr, double* berr)
{
    return c_zhprfs(uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, ferr, berr);
}

inline integer trrfs( char uplo, char trans, char diag, integer n,
                    integer nrhs, const float* a, integer lda,
                    const float* b, integer ldb, const float* x,
                    integer ldx, float* ferr, float* berr)
{
    return c_strrfs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, x, ldx, ferr, berr);
}

inline integer trrfs( char uplo, char trans, char diag, integer n,
                    integer nrhs, const double* a, integer lda,
                    const double* b, integer ldb, const double* x,
                    integer ldx, double* ferr, double* berr)
{
    return c_dtrrfs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, x, ldx, ferr, berr);
}

inline integer trrfs( char uplo, char trans, char diag, integer n,
                    integer nrhs, const scomplex* a, integer lda,
                    const scomplex* b, integer ldb, const scomplex* x,
                    integer ldx, float* ferr, float* berr)
{
    return c_ctrrfs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, x, ldx, ferr, berr);
}

inline integer trrfs( char uplo, char trans, char diag, integer n,
                    integer nrhs, const dcomplex* a, integer lda,
                    const dcomplex* b, integer ldb, const dcomplex* x,
                    integer ldx, double* ferr, double* berr)
{
    return c_ztrrfs(uplo, trans, diag, n, nrhs, a, lda, b, ldb, x, ldx, ferr, berr);
}

inline integer tprfs( char uplo, char trans, char diag, integer n,
                    integer nrhs, const float* ap, const float* b,
                    integer ldb, const float* x, integer ldx,
                    float* ferr, float* berr)
{
    return c_stprfs(uplo, trans, diag, n, nrhs, ap, b, ldb, x, ldx, ferr, berr);
}

inline integer tprfs( char uplo, char trans, char diag, integer n,
                    integer nrhs, const double* ap, const double* b,
                    integer ldb, const double* x, integer ldx,
                    double* ferr, double* berr)
{
    return c_dtprfs(uplo, trans, diag, n, nrhs, ap, b, ldb, x, ldx, ferr, berr);
}

inline integer tprfs( char uplo, char trans, char diag, integer n,
                    integer nrhs, const scomplex* ap, const scomplex* b,
                    integer ldb, const scomplex* x, integer ldx,
                    float* ferr, float* berr)
{
    return c_ctprfs(uplo, trans, diag, n, nrhs, ap, b, ldb, x, ldx, ferr, berr);
}

inline integer tprfs( char uplo, char trans, char diag, integer n,
                    integer nrhs, const dcomplex* ap, const dcomplex* b,
                    integer ldb, const dcomplex* x, integer ldx,
                    double* ferr, double* berr)
{
    return c_ztprfs(uplo, trans, diag, n, nrhs, ap, b, ldb, x, ldx, ferr, berr);
}

inline integer tbrfs( char uplo, char trans, char diag, integer n,
                    integer kd, integer nrhs, const float* ab,
                    integer ldab, const float* b, integer ldb,
                    const float* x, integer ldx, float* ferr,
                    float* berr)
{
    return c_stbrfs(uplo, trans, diag, n, kd, nrhs, ab, ldab, b, ldb, x, ldx, ferr, berr);
}

inline integer tbrfs( char uplo, char trans, char diag, integer n,
                    integer kd, integer nrhs, const double* ab,
                    integer ldab, const double* b, integer ldb,
                    const double* x, integer ldx, double* ferr,
                    double* berr)
{
    return c_dtbrfs(uplo, trans, diag, n, kd, nrhs, ab, ldab, b, ldb, x, ldx, ferr, berr);
}

inline integer tbrfs( char uplo, char trans, char diag, integer n,
                    integer kd, integer nrhs, const scomplex* ab,
                    integer ldab, const scomplex* b, integer ldb,
                    const scomplex* x, integer ldx, float* ferr,
                    float* berr)
{
    return c_ctbrfs(uplo, trans, diag, n, kd, nrhs, ab, ldab, b, ldb, x, ldx, ferr, berr);
}

inline integer tbrfs( char uplo, char trans, char diag, integer n,
                    integer kd, integer nrhs, const dcomplex* ab,
                    integer ldab, const dcomplex* b, integer ldb,
                    const dcomplex* x, integer ldx, double* ferr,
                    double* berr)
{
    return c_ztbrfs(uplo, trans, diag, n, kd, nrhs, ab, ldab, b, ldb, x, ldx, ferr, berr);
}

inline integer getri( integer n, float* a, integer lda,
                    const integer* ipiv)
{
    return c_sgetri(n, a, lda, ipiv);
}

inline integer getri( integer n, double* a, integer lda,
                    const integer* ipiv)
{
    return c_dgetri(n, a, lda, ipiv);
}

inline integer getri( integer n, scomplex* a, integer lda,
                    const integer* ipiv)
{
    return c_cgetri(n, a, lda, ipiv);
}

inline integer getri( integer n, dcomplex* a, integer lda,
                    const integer* ipiv)
{
    return c_zgetri(n, a, lda, ipiv);
}

inline integer potri( char uplo, integer n, float* a, integer lda )
{
    return c_spotri(uplo, n, a, lda);
}

inline integer potri( char uplo, integer n, double* a, integer lda )
{
    return c_dpotri(uplo, n, a, lda);
}

inline integer potri( char uplo, integer n, scomplex* a, integer lda )
{
    return c_cpotri(uplo, n, a, lda);
}

inline integer potri( char uplo, integer n, dcomplex* a, integer lda )
{
    return c_zpotri(uplo, n, a, lda);
}

inline integer pftri( char transr, char uplo, integer n, float* a )
{
    return c_spftri(transr, uplo, n, a);
}

inline integer pftri( char transr, char uplo, integer n, double* a )
{
    return c_dpftri(transr, uplo, n, a);
}

inline integer pftri( char transr, char uplo, integer n, scomplex* a )
{
    return c_cpftri(transr, uplo, n, a);
}

inline integer pftri( char transr, char uplo, integer n, dcomplex* a )
{
    return c_zpftri(transr, uplo, n, a);
}

inline integer pptri( char uplo, integer n, float* ap )
{
    return c_spptri(uplo, n, ap);
}

inline integer pptri( char uplo, integer n, double* ap )
{
    return c_dpptri(uplo, n, ap);
}

inline integer pptri( char uplo, integer n, scomplex* ap )
{
    return c_cpptri(uplo, n, ap);
}

inline integer pptri( char uplo, integer n, dcomplex* ap )
{
    return c_zpptri(uplo, n, ap);
}

inline integer sytri( char uplo, integer n, float* a, integer lda,
                    const integer* ipiv)
{
    return c_ssytri(uplo, n, a, lda, ipiv);
}

inline integer sytri( char uplo, integer n, double* a, integer lda,
                    const integer* ipiv)
{
    return c_dsytri(uplo, n, a, lda, ipiv);
}

inline integer sytri( char uplo, integer n, scomplex* a, integer lda,
                    const integer* ipiv)
{
    return c_csytri(uplo, n, a, lda, ipiv);
}

inline integer sytri( char uplo, integer n, dcomplex* a, integer lda,
                    const integer* ipiv)
{
    return c_zsytri(uplo, n, a, lda, ipiv);
}

inline integer hetri( char uplo, integer n, float* a, integer lda,
                    const integer* ipiv)
{
    return c_ssytri(uplo, n, a, lda, ipiv);
}

inline integer hetri( char uplo, integer n, double* a, integer lda,
                    const integer* ipiv)
{
    return c_dsytri(uplo, n, a, lda, ipiv);
}

inline integer hetri( char uplo, integer n, scomplex* a, integer lda,
                    const integer* ipiv)
{
    return c_chetri(uplo, n, a, lda, ipiv);
}

inline integer hetri( char uplo, integer n, dcomplex* a, integer lda,
                    const integer* ipiv)
{
    return c_zhetri(uplo, n, a, lda, ipiv);
}

inline integer sptri( char uplo, integer n, float* ap,
                    const integer* ipiv)
{
    return c_ssptri(uplo, n, ap, ipiv);
}

inline integer sptri( char uplo, integer n, double* ap,
                    const integer* ipiv)
{
    return c_dsptri(uplo, n, ap, ipiv);
}

inline integer sptri( char uplo, integer n, scomplex* ap,
                    const integer* ipiv)
{
    return c_csptri(uplo, n, ap, ipiv);
}

inline integer sptri( char uplo, integer n, dcomplex* ap,
                    const integer* ipiv)
{
    return c_zsptri(uplo, n, ap, ipiv);
}

inline integer hptri( char uplo, integer n, float* ap,
                    const integer* ipiv)
{
    return c_ssptri(uplo, n, ap, ipiv);
}

inline integer hptri( char uplo, integer n, double* ap,
                    const integer* ipiv)
{
    return c_dsptri(uplo, n, ap, ipiv);
}

inline integer hptri( char uplo, integer n, scomplex* ap,
                    const integer* ipiv)
{
    return c_chptri(uplo, n, ap, ipiv);
}

inline integer hptri( char uplo, integer n, dcomplex* ap,
                    const integer* ipiv)
{
    return c_zhptri(uplo, n, ap, ipiv);
}

inline integer trtri( char uplo, char diag, integer n, float* a,
                    integer lda )
{
    return c_strtri(uplo, diag, n, a, lda);
}

inline integer trtri( char uplo, char diag, integer n, double* a,
                    integer lda )
{
    return c_dtrtri(uplo, diag, n, a, lda);
}

inline integer trtri( char uplo, char diag, integer n, scomplex* a,
                    integer lda )
{
    return c_ctrtri(uplo, diag, n, a, lda);
}

inline integer trtri( char uplo, char diag, integer n, dcomplex* a,
                    integer lda )
{
    return c_ztrtri(uplo, diag, n, a, lda);
}

inline integer tftri( char transr, char uplo, char diag, integer n,
                    float* a )
{
    return c_stftri(transr, uplo, diag, n, a);
}

inline integer tftri( char transr, char uplo, char diag, integer n,
                    double* a )
{
    return c_dtftri(transr, uplo, diag, n, a);
}

inline integer tftri( char transr, char uplo, char diag, integer n,
                    scomplex* a )
{
    return c_ctftri(transr, uplo, diag, n, a);
}

inline integer tftri( char transr, char uplo, char diag, integer n,
                    dcomplex* a )
{
    return c_ztftri(transr, uplo, diag, n, a);
}

inline integer tptri( char uplo, char diag, integer n, float* ap )
{
    return c_stptri(uplo, diag, n, ap);
}

inline integer tptri( char uplo, char diag, integer n, double* ap )
{
    return c_dtptri(uplo, diag, n, ap);
}

inline integer tptri( char uplo, char diag, integer n, scomplex* ap )
{
    return c_ctptri(uplo, diag, n, ap);
}

inline integer tptri( char uplo, char diag, integer n, dcomplex* ap )
{
    return c_ztptri(uplo, diag, n, ap);
}

inline integer geequ( integer m, integer n, const float* a,
                    integer lda, float* r, float* c, float& rowcnd,
                    float& colcnd, float& amax )
{
    return c_sgeequ(m, n, a, lda, r, c, &rowcnd, &colcnd, &amax);
}

inline integer geequ( integer m, integer n, const double* a,
                    integer lda, double* r, double* c, double& rowcnd,
                    double& colcnd, double& amax )
{
    return c_dgeequ(m, n, a, lda, r, c, &rowcnd, &colcnd, &amax);
}

inline integer geequ( integer m, integer n, const scomplex* a,
                    integer lda, float* r, float* c, float& rowcnd,
                    float& colcnd, float& amax )
{
    return c_cgeequ(m, n, a, lda, r, c, &rowcnd, &colcnd, &amax);
}

inline integer geequ( integer m, integer n, const dcomplex* a,
                    integer lda, double* r, double* c, double& rowcnd,
                    double& colcnd, double& amax )
{
    return c_zgeequ(m, n, a, lda, r, c, &rowcnd, &colcnd, &amax);
}

inline integer geequb( integer m, integer n, const float* a,
                     integer lda, float* r, float* c, float& rowcnd,
                     float& colcnd, float& amax )
{
    return c_sgeequb(m, n, a, lda, r, c, &rowcnd, &colcnd, &amax);
}

inline integer geequb( integer m, integer n, const double* a,
                     integer lda, double* r, double* c, double& rowcnd,
                     double& colcnd, double& amax )
{
    return c_dgeequb(m, n, a, lda, r, c, &rowcnd, &colcnd, &amax);
}

inline integer geequb( integer m, integer n, const scomplex* a,
                     integer lda, float* r, float* c, float& rowcnd,
                     float& colcnd, float& amax )
{
    return c_cgeequb(m, n, a, lda, r, c, &rowcnd, &colcnd, &amax);
}

inline integer geequb( integer m, integer n, const dcomplex* a,
                     integer lda, double* r, double* c, double& rowcnd,
                     double& colcnd, double& amax )
{
    return c_zgeequb(m, n, a, lda, r, c, &rowcnd, &colcnd, &amax);
}

inline integer gbequ( integer m, integer n, integer kl,
                    integer ku, const float* ab, integer ldab,
                    float* r, float* c, float& rowcnd, float& colcnd,
                    float& amax )
{
    return c_sgbequ(m, n, kl, ku, ab, ldab, r, c, &rowcnd, &colcnd, &amax);
}

inline integer gbequ( integer m, integer n, integer kl,
                    integer ku, const double* ab, integer ldab,
                    double* r, double* c, double& rowcnd, double& colcnd,
                    double& amax )
{
    return c_dgbequ(m, n, kl, ku, ab, ldab, r, c, &rowcnd, &colcnd, &amax);
}

inline integer gbequ( integer m, integer n, integer kl,
                    integer ku, const scomplex* ab, integer ldab,
                    float* r, float* c, float& rowcnd, float& colcnd,
                    float& amax )
{
    return c_cgbequ(m, n, kl, ku, ab, ldab, r, c, &rowcnd, &colcnd, &amax);
}

inline integer gbequ( integer m, integer n, integer kl,
                    integer ku, const dcomplex* ab, integer ldab,
                    double* r, double* c, double& rowcnd, double& colcnd,
                    double& amax )
{
    return c_zgbequ(m, n, kl, ku, ab, ldab, r, c, &rowcnd, &colcnd, &amax);
}

inline integer gbequb( integer m, integer n, integer kl,
                     integer ku, const float* ab, integer ldab,
                     float* r, float* c, float& rowcnd, float& colcnd,
                     float& amax )
{
    return c_sgbequb(m, n, kl, ku, ab, ldab, r, c, &rowcnd, &colcnd, &amax);
}

inline integer gbequb( integer m, integer n, integer kl,
                     integer ku, const double* ab, integer ldab,
                     double* r, double* c, double& rowcnd, double& colcnd,
                     double& amax )
{
    return c_dgbequb(m, n, kl, ku, ab, ldab, r, c, &rowcnd, &colcnd, &amax);
}

inline integer gbequb( integer m, integer n, integer kl,
                     integer ku, const scomplex* ab, integer ldab,
                     float* r, float* c, float& rowcnd, float& colcnd,
                     float& amax )
{
    return c_cgbequb(m, n, kl, ku, ab, ldab, r, c, &rowcnd, &colcnd, &amax);
}

inline integer gbequb( integer m, integer n, integer kl,
                     integer ku, const dcomplex* ab, integer ldab,
                     double* r, double* c, double& rowcnd, double& colcnd,
                     double& amax )
{
    return c_zgbequb(m, n, kl, ku, ab, ldab, r, c, &rowcnd, &colcnd, &amax);
}

inline integer poequ( integer n, const float* a, integer lda, float* s,
                    float& scond, float& amax )
{
    return c_spoequ(n, a, lda, s, &scond, &amax);
}

inline integer poequ( integer n, const double* a, integer lda, double* s,
                    double& scond, double& amax )
{
    return c_dpoequ(n, a, lda, s, &scond, &amax);
}

inline integer poequ( integer n, const scomplex* a, integer lda, float* s,
                    float& scond, float& amax )
{
    return c_cpoequ(n, a, lda, s, &scond, &amax);
}

inline integer poequ( integer n, const dcomplex* a, integer lda, double* s,
                    double& scond, double& amax )
{
    return c_zpoequ(n, a, lda, s, &scond, &amax);
}

inline integer poequb( integer n, const float* a, integer lda, float* s,
                     float& scond, float& amax )
{
    return c_spoequb(n, a, lda, s, &scond, &amax);
}

inline integer poequb( integer n, const double* a, integer lda, double* s,
                     double& scond, double& amax )
{
    return c_dpoequb(n, a, lda, s, &scond, &amax);
}

inline integer poequb( integer n, const scomplex* a, integer lda, float* s,
                    float& scond, float& amax )
{
    return c_cpoequb(n, a, lda, s, &scond, &amax);
}

inline integer poequb( integer n, const dcomplex* a, integer lda, double* s,
                     double& scond, double& amax )
{
    return c_zpoequb(n, a, lda, s, &scond, &amax);
}

inline integer ppequ( char uplo, integer n, const float* ap, float* s,
                    float& scond, float& amax )
{
    return c_sppequ(uplo, n, ap, s, &scond, &amax);
}

inline integer ppequ( char uplo, integer n, const double* ap, double* s,
                    double& scond, double& amax )
{
    return c_dppequ(uplo, n, ap, s, &scond, &amax);
}

inline integer ppequ( char uplo, integer n, const scomplex* ap, float* s,
                    float& scond, float& amax )
{
    return c_cppequ(uplo, n, ap, s, &scond, &amax);
}

inline integer ppequ( char uplo, integer n, const dcomplex* ap, double* s,
                    double& scond, double& amax )
{
    return c_zppequ(uplo, n, ap, s, &scond, &amax);
}

inline integer pbequ( char uplo, integer n, integer kd, const float* ab,
                    integer ldab, float* s, float& scond, float& amax )
{
    return c_spbequ(uplo, n, kd, ab, ldab, s, &scond, &amax);
}

inline integer pbequ( char uplo, integer n, integer kd, const double* ab,
                    integer ldab, double* s, double& scond, double& amax )
{
    return c_dpbequ(uplo, n, kd, ab, ldab, s, &scond, &amax);
}

inline integer pbequ( char uplo, integer n, integer kd, const scomplex* ab,
                    integer ldab, float* s, float& scond, float& amax )
{
    return c_cpbequ(uplo, n, kd, ab, ldab, s, &scond, &amax);
}

inline integer pbequ( char uplo, integer n, integer kd, const dcomplex* ab,
                    integer ldab, double* s, double& scond, double& amax )
{
    return c_zpbequ(uplo, n, kd, ab, ldab, s, &scond, &amax);
}

inline integer syequb( char uplo, integer n, const float* a,
                     integer lda, float* s, float& scond, float& amax)
{
    return c_ssyequb(uplo, n, a, lda, s, &scond, &amax);
}

inline integer syequb( char uplo, integer n, const double* a,
                     integer lda, double* s, double& scond, double& amax)
{
    return c_dsyequb(uplo, n, a, lda, s, &scond, &amax);
}

inline integer syequb( char uplo, integer n, const scomplex* a,
                     integer lda, float* s, float& scond, float& amax)
{
    return c_csyequb(uplo, n, a, lda, s, &scond, &amax);
}

inline integer syequb( char uplo, integer n, const dcomplex* a,
                     integer lda, double* s, double& scond, double& amax)
{
    return c_zsyequb(uplo, n, a, lda, s, &scond, &amax);
}

inline integer heequb( char uplo, integer n, const float* a,
                     integer lda, float* s, float& scond, float& amax)
{
    return c_ssyequb(uplo, n, a, lda, s, &scond, &amax);
}

inline integer heequb( char uplo, integer n, const double* a,
                     integer lda, double* s, double& scond, double& amax)
{
    return c_dsyequb(uplo, n, a, lda, s, &scond, &amax);
}

inline integer heequb( char uplo, integer n, const scomplex* a,
                     integer lda, float* s, float& scond, float& amax)
{
    return c_cheequb(uplo, n, a, lda, s, &scond, &amax);
}

inline integer heequb( char uplo, integer n, const dcomplex* a,
                     integer lda, double* s, double& scond, double& amax)
{
    return c_zheequb(uplo, n, a, lda, s, &scond, &amax);
}

inline integer gesv( integer n, integer nrhs, float* a, integer lda,
                   integer* ipiv, float* b, integer ldb )
{
    return c_sgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer gesv( integer n, integer nrhs, double* a, integer lda,
                   integer* ipiv, double* b, integer ldb )
{
    return c_dgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer gesv( integer n, integer nrhs, scomplex* a, integer lda,
                   integer* ipiv, scomplex* b, integer ldb )
{
    return c_cgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer gesv( integer n, integer nrhs, dcomplex* a, integer lda,
                   integer* ipiv, dcomplex* b, integer ldb )
{
    return c_zgesv(n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer gesv( integer n, integer nrhs, double* a, integer lda,
                    integer* ipiv, double* b, integer ldb, double* x,
                    integer ldx, integer& iter )
{
    return c_dsgesv(n, nrhs, a, lda, ipiv, b, ldb, x, ldx, &iter);
}

inline integer gesv( integer n, integer nrhs, dcomplex* a, integer lda,
                    integer* ipiv, dcomplex* b, integer ldb, dcomplex* x,
                    integer ldx, integer& iter )
{
    return c_zcgesv(n, nrhs, a, lda, ipiv, b, ldb, x, ldx, &iter);
}

inline integer gesvx( char fact, char trans, integer n, integer nrhs,
                    float* a, integer lda, float* af, integer ldaf,
                    integer* ipiv, char* equed, float* r, float* c,
                    float* b, integer ldb, float* x, integer ldx,
                    float& rcond, float* ferr, float* berr)
{
    return c_sgesvx(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c,
                  b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer gesvx( char fact, char trans, integer n, integer nrhs,
                    double* a, integer lda, double* af, integer ldaf,
                    integer* ipiv, char* equed, double* r, double* c,
                    double* b, integer ldb, double* x, integer ldx,
                    double& rcond, double* ferr, double* berr)
{
    return c_dgesvx(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c,
                  b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer gesvx( char fact, char trans, integer n, integer nrhs,
                    scomplex* a, integer lda, scomplex* af, integer ldaf,
                    integer* ipiv, char* equed, float* r, float* c,
                    scomplex* b, integer ldb, scomplex* x, integer ldx,
                    float& rcond, float* ferr, float* berr)
{
    return c_cgesvx(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c,
                  b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer gesvx( char fact, char trans, integer n, integer nrhs,
                    dcomplex* a, integer lda, dcomplex* af, integer ldaf,
                    integer* ipiv, char* equed, double* r, double* c,
                    dcomplex* b, integer ldb, dcomplex* x, integer ldx,
                    double& rcond, double* ferr, double* berr)
{
    return c_zgesvx(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c,
                  b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer gesvxx( char fact, char trans, integer n, integer nrhs,
                    float* a, integer lda, float* af, integer ldaf,
                     integer* ipiv, char* equed, float* r, float* c,
                     float* b, integer ldb, float* x, integer ldx,
                     float& rcond, float& rpvgrw, float* berr,
                     integer n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, integer nparams, float* params)
{
    return c_sgesvxx(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer gesvxx( char fact, char trans, integer n, integer nrhs,
                     double* a, integer lda, double* af, integer ldaf,
                     integer* ipiv, char* equed, double* r, double* c,
                     double* b, integer ldb, double* x, integer ldx,
                     double& rcond, double& rpvgrw, double* berr,
                     integer n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, integer nparams, double* params)
{
    return c_dgesvxx(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer gesvxx( char fact, char trans, integer n, integer nrhs,
                     scomplex* a, integer lda, scomplex* af, integer ldaf,
                     integer* ipiv, char* equed, float* r, float* c,
                     scomplex* b, integer ldb, scomplex* x, integer ldx,
                     float& rcond, float& rpvgrw, float* berr,
                     integer n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, integer nparams, float* params)
{
    return c_cgesvxx(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer gesvxx( char fact, char trans, integer n, integer nrhs,
                    dcomplex* a, integer lda, dcomplex* af, integer ldaf,
                     integer* ipiv, char* equed, double* r, double* c,
                     dcomplex* b, integer ldb, dcomplex* x, integer ldx,
                     double& rcond, double& rpvgrw, double* berr,
                     integer n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, integer nparams, double* params)
{
    return c_zgesvxx(fact, trans, n, nrhs, a, lda, af, ldaf, ipiv, equed, r, c, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer gbsv( integer n, integer kl, integer ku,
                   integer nrhs, float* ab, integer ldab,
                   integer* ipiv, float* b, integer ldb )
{
    return c_sgbsv(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
}

inline integer gbsv( integer n, integer kl, integer ku,
                   integer nrhs, double* ab, integer ldab,
                   integer* ipiv, double* b, integer ldb )
{
    return c_dgbsv(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
}

inline integer gbsv( integer n, integer kl, integer ku,
                   integer nrhs, scomplex* ab, integer ldab,
                   integer* ipiv, scomplex* b, integer ldb )
{
    return c_cgbsv(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
}

inline integer gbsv( integer n, integer kl, integer ku,
                   integer nrhs, dcomplex* ab, integer ldab,
                   integer* ipiv, dcomplex* b, integer ldb )
{
    return c_zgbsv(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb);
}

inline integer gbsvx( char fact, char trans, integer n, integer kl,
                    integer ku, integer nrhs, float* ab,
                    integer ldab, float* afb, integer ldafb,
                    integer* ipiv, char* equed, float* r, float* c,
                    float* b, integer ldb, float* x, integer ldx,
                    float& rcond, float* ferr, float* berr)
{
    return c_sgbsvx(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r, c,
                  b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer gbsvx( char fact, char trans, integer n, integer kl,
                    integer ku, integer nrhs, double* ab,
                    integer ldab, double* afb, integer ldafb,
                    integer* ipiv, char* equed, double* r, double* c,
                    double* b, integer ldb, double* x, integer ldx,
                    double& rcond, double* ferr, double* berr)
{
    return c_dgbsvx(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r, c,
                  b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer gbsvx( char fact, char trans, integer n, integer kl,
                    integer ku, integer nrhs, scomplex* ab,
                    integer ldab, scomplex* afb, integer ldafb,
                    integer* ipiv, char* equed, float* r, float* c,
                    scomplex* b, integer ldb, scomplex* x, integer ldx,
                    float& rcond, float* ferr, float* berr)
{
    return c_cgbsvx(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r, c,
                  b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer gbsvx( char fact, char trans, integer n, integer kl,
                    integer ku, integer nrhs, dcomplex* ab,
                    integer ldab, dcomplex* afb, integer ldafb,
                    integer* ipiv, char* equed, double* r, double* c,
                    dcomplex* b, integer ldb, dcomplex* x, integer ldx,
                    double& rcond, double* ferr, double* berr)
{
    return c_zgbsvx(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r, c,
                  b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer gbsvxx( char fact, char trans, integer n, integer kl,
                     integer ku, integer nrhs, float* ab,
                     integer ldab, float* afb, integer ldafb,
                     integer* ipiv, char* equed, float* r, float* c,
                     float* b, integer ldb, float* x, integer ldx,
                     float& rcond, float& rpvgrw, float* berr,
                     integer n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, integer nparams, float* params)
{
    return c_sgbsvxx(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r, c, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer gbsvxx( char fact, char trans, integer n, integer kl,
                     integer ku, integer nrhs, double* ab,
                     integer ldab, double* afb, integer ldafb,
                     integer* ipiv, char* equed, double* r, double* c,
                     double* b, integer ldb, double* x, integer ldx,
                     double& rcond, double& rpvgrw, double* berr,
                     integer n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, integer nparams, double* params)
{
    return c_dgbsvxx(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r, c, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer gbsvxx( char fact, char trans, integer n, integer kl,
                     integer ku, integer nrhs, scomplex* ab,
                     integer ldab, scomplex* afb, integer ldafb,
                     integer* ipiv, char* equed, float* r, float* c,
                     scomplex* b, integer ldb, scomplex* x, integer ldx,
                     float& rcond, float& rpvgrw, float* berr,
                     integer n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, integer nparams, float* params)
{
    return c_cgbsvxx(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r, c, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer gbsvxx( char fact, char trans, integer n, integer kl,
                     integer ku, integer nrhs, dcomplex* ab,
                     integer ldab, dcomplex* afb, integer ldafb,
                     integer* ipiv, char* equed, double* r, double* c,
                     dcomplex* b, integer ldb, dcomplex* x, integer ldx,
                     double& rcond, double& rpvgrw, double* berr,
                     integer n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, integer nparams, double* params)
{
    return c_zgbsvxx(fact, trans, n, kl, ku, nrhs, ab, ldab, afb, ldafb, ipiv, equed, r, c, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer gtsv( integer n, integer nrhs, float* dl, float* d,
                   float* du, float* b, integer ldb )
{
    return c_sgtsv(n, nrhs, dl, d, du, b, ldb);
}

inline integer gtsv( integer n, integer nrhs, double* dl, double* d,
                   double* du, double* b, integer ldb )
{
    return c_dgtsv(n, nrhs, dl, d, du, b, ldb);
}

inline integer gtsv( integer n, integer nrhs, scomplex* dl, scomplex* d,
                   scomplex* du, scomplex* b, integer ldb )
{
    return c_cgtsv(n, nrhs, dl, d, du, b, ldb);
}

inline integer gtsv( integer n, integer nrhs, dcomplex* dl, dcomplex* d,
                   dcomplex* du, dcomplex* b, integer ldb )
{
    return c_zgtsv(n, nrhs, dl, d, du, b, ldb);
}

inline integer gtsvx( char fact, char trans, integer n, integer nrhs,
                    const float* dl, const float* d, const float* du,
                    float* dlf, float* df, float* duf, float* du2,
                    integer* ipiv, const float* b, integer ldb,
                    float* x, integer ldx, float& rcond, float* ferr,
                    float* berr)
{
    return c_sgtsvx(fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv,
                  b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer gtsvx( char fact, char trans, integer n, integer nrhs,
                    const double* dl, const double* d, const double* du,
                    double* dlf, double* df, double* duf, double* du2,
                    integer* ipiv, const double* b, integer ldb,
                    double* x, integer ldx, double& rcond, double* ferr,
                    double* berr)
{
    return c_dgtsvx(fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv,
                  b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer gtsvx( char fact, char trans, integer n, integer nrhs,
                    const scomplex* dl, const scomplex* d, const scomplex* du,
                    scomplex* dlf, scomplex* df, scomplex* duf, scomplex* du2,
                    integer* ipiv, const scomplex* b, integer ldb,
                    scomplex* x, integer ldx, float& rcond, float* ferr,
                    float* berr)
{
    return c_cgtsvx(fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv,
                  b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer gtsvx( char fact, char trans, integer n, integer nrhs,
                    const dcomplex* dl, const dcomplex* d, const dcomplex* du,
                    dcomplex* dlf, dcomplex* df, dcomplex* duf, dcomplex* du2,
                    integer* ipiv, const dcomplex* b, integer ldb,
                    dcomplex* x, integer ldx, double& rcond, double* ferr,
                    double* berr)
{
    return c_zgtsvx(fact, trans, n, nrhs, dl, d, du, dlf, df, duf, du2, ipiv,
                  b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer posv( char uplo, integer n, integer nrhs, float* a,
                   integer lda, float* b, integer ldb )
{
    return c_sposv(uplo, n, nrhs, a, lda, b, ldb);
}

inline integer posv( char uplo, integer n, integer nrhs, double* a,
                   integer lda, double* b, integer ldb )
{
    return c_dposv(uplo, n, nrhs, a, lda, b, ldb);
}

inline integer posv( char uplo, integer n, integer nrhs, scomplex* a,
                   integer lda, scomplex* b, integer ldb )
{
    return c_cposv(uplo, n, nrhs, a, lda, b, ldb);
}

inline integer posv( char uplo, integer n, integer nrhs, dcomplex* a,
                   integer lda, dcomplex* b, integer ldb )
{
    return c_zposv(uplo, n, nrhs, a, lda, b, ldb);
}

inline integer posv( char uplo, integer n, integer nrhs, double* a,
                    integer lda, double* b, integer ldb, double* x,
                    integer ldx, integer& iter )
{
    return c_dsposv(uplo, n, nrhs, a, lda, b, ldb, x, ldx, &iter);
}

inline integer posv( char uplo, integer n, integer nrhs, dcomplex* a,
                    integer lda, dcomplex* b, integer ldb, dcomplex* x,
                    integer ldx, integer& iter )
{
    return c_zcposv(uplo, n, nrhs, a, lda, b, ldb, x, ldx, &iter);
}

inline integer posvx( char fact, char uplo, integer n, integer nrhs,
                    float* a, integer lda, float* af, integer ldaf,
                    char* equed, float* s, float* b, integer ldb,
                    float* x, integer ldx, float& rcond, float* ferr,
                    float* berr)
{
    return c_sposvx(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb,
                  x, ldx, &rcond, ferr, berr);
}

inline integer posvx( char fact, char uplo, integer n, integer nrhs,
                    double* a, integer lda, double* af, integer ldaf,
                    char* equed, double* s, double* b, integer ldb,
                    double* x, integer ldx, double& rcond, double* ferr,
                    double* berr)
{
    return c_dposvx(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb,
                  x, ldx, &rcond, ferr, berr);
}

inline integer posvx( char fact, char uplo, integer n, integer nrhs,
                    scomplex* a, integer lda, scomplex* af, integer ldaf,
                    char* equed, float* s, scomplex* b, integer ldb,
                    scomplex* x, integer ldx, float& rcond, float* ferr,
                    float* berr)
{
    return c_cposvx(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb,
                  x, ldx, &rcond, ferr, berr);
}

inline integer posvx( char fact, char uplo, integer n, integer nrhs,
                    dcomplex* a, integer lda, dcomplex* af, integer ldaf,
                    char* equed, double* s, dcomplex* b, integer ldb,
                    dcomplex* x, integer ldx, double& rcond, double* ferr,
                    double* berr)
{
    return c_zposvx(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb,
                  x, ldx, &rcond, ferr, berr);
}

inline integer posvxx( char fact, char uplo, integer n, integer nrhs,
                    float* a, integer lda, float* af, integer ldaf,
                     char* equed, float* s, float* b, integer ldb,
                     float* x, integer ldx, float& rcond, float& rpvgrw,
                     float* berr, integer n_err_bnds,
                     float* err_bnds_norm, float* err_bnds_comp,
                     integer nparams, float* params)
{
    return c_sposvxx(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer posvxx( char fact, char uplo, integer n, integer nrhs,
                     double* a, integer lda, double* af, integer ldaf,
                     char* equed, double* s, double* b, integer ldb,
                     double* x, integer ldx, double& rcond, double& rpvgrw,
                     double* berr, integer n_err_bnds,
                     double* err_bnds_norm, double* err_bnds_comp,
                     integer nparams, double* params)
{
    return c_dposvxx(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer posvxx( char fact, char uplo, integer n, integer nrhs,
                     scomplex* a, integer lda, scomplex* af, integer ldaf,
                     char* equed, float* s, scomplex* b, integer ldb,
                     scomplex* x, integer ldx, float& rcond, float& rpvgrw,
                     float* berr, integer n_err_bnds,
                     float* err_bnds_norm, float* err_bnds_comp,
                     integer nparams, float* params)
{
    return c_cposvxx(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer posvxx( char fact, char uplo, integer n, integer nrhs,
                     dcomplex* a, integer lda, dcomplex* af, integer ldaf,
                     char* equed, double* s, dcomplex* b, integer ldb,
                     dcomplex* x, integer ldx, double& rcond, double& rpvgrw,
                     double* berr, integer n_err_bnds,
                     double* err_bnds_norm, double* err_bnds_comp,
                     integer nparams, double* params)
{
    return c_zposvxx(fact, uplo, n, nrhs, a, lda, af, ldaf, equed, s, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer ppsv( char uplo, integer n, integer nrhs, float* ap,
                    float* b, integer ldb )
{
    return c_sppsv(uplo, n, nrhs, ap, b, ldb);
}

inline integer ppsv( char uplo, integer n, integer nrhs, double* ap,
                   double* b, integer ldb )
{
    return c_dppsv(uplo, n, nrhs, ap, b, ldb);
}

inline integer ppsv( char uplo, integer n, integer nrhs, scomplex* ap,
                   scomplex* b, integer ldb )
{
    return c_cppsv(uplo, n, nrhs, ap, b, ldb);
}

inline integer ppsv( char uplo, integer n, integer nrhs, dcomplex* ap,
                   dcomplex* b, integer ldb )
{
    return c_zppsv(uplo, n, nrhs, ap, b, ldb);
}

inline integer ppsvx( char fact, char uplo, integer n, integer nrhs,
                    float* ap, float* afp, char* equed, float* s, float* b,
                    integer ldb, float* x, integer ldx, float& rcond,
                    float* ferr, float* berr)
{
    return c_sppsvx(fact, uplo, n, nrhs, ap, afp, equed, s, b, ldb, x, ldx,
                  &rcond, ferr, berr);
}

inline integer ppsvx( char fact, char uplo, integer n, integer nrhs,
                    double* ap, double* afp, char* equed, double* s, double* b,
                    integer ldb, double* x, integer ldx, double& rcond,
                    double* ferr, double* berr)
{
    return c_dppsvx(fact, uplo, n, nrhs, ap, afp, equed, s, b, ldb, x, ldx,
                  &rcond, ferr, berr);
}

inline integer ppsvx( char fact, char uplo, integer n, integer nrhs,
                    scomplex* ap, scomplex* afp, char* equed, float* s, scomplex* b,
                    integer ldb, scomplex* x, integer ldx, float& rcond,
                    float* ferr, float* berr)
{
    return c_cppsvx(fact, uplo, n, nrhs, ap, afp, equed, s, b, ldb, x, ldx,
                  &rcond, ferr, berr);
}

inline integer ppsvx( char fact, char uplo, integer n, integer nrhs,
                    dcomplex* ap, dcomplex* afp, char* equed, double* s, dcomplex* b,
                    integer ldb, dcomplex* x, integer ldx, double& rcond,
                    double* ferr, double* berr)
{
    return c_zppsvx(fact, uplo, n, nrhs, ap, afp, equed, s, b, ldb, x, ldx,
                  &rcond, ferr, berr);
}

inline integer pbsv( char uplo, integer n, integer kd, integer nrhs,
                   float* ab, integer ldab, float* b, integer ldb )
{
    return c_spbsv(uplo, n, kd, nrhs, ab, ldab, b, ldb);
}

inline integer pbsv( char uplo, integer n, integer kd, integer nrhs,
                   double* ab, integer ldab, double* b, integer ldb )
{
    return c_dpbsv(uplo, n, kd, nrhs, ab, ldab, b, ldb);
}

inline integer pbsv( char uplo, integer n, integer kd, integer nrhs,
                   scomplex* ab, integer ldab, scomplex* b, integer ldb )
{
    return c_cpbsv(uplo, n, kd, nrhs, ab, ldab, b, ldb);
}

inline integer pbsv( char uplo, integer n, integer kd, integer nrhs,
                   dcomplex* ab, integer ldab, dcomplex* b, integer ldb )
{
    return c_zpbsv(uplo, n, kd, nrhs, ab, ldab, b, ldb);
}

inline integer pbsvx( char fact, char uplo, integer n, integer kd,
                    integer nrhs, float* ab, integer ldab, float* afb,
                    integer ldafb, char* equed, float* s, float* b,
                    integer ldb, float* x, integer ldx, float& rcond,
                    float* ferr, float* berr)
{
    return c_spbsvx(fact, uplo, n, kd, nrhs, ab, ldab, afb, ldafb, equed, s, b, ldb,
                  x, ldx, &rcond, ferr, berr);
}

inline integer pbsvx( char fact, char uplo, integer n, integer kd,
                    integer nrhs, double* ab, integer ldab, double* afb,
                    integer ldafb, char* equed, double* s, double* b,
                    integer ldb, double* x, integer ldx, double& rcond,
                    double* ferr, double* berr)
{
    return c_dpbsvx(fact, uplo, n, kd, nrhs, ab, ldab, afb, ldafb, equed, s, b, ldb,
                  x, ldx, &rcond, ferr, berr);
}

inline integer pbsvx( char fact, char uplo, integer n, integer kd,
                    integer nrhs, scomplex* ab, integer ldab, scomplex* afb,
                    integer ldafb, char* equed, float* s, scomplex* b,
                    integer ldb, scomplex* x, integer ldx, float& rcond,
                    float* ferr, float* berr)
{
    return c_cpbsvx(fact, uplo, n, kd, nrhs, ab, ldab, afb, ldafb, equed, s, b, ldb,
                  x, ldx, &rcond, ferr, berr);
}

inline integer pbsvx( char fact, char uplo, integer n, integer kd,
                    integer nrhs, dcomplex* ab, integer ldab, dcomplex* afb,
                    integer ldafb, char* equed, double* s, dcomplex* b,
                    integer ldb, dcomplex* x, integer ldx, double& rcond,
                    double* ferr, double* berr)
{
    return c_zpbsvx(fact, uplo, n, kd, nrhs, ab, ldab, afb, ldafb, equed, s, b, ldb,
                  x, ldx, &rcond, ferr, berr);
}

inline integer ptsv( integer n, integer nrhs, float* d, float* e,
                   float* b, integer ldb )
{
    return c_sptsv(n, nrhs, d, e, b, ldb);
}

inline integer ptsv( integer n, integer nrhs, double* d, double* e,
                   double* b, integer ldb )
{
    return c_dptsv(n, nrhs, d, e, b, ldb);
}

inline integer ptsv( integer n, integer nrhs, float* d, scomplex* e,
                   scomplex* b, integer ldb )
{
    return c_cptsv(n, nrhs, d, e, b, ldb);
}

inline integer ptsv( integer n, integer nrhs, double* d, dcomplex* e,
                   dcomplex* b, integer ldb )
{
    return c_zptsv(n, nrhs, d, e, b, ldb);
}

inline integer ptsvx( char fact, integer n, integer nrhs,
                    const float* d, const float* e, float* df, float* ef,
                    const float* b, integer ldb, float* x,
                    integer ldx, float& rcond, float* ferr, float* berr)
{
    return c_sptsvx(fact, n, nrhs, d, e, df, ef, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer ptsvx( char fact, integer n, integer nrhs,
                    const double* d, const double* e, double* df, double* ef,
                    const double* b, integer ldb, double* x,
                    integer ldx, double& rcond, double* ferr, double* berr)
{
    return c_dptsvx(fact, n, nrhs, d, e, df, ef, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer ptsvx( char fact, integer n, integer nrhs,
                    const float* d, const scomplex* e, float* df, scomplex* ef,
                    const scomplex* b, integer ldb, scomplex* x,
                    integer ldx, float& rcond, float* ferr, float* berr)
{
    return c_cptsvx(fact, n, nrhs, d, e, df, ef, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer ptsvx( char fact, integer n, integer nrhs,
                    const double* d, const dcomplex* e, double* df, dcomplex* ef,
                    const dcomplex* b, integer ldb, dcomplex* x,
                    integer ldx, double& rcond, double* ferr, double* berr)
{
    return c_zptsvx(fact, n, nrhs, d, e, df, ef, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer sysv( char uplo, integer n, integer nrhs, float* a,
                   integer lda, integer* ipiv, float* b,
                   integer ldb)
{
    return c_ssysv(uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer sysv( char uplo, integer n, integer nrhs, double* a,
                   integer lda, integer* ipiv, double* b,
                   integer ldb)
{
    return c_dsysv(uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer sysv( char uplo, integer n, integer nrhs, scomplex* a,
                   integer lda, integer* ipiv, scomplex* b,
                   integer ldb)
{
    return c_csysv(uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer sysv( char uplo, integer n, integer nrhs, dcomplex* a,
                   integer lda, integer* ipiv, dcomplex* b,
                   integer ldb)
{
    return c_zsysv(uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer hesv( char uplo, integer n, integer nrhs, float* a,
                   integer lda, integer* ipiv, float* b,
                   integer ldb)
{
    return c_ssysv(uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer hesv( char uplo, integer n, integer nrhs, double* a,
                   integer lda, integer* ipiv, double* b,
                   integer ldb)
{
    return c_dsysv(uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer hesv( char uplo, integer n, integer nrhs, scomplex* a,
                   integer lda, integer* ipiv, scomplex* b,
                   integer ldb)
{
    return c_chesv(uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer hesv( char uplo, integer n, integer nrhs, dcomplex* a,
                   integer lda, integer* ipiv, dcomplex* b,
                   integer ldb)
{
    return c_zhesv(uplo, n, nrhs, a, lda, ipiv, b, ldb);
}

inline integer sysvx( char fact, char uplo, integer n, integer nrhs,
                    const float* a, integer lda, float* af,
                    integer ldaf, integer* ipiv, const float* b,
                    integer ldb, float* x, integer ldx, float& rcond,
                    float* ferr, float* berr)
{
    return c_ssysvx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer sysvx( char fact, char uplo, integer n, integer nrhs,
                    const double* a, integer lda, double* af,
                    integer ldaf, integer* ipiv, const double* b,
                    integer ldb, double* x, integer ldx, double& rcond,
                    double* ferr, double* berr)
{
    return c_dsysvx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer sysvx( char fact, char uplo, integer n, integer nrhs,
                    const scomplex* a, integer lda, scomplex* af,
                    integer ldaf, integer* ipiv, const scomplex* b,
                    integer ldb, scomplex* x, integer ldx, float& rcond,
                    float* ferr, float* berr)
{
    return c_csysvx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer sysvx( char fact, char uplo, integer n, integer nrhs,
                    const dcomplex* a, integer lda, dcomplex* af,
                    integer ldaf, integer* ipiv, const dcomplex* b,
                    integer ldb, dcomplex* x, integer ldx, double& rcond,
                    double* ferr, double* berr)
{
    return c_zsysvx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer hesvx( char fact, char uplo, integer n, integer nrhs,
                    const float* a, integer lda, float* af,
                    integer ldaf, integer* ipiv, const float* b,
                    integer ldb, float* x, integer ldx, float& rcond,
                    float* ferr, float* berr)
{
    return c_ssysvx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer hesvx( char fact, char uplo, integer n, integer nrhs,
                    const double* a, integer lda, double* af,
                    integer ldaf, integer* ipiv, const double* b,
                    integer ldb, double* x, integer ldx, double& rcond,
                    double* ferr, double* berr)
{
    return c_dsysvx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer hesvx( char fact, char uplo, integer n, integer nrhs,
                    const scomplex* a, integer lda, scomplex* af,
                    integer ldaf, integer* ipiv, const scomplex* b,
                    integer ldb, scomplex* x, integer ldx, float& rcond,
                    float* ferr, float* berr)
{
    return c_chesvx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer hesvx( char fact, char uplo, integer n, integer nrhs,
                    const dcomplex* a, integer lda, dcomplex* af,
                    integer ldaf, integer* ipiv, const dcomplex* b,
                    integer ldb, dcomplex* x, integer ldx, double& rcond,
                    double* ferr, double* berr)
{
    return c_zhesvx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer sysvxx( char fact, char uplo, integer n, integer nrhs,
                     float* a, integer lda, float* af, integer ldaf,
                     integer* ipiv, char* equed, float* s, float* b,
                     integer ldb, float* x, integer ldx, float& rcond,
                     float& rpvgrw, float* berr, integer n_err_bnds,
                     float* err_bnds_norm, float* err_bnds_comp,
                     integer nparams, float* params)
{
    return c_ssysvxx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, equed, s, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer sysvxx( char fact, char uplo, integer n, integer nrhs,
                     double* a, integer lda, double* af, integer ldaf,
                     integer* ipiv, char* equed, double* s, double* b,
                     integer ldb, double* x, integer ldx, double& rcond,
                     double& rpvgrw, double* berr, integer n_err_bnds,
                     double* err_bnds_norm, double* err_bnds_comp,
                     integer nparams, double* params)
{
    return c_dsysvxx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, equed, s, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer sysvxx( char fact, char uplo, integer n, integer nrhs,
                     scomplex* a, integer lda, scomplex* af, integer ldaf,
                     integer* ipiv, char* equed, float* s, scomplex* b,
                     integer ldb, scomplex* x, integer ldx, float& rcond,
                     float& rpvgrw, float* berr, integer n_err_bnds,
                     float* err_bnds_norm, float* err_bnds_comp,
                     integer nparams, float* params)
{
    return c_csysvxx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, equed, s, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer sysvxx( char fact, char uplo, integer n, integer nrhs,
                     dcomplex* a, integer lda, dcomplex* af, integer ldaf,
                     integer* ipiv, char* equed, double* s, dcomplex* b,
                     integer ldb, dcomplex* x, integer ldx, double& rcond,
                     double& rpvgrw, double* berr, integer n_err_bnds,
                     double* err_bnds_norm, double* err_bnds_comp,
                     integer nparams, double* params)
{
    return c_zsysvxx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, equed, s, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer hesvxx( char fact, char uplo, integer n, integer nrhs,
                     float* a, integer lda, float* af, integer ldaf,
                     integer* ipiv, char* equed, float* s, float* b,
                     integer ldb, float* x, integer ldx, float& rcond,
                     float& rpvgrw, float* berr, integer n_err_bnds,
                     float* err_bnds_norm, float* err_bnds_comp,
                     integer nparams, float* params)
{
    return c_ssysvxx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, equed, s, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer hesvxx( char fact, char uplo, integer n, integer nrhs,
                     double* a, integer lda, double* af, integer ldaf,
                     integer* ipiv, char* equed, double* s, double* b,
                     integer ldb, double* x, integer ldx, double& rcond,
                     double& rpvgrw, double* berr, integer n_err_bnds,
                     double* err_bnds_norm, double* err_bnds_comp,
                     integer nparams, double* params)
{
    return c_dsysvxx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, equed, s, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer hesvxx( char fact, char uplo, integer n, integer nrhs,
                     scomplex* a, integer lda, scomplex* af, integer ldaf,
                     integer* ipiv, char* equed, float* s, scomplex* b,
                     integer ldb, scomplex* x, integer ldx, float& rcond,
                     float& rpvgrw, float* berr, integer n_err_bnds,
                     float* err_bnds_norm, float* err_bnds_comp,
                     integer nparams, float* params)
{
    return c_chesvxx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, equed, s, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer hesvxx( char fact, char uplo, integer n, integer nrhs,
                     dcomplex* a, integer lda, dcomplex* af, integer ldaf,
                     integer* ipiv, char* equed, double* s, dcomplex* b,
                     integer ldb, dcomplex* x, integer ldx, double& rcond,
                     double& rpvgrw, double* berr, integer n_err_bnds,
                     double* err_bnds_norm, double* err_bnds_comp,
                     integer nparams, double* params)
{
    return c_zhesvxx(fact, uplo, n, nrhs, a, lda, af, ldaf, ipiv, equed, s, b, ldb, x, ldx,
                   &rcond, &rpvgrw, berr, n_err_bnds, err_bnds_norm, err_bnds_comp, nparams, params);
}

inline integer spsv( char uplo, integer n, integer nrhs, float* ap,
                   integer* ipiv, float* b, integer ldb )
{
    return c_sspsv(uplo, n, nrhs, ap, ipiv, b, ldb);
}

inline integer spsv( char uplo, integer n, integer nrhs, double* ap,
                   integer* ipiv, double* b, integer ldb )
{
    return c_dspsv(uplo, n, nrhs, ap, ipiv, b, ldb);
}

inline integer spsv( char uplo, integer n, integer nrhs, scomplex* ap,
                   integer* ipiv, scomplex* b, integer ldb )
{
    return c_cspsv(uplo, n, nrhs, ap, ipiv, b, ldb);
}

inline integer spsv( char uplo, integer n, integer nrhs, dcomplex* ap,
                   integer* ipiv, dcomplex* b, integer ldb )
{
    return c_zspsv(uplo, n, nrhs, ap, ipiv, b, ldb);
}

inline integer hpsv( char uplo, integer n, integer nrhs, float* ap,
                   integer* ipiv, float* b, integer ldb )
{
    return c_sspsv(uplo, n, nrhs, ap, ipiv, b, ldb);
}

inline integer hpsv( char uplo, integer n, integer nrhs, double* ap,
                   integer* ipiv, double* b, integer ldb )
{
    return c_dspsv(uplo, n, nrhs, ap, ipiv, b, ldb);
}

inline integer hpsv( char uplo, integer n, integer nrhs, scomplex* ap,
                   integer* ipiv, scomplex* b, integer ldb )
{
    return c_chpsv(uplo, n, nrhs, ap, ipiv, b, ldb);
}

inline integer hpsv( char uplo, integer n, integer nrhs, dcomplex* ap,
                   integer* ipiv, dcomplex* b, integer ldb )
{
    return c_zhpsv(uplo, n, nrhs, ap, ipiv, b, ldb);
}

inline integer spsvx( char fact, char uplo, integer n, integer nrhs,
                    const float* ap, float* afp, integer* ipiv,
                    const float* b, integer ldb, float* x,
                    integer ldx, float& rcond, float* ferr, float* berr)
{
    return c_sspsvx(fact, uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer spsvx( char fact, char uplo, integer n, integer nrhs,
                    const double* ap, double* afp, integer* ipiv,
                    const double* b, integer ldb, double* x,
                    integer ldx, double& rcond, double* ferr, double* berr)
{
    return c_dspsvx(fact, uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer spsvx( char fact, char uplo, integer n, integer nrhs,
                    const scomplex* ap, scomplex* afp, integer* ipiv,
                    const scomplex* b, integer ldb, scomplex* x,
                    integer ldx, float& rcond, float* ferr, float* berr)
{
    return c_cspsvx(fact, uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer spsvx( char fact, char uplo, integer n, integer nrhs,
                    const dcomplex* ap, dcomplex* afp, integer* ipiv,
                    const dcomplex* b, integer ldb, dcomplex* x,
                    integer ldx, double& rcond, double* ferr, double* berr)
{
    return c_zspsvx(fact, uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer hpsvx( char fact, char uplo, integer n, integer nrhs,
                    const float* ap, float* afp, integer* ipiv,
                    const float* b, integer ldb, float* x,
                    integer ldx, float& rcond, float* ferr, float* berr)
{
    return c_sspsvx(fact, uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer hpsvx( char fact, char uplo, integer n, integer nrhs,
                    const double* ap, double* afp, integer* ipiv,
                    const double* b, integer ldb, double* x,
                    integer ldx, double& rcond, double* ferr, double* berr)
{
    return c_dspsvx(fact, uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer hpsvx( char fact, char uplo, integer n, integer nrhs,
                    const scomplex* ap, scomplex* afp, integer* ipiv,
                    const scomplex* b, integer ldb, scomplex* x,
                    integer ldx, float& rcond, float* ferr, float* berr)
{
    return c_chpsvx(fact, uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer hpsvx( char fact, char uplo, integer n, integer nrhs,
                    const dcomplex* ap, dcomplex* afp, integer* ipiv,
                    const dcomplex* b, integer ldb, dcomplex* x,
                    integer ldx, double& rcond, double* ferr, double* berr)
{
    return c_zhpsvx(fact, uplo, n, nrhs, ap, afp, ipiv, b, ldb, x, ldx, &rcond, ferr, berr);
}

inline integer geqrf( integer m, integer n, float* a, integer lda,
                    float* tau)
{
    return c_sgeqrf(m, n, a, lda, tau);
}

inline integer geqrf( integer m, integer n, double* a, integer lda,
                    double* tau)
{
    return c_dgeqrf(m, n, a, lda, tau);
}

inline integer geqrf( integer m, integer n, scomplex* a, integer lda,
                    scomplex* tau)
{
    return c_cgeqrf(m, n, a, lda, tau);
}

inline integer geqrf( integer m, integer n, dcomplex* a, integer lda,
                    dcomplex* tau)
{
    return c_zgeqrf(m, n, a, lda, tau);
}

inline integer geqpf( integer m, integer n, float* a, integer lda,
                    integer* jpvt, float* tau)
{
    return c_sgeqpf(m, n, a, lda, jpvt, tau);
}

inline integer geqpf( integer m, integer n, double* a, integer lda,
                    integer* jpvt, double* tau)
{
    return c_dgeqpf(m, n, a, lda, jpvt, tau);
}

inline integer geqpf( integer m, integer n, scomplex* a, integer lda,
                    integer* jpvt, scomplex* tau)
{
    return c_cgeqpf(m, n, a, lda, jpvt, tau);
}

inline integer geqpf( integer m, integer n, dcomplex* a, integer lda,
                    integer* jpvt, dcomplex* tau)
{
    return c_zgeqpf(m, n, a, lda, jpvt, tau);
}

inline integer geqp3( integer m, integer n, float* a, integer lda,
                    integer* jpvt, float* tau)
{
    return c_sgeqp3(m, n, a, lda, jpvt, tau);
}

inline integer geqp3( integer m, integer n, double* a, integer lda,
                    integer* jpvt, double* tau)
{
    return c_dgeqp3(m, n, a, lda, jpvt, tau);
}

inline integer geqp3( integer m, integer n, scomplex* a, integer lda,
                    integer* jpvt, scomplex* tau)
{
    return c_cgeqp3(m, n, a, lda, jpvt, tau);
}

inline integer geqp3( integer m, integer n, dcomplex* a, integer lda,
                    integer* jpvt, dcomplex* tau)
{
    return c_zgeqp3(m, n, a, lda, jpvt, tau);
}

inline integer ungqr( integer m, integer n, integer k, float* a,
                    integer lda, const float* tau)
{
    return c_sorgqr(m, n, k, a, lda, tau);
}

inline integer ungqr( integer m, integer n, integer k, double* a,
                    integer lda, const double* tau)
{
    return c_dorgqr(m, n, k, a, lda, tau);
}

inline integer ungqr( integer m, integer n, integer k, scomplex* a,
                    integer lda, const scomplex* tau)
{
    return c_cungqr(m, n, k, a, lda, tau);
}

inline integer ungqr( integer m, integer n, integer k, dcomplex* a,
                    integer lda, const dcomplex* tau)
{
    return c_zungqr(m, n, k, a, lda, tau);
}

inline integer unmqr( char side, char trans, integer m, integer n,
                    integer k, const float* a, integer lda,
                    const float* tau, float* c, integer ldc)
{
    return c_sormqr(side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer unmqr( char side, char trans, integer m, integer n,
                    integer k, const double* a, integer lda,
                    const double* tau, double* c, integer ldc)
{
    return c_dormqr(side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer unmqr( char side, char trans, integer m, integer n,
                    integer k, const scomplex* a, integer lda,
                    const scomplex* tau, scomplex* c, integer ldc)
{
    return c_cunmqr(side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer unmqr( char side, char trans, integer m, integer n,
                    integer k, const dcomplex* a, integer lda,
                    const dcomplex* tau, dcomplex* c, integer ldc)
{
    return c_zunmqr(side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer gelqf( integer m, integer n, float* a, integer lda,
                    float* tau)
{
    return c_sgelqf(m, n, a, lda, tau);
}

inline integer gelqf( integer m, integer n, double* a, integer lda,
                    double* tau)
{
    return c_dgelqf(m, n, a, lda, tau);
}

inline integer gelqf( integer m, integer n, scomplex* a, integer lda,
                    scomplex* tau)
{
    return c_cgelqf(m, n, a, lda, tau);
}

inline integer gelqf( integer m, integer n, dcomplex* a, integer lda,
                    dcomplex* tau)
{
    return c_zgelqf(m, n, a, lda, tau);
}

inline integer unglq( integer m, integer n, integer k, float* a,
                    integer lda, const float* tau)
{
    return c_sorglq(m, n, k, a, lda, tau);
}

inline integer unglq( integer m, integer n, integer k, double* a,
                    integer lda, const double* tau)
{
    return c_dorglq(m, n, k, a, lda, tau);
}

inline integer unglq( integer m, integer n, integer k, scomplex* a,
                    integer lda, const scomplex* tau)
{
    return c_cunglq(m, n, k, a, lda, tau);
}

inline integer unglq( integer m, integer n, integer k, dcomplex* a,
                    integer lda, const dcomplex* tau)
{
    return c_zunglq(m, n, k, a, lda, tau);
}

inline integer unmlq( char side, char trans, integer m, integer n,
                    integer k, const float* a, integer lda,
                    const float* tau, float* c, integer ldc)
{
    return c_sormlq(side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer unmlq( char side, char trans, integer m, integer n,
                    integer k, const double* a, integer lda,
                    const double* tau, double* c, integer ldc)
{
    return c_dormlq(side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer unmlq( char side, char trans, integer m, integer n,
                    integer k, const scomplex* a, integer lda,
                    const scomplex* tau, scomplex* c, integer ldc)
{
    return c_cunmlq(side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer unmlq( char side, char trans, integer m, integer n,
                    integer k, const dcomplex* a, integer lda,
                    const dcomplex* tau, dcomplex* c, integer ldc)
{
    return c_zunmlq(side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer geqlf( integer m, integer n, float* a, integer lda,
                    float* tau)
{
    return c_sgeqlf(m, n, a, lda, tau);
}

inline integer geqlf( integer m, integer n, double* a, integer lda,
                    double* tau)
{
    return c_dgeqlf(m, n, a, lda, tau);
}

inline integer geqlf( integer m, integer n, scomplex* a, integer lda,
                    scomplex* tau)
{
    return c_cgeqlf(m, n, a, lda, tau);
}

inline integer geqlf( integer m, integer n, dcomplex* a, integer lda,
                    dcomplex* tau)
{
    return c_zgeqlf(m, n, a, lda, tau);
}

inline integer ungql( integer m, integer n, integer k, float* a,
                    integer lda, const float* tau)
{
    return c_sorgql(m, n, k, a, lda, tau);
}

inline integer ungql( integer m, integer n, integer k, double* a,
                    integer lda, const double* tau)
{
    return c_dorgql(m, n, k, a, lda, tau);
}

inline integer ungql( integer m, integer n, integer k, scomplex* a,
                    integer lda, const scomplex* tau)
{
    return c_cungql(m, n, k, a, lda, tau);
}

inline integer ungql( integer m, integer n, integer k, dcomplex* a,
                    integer lda, const dcomplex* tau)
{
    return c_zungql(m, n, k, a, lda, tau);
}

inline integer unmql( char side, char trans, integer m, integer n,
                    integer k, const float* a, integer lda,
                    const float* tau, float* c, integer ldc)
{
    return c_sormql(side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer unmql( char side, char trans, integer m, integer n,
                    integer k, const double* a, integer lda,
                    const double* tau, double* c, integer ldc)
{
    return c_dormql(side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer unmql( char side, char trans, integer m, integer n,
                    integer k, const scomplex* a, integer lda,
                    const scomplex* tau, scomplex* c, integer ldc)
{
    return c_cunmql(side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer unmql( char side, char trans, integer m, integer n,
                    integer k, const dcomplex* a, integer lda,
                    const dcomplex* tau, dcomplex* c, integer ldc)
{
    return c_zunmql(side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer gerqf( integer m, integer n, float* a, integer lda,
                    float* tau)
{
    return c_sgerqf(m, n, a, lda, tau);
}

inline integer gerqf( integer m, integer n, double* a, integer lda,
                    double* tau)
{
    return c_dgerqf(m, n, a, lda, tau);
}

inline integer gerqf( integer m, integer n, scomplex* a, integer lda,
                    scomplex* tau)
{
    return c_cgerqf(m, n, a, lda, tau);
}

inline integer gerqf( integer m, integer n, dcomplex* a, integer lda,
                    dcomplex* tau)
{
    return c_zgerqf(m, n, a, lda, tau);
}

inline integer ungrq( integer m, integer n, integer k, float* a,
                    integer lda, const float* tau)
{
    return c_sorgrq(m, n, k, a, lda, tau);
}

inline integer ungrq( integer m, integer n, integer k, double* a,
                    integer lda, const double* tau)
{
    return c_dorgrq(m, n, k, a, lda, tau);
}

inline integer ungrq( integer m, integer n, integer k, scomplex* a,
                    integer lda, const scomplex* tau)
{
    return c_cungrq(m, n, k, a, lda, tau);
}

inline integer ungrq( integer m, integer n, integer k, dcomplex* a,
                    integer lda, const dcomplex* tau)
{
    return c_zungrq(m, n, k, a, lda, tau);
}

inline integer unmrq( char side, char trans, integer m, integer n,
                    integer k, const float* a, integer lda,
                    const float* tau, float* c, integer ldc)
{
    return c_sormrq(side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer unmrq( char side, char trans, integer m, integer n,
                    integer k, const double* a, integer lda,
                    const double* tau, double* c, integer ldc)
{
    return c_dormrq(side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer unmrq( char side, char trans, integer m, integer n,
                    integer k, const scomplex* a, integer lda,
                    const scomplex* tau, scomplex* c, integer ldc)
{
    return c_cunmrq(side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer unmrq( char side, char trans, integer m, integer n,
                    integer k, const dcomplex* a, integer lda,
                    const dcomplex* tau, dcomplex* c, integer ldc)
{
    return c_zunmrq(side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer tzrzf( integer m, integer n, float* a, integer lda,
                    float* tau)
{
    return c_stzrzf(m, n, a, lda, tau);
}

inline integer tzrzf( integer m, integer n, double* a, integer lda,
                    double* tau)
{
    return c_dtzrzf(m, n, a, lda, tau);
}

inline integer tzrzf( integer m, integer n, scomplex* a, integer lda,
                    scomplex* tau)
{
    return c_ctzrzf(m, n, a, lda, tau);
}

inline integer tzrzf( integer m, integer n, dcomplex* a, integer lda,
                    dcomplex* tau)
{
    return c_ztzrzf(m, n, a, lda, tau);
}

inline integer unmrz( char side, char trans, integer m, integer n,
                    integer k, integer l, const float* a,
                    integer lda, const float* tau, float* c,
                    integer ldc)
{
    return c_sormrz(side, trans, m, n, k, l, a, lda, tau, c, ldc);
}

inline integer unmrz( char side, char trans, integer m, integer n,
                    integer k, integer l, const double* a,
                    integer lda, const double* tau, double* c,
                    integer ldc)
{
    return c_dormrz(side, trans, m, n, k, l, a, lda, tau, c, ldc);
}

inline integer unmrz( char side, char trans, integer m, integer n,
                    integer k, integer l, const scomplex* a,
                    integer lda, const scomplex* tau, scomplex* c,
                    integer ldc)
{
    return c_cunmrz(side, trans, m, n, k, l, a, lda, tau, c, ldc);
}

inline integer unmrz( char side, char trans, integer m, integer n,
                    integer k, integer l, const dcomplex* a,
                    integer lda, const dcomplex* tau, dcomplex* c,
                    integer ldc)
{
    return c_zunmrz(side, trans, m, n, k, l, a, lda, tau, c, ldc);
}

inline integer ggqrf( integer n, integer m, integer p, float* a,
                    integer lda, float* taua, float* b, integer ldb,
                    float* taub)
{
    return c_sggqrf(n, m, p, a, lda, taua, b, ldb, taub);
}

inline integer ggqrf( integer n, integer m, integer p, double* a,
                    integer lda, double* taua, double* b, integer ldb,
                    double* taub)
{
    return c_dggqrf(n, m, p, a, lda, taua, b, ldb, taub);
}

inline integer ggqrf( integer n, integer m, integer p, scomplex* a,
                    integer lda, scomplex* taua, scomplex* b, integer ldb,
                    scomplex* taub)
{
    return c_cggqrf(n, m, p, a, lda, taua, b, ldb, taub);
}

inline integer ggqrf( integer n, integer m, integer p, dcomplex* a,
                    integer lda, dcomplex* taua, dcomplex* b, integer ldb,
                    dcomplex* taub)
{
    return c_zggqrf(n, m, p, a, lda, taua, b, ldb, taub);
}

inline integer ggrqf( integer m, integer p, integer n, float* a,
                    integer lda, float* taua, float* b, integer ldb,
                    float* taub)
{
    return c_sggrqf(m, p, n, a, lda, taua, b, ldb, taub);
}

inline integer ggrqf( integer m, integer p, integer n, double* a,
                    integer lda, double* taua, double* b, integer ldb,
                    double* taub)
{
    return c_dggrqf(m, p, n, a, lda, taua, b, ldb, taub);
}

inline integer ggrqf( integer m, integer p, integer n, scomplex* a,
                    integer lda, scomplex* taua, scomplex* b, integer ldb,
                    scomplex* taub)
{
    return c_cggrqf(m, p, n, a, lda, taua, b, ldb, taub);
}

inline integer ggrqf( integer m, integer p, integer n, dcomplex* a,
                    integer lda, dcomplex* taua, dcomplex* b, integer ldb,
                    dcomplex* taub)
{
    return c_zggrqf(m, p, n, a, lda, taua, b, ldb, taub);
}

inline integer gebrd( integer m, integer n, float* a, integer lda,
                    float* d, float* e, float* tauq, float* taup)
{
    return c_sgebrd(m, n, a, lda, d, e, tauq, taup);
}

inline integer gebrd( integer m, integer n, double* a, integer lda,
                    double* d, double* e, double* tauq, double* taup)
{
    return c_dgebrd(m, n, a, lda, d, e, tauq, taup);
}

inline integer gebrd( integer m, integer n, scomplex* a, integer lda,
                    float* d, float* e, scomplex* tauq, scomplex* taup)
{
    return c_cgebrd(m, n, a, lda, d, e, tauq, taup);
}

inline integer gebrd( integer m, integer n, dcomplex* a, integer lda,
                    double* d, double* e, dcomplex* tauq, dcomplex* taup)
{
    return c_zgebrd(m, n, a, lda, d, e, tauq, taup);
}

inline integer gbbrd( char vect, integer m, integer n, integer ncc,
                    integer kl, integer ku, float* ab,
                    integer ldab, float* d, float* e, float* q,
                    integer ldq, float* pt, integer ldpt, float* c,
                    integer ldc)
{
    return c_sgbbrd(vect, m, n, ncc, kl, ku, ab, ldab, d, e, q, ldq, pt, ldpt, c, ldc);
}

inline integer gbbrd( char vect, integer m, integer n, integer ncc,
                    integer kl, integer ku, double* ab,
                    integer ldab, double* d, double* e, double* q,
                    integer ldq, double* pt, integer ldpt, double* c,
                    integer ldc)
{
    return c_dgbbrd(vect, m, n, ncc, kl, ku, ab, ldab, d, e, q, ldq, pt, ldpt, c, ldc);
}

inline integer gbbrd( char vect, integer m, integer n, integer ncc,
                    integer kl, integer ku, scomplex* ab,
                    integer ldab, float* d, float* e, scomplex* q,
                    integer ldq, scomplex* pt, integer ldpt, scomplex* c,
                    integer ldc)
{
    return c_cgbbrd(vect, m, n, ncc, kl, ku, ab, ldab, d, e, q, ldq, pt, ldpt, c, ldc);
}

inline integer gbbrd( char vect, integer m, integer n, integer ncc,
                    integer kl, integer ku, dcomplex* ab,
                    integer ldab, double* d, double* e, dcomplex* q,
                    integer ldq, dcomplex* pt, integer ldpt, dcomplex* c,
                    integer ldc)
{
    return c_zgbbrd(vect, m, n, ncc, kl, ku, ab, ldab, d, e, q, ldq, pt, ldpt, c, ldc);
}

inline integer ungbr( char vect, integer m, integer n, integer k,
                    float* a, integer lda, const float* tau)
{
    return c_sorgbr(vect, m, n, k, a, lda, tau);
}

inline integer ungbr( char vect, integer m, integer n, integer k,
                    double* a, integer lda, const double* tau)
{
    return c_dorgbr(vect, m, n, k, a, lda, tau);
}

inline integer ungbr( char vect, integer m, integer n, integer k,
                    scomplex* a, integer lda, const scomplex* tau)
{
    return c_cungbr(vect, m, n, k, a, lda, tau);
}

inline integer ungbr( char vect, integer m, integer n, integer k,
                    dcomplex* a, integer lda, const dcomplex* tau)
{
    return c_zungbr(vect, m, n, k, a, lda, tau);
}

inline integer unmbr( char vect, char side, char trans, integer m,
                    integer n, integer k, const float* a,
                    integer lda, const float* tau, float* c,
                    integer ldc)
{
    return c_sormbr(vect, side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer unmbr( char vect, char side, char trans, integer m,
                    integer n, integer k, const double* a,
                    integer lda, const double* tau, double* c,
                    integer ldc)
{
    return c_dormbr(vect, side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer unmbr( char vect, char side, char trans, integer m,
                    integer n, integer k, const scomplex* a,
                    integer lda, const scomplex* tau, scomplex* c,
                    integer ldc)
{
    return c_cunmbr(vect, side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer unmbr( char vect, char side, char trans, integer m,
                    integer n, integer k, const dcomplex* a,
                    integer lda, const dcomplex* tau, dcomplex* c,
                    integer ldc)
{
    return c_zunmbr(vect, side, trans, m, n, k, a, lda, tau, c, ldc);
}

inline integer bdsqr( char uplo, integer n, integer ncvt,
                    integer nru, integer ncc, float* d, float* e,
                    float* vt, integer ldvt, float* u, integer ldu,
                    float* c, integer ldc)
{
    return c_sbdsqr(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc);
}

inline integer bdsqr( char uplo, integer n, integer ncvt,
                    integer nru, integer ncc, double* d, double* e,
                    double* vt, integer ldvt, double* u, integer ldu,
                    double* c, integer ldc)
{
    return c_dbdsqr(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc);
}

inline integer bdsqr( char uplo, integer n, integer ncvt,
                    integer nru, integer ncc, float* d, float* e,
                    scomplex* vt, integer ldvt, scomplex* u, integer ldu,
                    scomplex* c, integer ldc)
{
    return c_cbdsqr(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc);
}

inline integer bdsqr( char uplo, integer n, integer ncvt,
                    integer nru, integer ncc, double* d, double* e,
                    dcomplex* vt, integer ldvt, dcomplex* u, integer ldu,
                    dcomplex* c, integer ldc)
{
    return c_zbdsqr(uplo, n, ncvt, nru, ncc, d, e, vt, ldvt, u, ldu, c, ldc);
}

inline integer bdsdc( char uplo, char compq, integer n, float* d,
                    float* e, float* u, integer ldu, float* vt,
                    integer ldvt, float* q, integer* iq)
{
    return c_sbdsdc(uplo, compq, n, d, e, u, ldu, vt, ldvt, q, iq);
}

inline integer bdsdc( char uplo, char compq, integer n, double* d,
                    double* e, double* u, integer ldu, double* vt,
                    integer ldvt, double* q, integer* iq)
{
    return c_dbdsdc(uplo, compq, n, d, e, u, ldu, vt, ldvt, q, iq);
}

inline integer hetrd( char uplo, integer n, float* a, integer lda,
                    float* d, float* e, float* tau)
{
    return c_ssytrd(uplo, n, a, lda, d, e, tau);
}

inline integer hetrd( char uplo, integer n, double* a, integer lda,
                    double* d, double* e, double* tau)
{
    return c_dsytrd(uplo, n, a, lda, d, e, tau);
}

inline integer hetrd( char uplo, integer n, scomplex* a, integer lda,
                    float* d, float* e, scomplex* tau)
{
    return c_chetrd(uplo, n, a, lda, d, e, tau);
}

inline integer hetrd( char uplo, integer n, dcomplex* a, integer lda,
                    double* d, double* e, dcomplex* tau)
{
    return c_zhetrd(uplo, n, a, lda, d, e, tau);
}

inline integer ungtr( char uplo, integer n, float* a, integer lda,
                    const float* tau)
{
    return c_sorgtr(uplo, n, a, lda, tau);
}

inline integer ungtr( char uplo, integer n, double* a, integer lda,
                    const double* tau)
{
    return c_dorgtr(uplo, n, a, lda, tau);
}

inline integer ungtr( char uplo, integer n, scomplex* a, integer lda,
                    const scomplex* tau)
{
    return c_cungtr(uplo, n, a, lda, tau);
}

inline integer ungtr( char uplo, integer n, dcomplex* a, integer lda,
                    const dcomplex* tau)
{
    return c_zungtr(uplo, n, a, lda, tau);
}

inline integer unmtr( char side, char uplo, char trans, integer m,
                    integer n, const float* a, integer lda,
                    const float* tau, float* c, integer ldc)
{
    return c_sormtr(side, uplo, trans, m, n, a, lda, tau, c, ldc);
}

inline integer unmtr( char side, char uplo, char trans, integer m,
                    integer n, const double* a, integer lda,
                    const double* tau, double* c, integer ldc)
{
    return c_dormtr(side, uplo, trans, m, n, a, lda, tau, c, ldc);
}

inline integer unmtr( char side, char uplo, char trans, integer m,
                    integer n, const scomplex* a, integer lda,
                    const scomplex* tau, scomplex* c, integer ldc)
{
    return c_cunmtr(side, uplo, trans, m, n, a, lda, tau, c, ldc);
}

inline integer unmtr( char side, char uplo, char trans, integer m,
                    integer n, const dcomplex* a, integer lda,
                    const dcomplex* tau, dcomplex* c, integer ldc)
{
    return c_zunmtr(side, uplo, trans, m, n, a, lda, tau, c, ldc);
}

inline integer hptrd( char uplo, integer n, float* ap, float* d, float* e,
                    float* tau )
{
    return c_ssptrd(uplo, n, ap, d, e, tau);
}

inline integer hptrd( char uplo, integer n, double* ap, double* d, double* e,
                    double* tau )
{
    return c_dsptrd(uplo, n, ap, d, e, tau);
}

inline integer hptrd( char uplo, integer n, scomplex* ap, float* d, float* e,
                    scomplex* tau )
{
    return c_chptrd(uplo, n, ap, d, e, tau);
}

inline integer hptrd( char uplo, integer n, dcomplex* ap, double* d, double* e,
                    dcomplex* tau )
{
    return c_zhptrd(uplo, n, ap, d, e, tau);
}

inline integer upgtr( char uplo, integer n, const float* ap,
                    const float* tau, float* q, integer ldq)
{
    return c_sopgtr(uplo, n, ap, tau, q, ldq);
}

inline integer upgtr( char uplo, integer n, const double* ap,
                    const double* tau, double* q, integer ldq)
{
    return c_dopgtr(uplo, n, ap, tau, q, ldq);
}

inline integer upgtr( char uplo, integer n, const scomplex* ap,
                    const scomplex* tau, scomplex* q, integer ldq)
{
    return c_cupgtr(uplo, n, ap, tau, q, ldq);
}

inline integer upgtr( char uplo, integer n, const dcomplex* ap,
                    const dcomplex* tau, dcomplex* q, integer ldq)
{
    return c_zupgtr(uplo, n, ap, tau, q, ldq);
}

inline integer upmtr( char side, char uplo, char trans, integer m,
                    integer n, const float* ap, const float* tau,
                    float* c, integer ldc)
{
    return c_sopmtr(side, uplo, trans, m, n, ap, tau, c, ldc);
}

inline integer upmtr( char side, char uplo, char trans, integer m,
                    integer n, const double* ap, const double* tau,
                    double* c, integer ldc)
{
    return c_dopmtr(side, uplo, trans, m, n, ap, tau, c, ldc);
}

inline integer upmtr( char side, char uplo, char trans, integer m,
                    integer n, const scomplex* ap, const scomplex* tau,
                    scomplex* c, integer ldc)
{
    return c_cupmtr(side, uplo, trans, m, n, ap, tau, c, ldc);
}

inline integer upmtr( char side, char uplo, char trans, integer m,
                    integer n, const dcomplex* ap, const dcomplex* tau,
                    dcomplex* c, integer ldc)
{
    return c_zupmtr(side, uplo, trans, m, n, ap, tau, c, ldc);
}

inline integer hbtrd( char vect, char uplo, integer n, integer kd,
                    float* ab, integer ldab, float* d, float* e,
                    float* q, integer ldq)
{
    return c_ssbtrd(vect, uplo, n, kd, ab, ldab, d, e, q, ldq);
}

inline integer hbtrd( char vect, char uplo, integer n, integer kd,
                    double* ab, integer ldab, double* d, double* e,
                    double* q, integer ldq)
{
    return c_dsbtrd(vect, uplo, n, kd, ab, ldab, d, e, q, ldq);
}

inline integer hbtrd( char vect, char uplo, integer n, integer kd,
                    scomplex* ab, integer ldab, float* d, float* e,
                    scomplex* q, integer ldq)
{
    return c_chbtrd(vect, uplo, n, kd, ab, ldab, d, e, q, ldq);
}

inline integer hbtrd( char vect, char uplo, integer n, integer kd,
                    dcomplex* ab, integer ldab, double* d, double* e,
                    dcomplex* q, integer ldq)
{
    return c_zhbtrd(vect, uplo, n, kd, ab, ldab, d, e, q, ldq);
}

inline integer sterf( integer n, float* d, float* e )
{
    return c_ssterf(n, d, e);
}

inline integer sterf( integer n, double* d, double* e )
{
    return c_dsterf(n, d, e);
}

inline integer steqr( char compz, integer n, float* d, float* e, float* z,
                    integer ldz)
{
    return c_ssteqr(compz, n, d, e, z, ldz);
}

inline integer steqr( char compz, integer n, double* d, double* e, double* z,
                    integer ldz)
{
    return c_dsteqr(compz, n, d, e, z, ldz);
}

inline integer steqr( char compz, integer n, float* d, float* e, scomplex* z,
                    integer ldz)
{
    return c_csteqr(compz, n, d, e, z, ldz);
}

inline integer steqr( char compz, integer n, double* d, double* e, dcomplex* z,
                    integer ldz)
{
    return c_zsteqr(compz, n, d, e, z, ldz);
}

inline integer stemr( char jobz, char range, integer n, float* d,
                    float* e, float vl, float vu, integer il,
                    integer iu, integer& m, float* w, float* z,
                    integer ldz, integer nzc, integer* isuppz,
                    logical& tryrac)
{
    return c_sstemr(jobz, range, n, d, e, vl, vu, il, iu, &m, w, z, ldz, nzc, isuppz, &tryrac);
}

inline integer stemr( char jobz, char range, integer n, double* d,
                    double* e, double vl, double vu, integer il,
                    integer iu, integer& m, double* w, double* z,
                    integer ldz, integer nzc, integer* isuppz,
                    logical& tryrac)
{
    return c_dstemr(jobz, range, n, d, e, vl, vu, il, iu, &m, w, z, ldz, nzc, isuppz, &tryrac);
}

inline integer stemr( char jobz, char range, integer n, float* d,
                    float* e, float vl, float vu, integer il,
                    integer iu, integer& m, float* w, scomplex* z,
                    integer ldz, integer nzc, integer* isuppz,
                    logical& tryrac)
{
    return c_cstemr(jobz, range, n, d, e, vl, vu, il, iu, &m, w, z, ldz, nzc, isuppz, &tryrac);
}

inline integer stemr( char jobz, char range, integer n, double* d,
                    double* e, double vl, double vu, integer il,
                    integer iu, integer& m, double* w, dcomplex* z,
                    integer ldz, integer nzc, integer* isuppz,
                    logical& tryrac)
{
    return c_zstemr(jobz, range, n, d, e, vl, vu, il, iu, &m, w, z, ldz, nzc, isuppz, &tryrac);
}

inline integer stedc( char compz, integer n, float* d, float* e, float* z,
                    integer ldz)
{
    return c_sstedc(compz, n, d, e, z, ldz);
}

inline integer stedc( char compz, integer n, double* d, double* e, double* z,
                    integer ldz)
{
    return c_dstedc(compz, n, d, e, z, ldz);
}

inline integer stedc( char compz, integer n, float* d, float* e, scomplex* z,
                    integer ldz)
{
    return c_cstedc(compz, n, d, e, z, ldz);
}

inline integer stedc( char compz, integer n, double* d, double* e, dcomplex* z,
                    integer ldz)
{
    return c_zstedc(compz, n, d, e, z, ldz);
}

inline integer stegr( char jobz, char range, integer n, float* d,
                    float* e, float vl, float vu, integer il,
                    integer iu, float abstol, integer& m, float* w,
                    float* z, integer ldz, integer* isuppz)
{
    return c_sstegr(jobz, range, n, d, e, vl, vu, il, iu, abstol, &m, w, z, ldz, isuppz);
}

inline integer stegr( char jobz, char range, integer n, double* d,
                    double* e, double vl, double vu, integer il,
                    integer iu, double abstol, integer& m, double* w,
                    double* z, integer ldz, integer* isuppz)
{
    return c_dstegr(jobz, range, n, d, e, vl, vu, il, iu, abstol, &m, w, z, ldz, isuppz);
}

inline integer stegr( char jobz, char range, integer n, float* d,
                    float* e, float vl, float vu, integer il,
                    integer iu, float abstol, integer& m, float* w,
                    scomplex* z, integer ldz, integer* isuppz)
{
    return c_cstegr(jobz, range, n, d, e, vl, vu, il, iu, abstol, &m, w, z, ldz, isuppz);
}

inline integer stegr( char jobz, char range, integer n, double* d,
                    double* e, double vl, double vu, integer il,
                    integer iu, double abstol, integer& m, double* w,
                    dcomplex* z, integer ldz, integer* isuppz)
{
    return c_zstegr(jobz, range, n, d, e, vl, vu, il, iu, abstol, &m, w, z, ldz, isuppz);
}

inline integer pteqr( char compz, integer n, float* d, float* e, float* z,
                    integer ldz)
{
    return c_spteqr(compz, n, d, e, z, ldz);
}

inline integer pteqr( char compz, integer n, double* d, double* e, double* z,
                    integer ldz)
{
    return c_dpteqr(compz, n, d, e, z, ldz);
}

inline integer pteqr( char compz, integer n, float* d, float* e, scomplex* z,
                    integer ldz)
{
    return c_cpteqr(compz, n, d, e, z, ldz);
}

inline integer pteqr( char compz, integer n, double* d, double* e, dcomplex* z,
                    integer ldz)
{
    return c_zpteqr(compz, n, d, e, z, ldz);
}

inline integer stebz( char range, char order, integer n, float vl,
                    float vu, integer il, integer iu, float abstol,
                    const float* d, const float* e, integer& m,
                    integer& nsplit, float* w, integer* iblock,
                    integer* isplit)
{
    return c_sstebz(range, order, n, vl, vu, il, iu, abstol, d, e, &m, &nsplit, w, iblock, isplit);
}

inline integer stebz( char range, char order, integer n, double vl,
                    double vu, integer il, integer iu, double abstol,
                    const double* d, const double* e, integer& m,
                    integer& nsplit, double* w, integer* iblock,
                    integer* isplit)
{
    return c_dstebz(range, order, n, vl, vu, il, iu, abstol, d, e, &m, &nsplit, w, iblock, isplit);
}

inline integer stein( integer n, const float* d, const float* e,
                    integer m, const float* w, const integer* iblock,
                    const integer* isplit, float* z, integer ldz,
                    integer* ifailv )
{
    return c_sstein(n, d, e, m, w, iblock, isplit, z, ldz, ifailv);
}

inline integer stein( integer n, const double* d, const double* e,
                    integer m, const double* w, const integer* iblock,
                    const integer* isplit, double* z, integer ldz,
                    integer* ifailv )
{
    return c_dstein(n, d, e, m, w, iblock, isplit, z, ldz, ifailv);
}

inline integer stein( integer n, const float* d, const float* e,
                    integer m, const float* w, const integer* iblock,
                    const integer* isplit, scomplex* z, integer ldz,
                    integer* ifailv )
{
    return c_cstein(n, d, e, m, w, iblock, isplit, z, ldz, ifailv);
}

inline integer stein( integer n, const double* d, const double* e,
                    integer m, const double* w, const integer* iblock,
                    const integer* isplit, dcomplex* z, integer ldz,
                    integer* ifailv )
{
    return c_zstein(n, d, e, m, w, iblock, isplit, z, ldz, ifailv);
}

inline integer disna( char job, integer m, integer n, const float* d,
                    float* sep )
{
    return c_sdisna(job, m, n, d, sep);
}

inline integer disna( char job, integer m, integer n, const double* d,
                    double* sep )
{
    return c_ddisna(job, m, n, d, sep);
}

inline integer hegst( integer itype, char uplo, integer n, float* a,
                    integer lda, const float* b, integer ldb )
{
    return c_ssygst(itype, uplo, n, a, lda, b, ldb);
}

inline integer hegst( integer itype, char uplo, integer n, double* a,
                    integer lda, const double* b, integer ldb )
{
    return c_dsygst(itype, uplo, n, a, lda, b, ldb);
}

inline integer hegst( integer itype, char uplo, integer n, scomplex* a,
                    integer lda, const scomplex* b, integer ldb )
{
    return c_chegst(itype, uplo, n, a, lda, b, ldb);
}

inline integer hegst( integer itype, char uplo, integer n, dcomplex* a,
                    integer lda, const dcomplex* b, integer ldb )
{
    return c_zhegst(itype, uplo, n, a, lda, b, ldb);
}

inline integer hpgst( integer itype, char uplo, integer n, float* ap,
                    const float* bp )
{
    return c_sspgst(itype, uplo, n, ap, bp);
}

inline integer hpgst( integer itype, char uplo, integer n, double* ap,
                    const double* bp )
{
    return c_dspgst(itype, uplo, n, ap, bp);
}

inline integer hpgst( integer itype, char uplo, integer n, scomplex* ap,
                    const scomplex* bp )
{
    return c_chpgst(itype, uplo, n, ap, bp);
}

inline integer hpgst( integer itype, char uplo, integer n, dcomplex* ap,
                    const dcomplex* bp )
{
    return c_zhpgst(itype, uplo, n, ap, bp);
}

inline integer hbgst( char vect, char uplo, integer n, integer ka,
                    integer kb, float* ab, integer ldab,
                    const float* bb, integer ldbb, float* x,
                    integer ldx)
{
    return c_ssbgst(vect, uplo, n, ka, kb, ab, ldab, bb, ldbb, x, ldx);
}

inline integer hbgst( char vect, char uplo, integer n, integer ka,
                    integer kb, double* ab, integer ldab,
                    const double* bb, integer ldbb, double* x,
                    integer ldx)
{
    return c_dsbgst(vect, uplo, n, ka, kb, ab, ldab, bb, ldbb, x, ldx);
}

inline integer hbgst( char vect, char uplo, integer n, integer ka,
                    integer kb, scomplex* ab, integer ldab,
                    const scomplex* bb, integer ldbb, scomplex* x,
                    integer ldx)
{
    return c_chbgst(vect, uplo, n, ka, kb, ab, ldab, bb, ldbb, x, ldx);
}

inline integer hbgst( char vect, char uplo, integer n, integer ka,
                    integer kb, dcomplex* ab, integer ldab,
                    const dcomplex* bb, integer ldbb, dcomplex* x,
                    integer ldx)
{
    return c_zhbgst(vect, uplo, n, ka, kb, ab, ldab, bb, ldbb, x, ldx);
}

inline integer pbstf( char uplo, integer n, integer kb, float* bb,
                    integer ldbb )
{
    return c_spbstf(uplo, n, kb, bb, ldbb);
}

inline integer pbstf( char uplo, integer n, integer kb, double* bb,
                    integer ldbb )
{
    return c_dpbstf(uplo, n, kb, bb, ldbb);
}

inline integer pbstf( char uplo, integer n, integer kb, scomplex* bb,
                    integer ldbb )
{
    return c_cpbstf(uplo, n, kb, bb, ldbb);
}

inline integer pbstf( char uplo, integer n, integer kb, dcomplex* bb,
                    integer ldbb )
{
    return c_zpbstf(uplo, n, kb, bb, ldbb);
}

inline integer gehrd( integer n, integer ilo, integer ihi, float* a,
                    integer lda, float* tau)
{
    return c_sgehrd(n, ilo, ihi, a, lda, tau);
}

inline integer gehrd( integer n, integer ilo, integer ihi, double* a,
                    integer lda, double* tau)
{
    return c_dgehrd(n, ilo, ihi, a, lda, tau);
}

inline integer gehrd( integer n, integer ilo, integer ihi, scomplex* a,
                    integer lda, scomplex* tau)
{
    return c_cgehrd(n, ilo, ihi, a, lda, tau);
}

inline integer gehrd( integer n, integer ilo, integer ihi, dcomplex* a,
                    integer lda, dcomplex* tau)
{
    return c_zgehrd(n, ilo, ihi, a, lda, tau);
}

inline integer unghr( integer n, integer ilo, integer ihi, float* a,
                    integer lda, const float* tau)
{
    return c_sorghr(n, ilo, ihi, a, lda, tau);
}

inline integer unghr( integer n, integer ilo, integer ihi, double* a,
                    integer lda, const double* tau)
{
    return c_dorghr(n, ilo, ihi, a, lda, tau);
}

inline integer unghr( integer n, integer ilo, integer ihi, scomplex* a,
                    integer lda, const scomplex* tau)
{
    return c_cunghr(n, ilo, ihi, a, lda, tau);
}

inline integer unghr( integer n, integer ilo, integer ihi, dcomplex* a,
                    integer lda, const dcomplex* tau)
{
    return c_zunghr(n, ilo, ihi, a, lda, tau);
}

inline integer unmhr( char side, char trans, integer m, integer n,
                    integer ilo, integer ihi, const float* a,
                    integer lda, const float* tau, float* c,
                    integer ldc)
{
    return c_sormhr(side, trans, m, n, ilo, ihi, a, lda, tau, c, ldc);
}

inline integer unmhr( char side, char trans, integer m, integer n,
                    integer ilo, integer ihi, const double* a,
                    integer lda, const double* tau, double* c,
                    integer ldc)
{
    return c_dormhr(side, trans, m, n, ilo, ihi, a, lda, tau, c, ldc);
}

inline integer unmhr( char side, char trans, integer m, integer n,
                    integer ilo, integer ihi, const scomplex* a,
                    integer lda, const scomplex* tau, scomplex* c,
                    integer ldc)
{
    return c_cunmhr(side, trans, m, n, ilo, ihi, a, lda, tau, c, ldc);
}

inline integer unmhr( char side, char trans, integer m, integer n,
                    integer ilo, integer ihi, const dcomplex* a,
                    integer lda, const dcomplex* tau, dcomplex* c,
                    integer ldc)
{
    return c_zunmhr(side, trans, m, n, ilo, ihi, a, lda, tau, c, ldc);
}

inline integer gebal( char job, integer n, float* a, integer lda,
                    integer& ilo, integer& ihi, float* scale )
{
    return c_sgebal(job, n, a, lda, &ilo, &ihi, scale);
}

inline integer gebal( char job, integer n, double* a, integer lda,
                    integer& ilo, integer& ihi, double* scale )
{
    return c_dgebal(job, n, a, lda, &ilo, &ihi, scale);
}

inline integer gebal( char job, integer n, scomplex* a, integer lda,
                    integer& ilo, integer& ihi, float* scale )
{
    return c_cgebal(job, n, a, lda, &ilo, &ihi, scale);
}

inline integer gebal( char job, integer n, dcomplex* a, integer lda,
                    integer& ilo, integer& ihi, double* scale )
{
    return c_zgebal(job, n, a, lda, &ilo, &ihi, scale);
}

inline integer gebak( char job, char side, integer n, integer ilo,
                    integer ihi, const float* scale, integer m,
                    float* v, integer ldv )
{
    return c_sgebak(job, side, n, ilo, ihi, scale, m, v, ldv);
}

inline integer gebak( char job, char side, integer n, integer ilo,
                    integer ihi, const double* scale, integer m,
                    double* v, integer ldv )
{
    return c_dgebak(job, side, n, ilo, ihi, scale, m, v, ldv);
}

inline integer gebak( char job, char side, integer n, integer ilo,
                    integer ihi, const float* scale, integer m,
                    scomplex* v, integer ldv )
{
    return c_cgebak(job, side, n, ilo, ihi, scale, m, v, ldv);
}

inline integer gebak( char job, char side, integer n, integer ilo,
                    integer ihi, const double* scale, integer m,
                    dcomplex* v, integer ldv )
{
    return c_zgebak(job, side, n, ilo, ihi, scale, m, v, ldv);
}

inline integer hseqr( char job, char compz, integer n, integer ilo,
                    integer ihi, float* h, integer ldh, scomplex* w,
                    float* z, integer ldz)
{
    return c_shseqr(job, compz, n, ilo, ihi, h, ldh, w, z, ldz);
}

inline integer hseqr( char job, char compz, integer n, integer ilo,
                    integer ihi, double* h, integer ldh, dcomplex* w,
                    double* z, integer ldz)
{
    return c_dhseqr(job, compz, n, ilo, ihi, h, ldh, w, z, ldz);
}

inline integer hseqr( char job, char compz, integer n, integer ilo,
                    integer ihi, scomplex* h, integer ldh, scomplex* w,
                    scomplex* z, integer ldz)
{
    return c_chseqr(job, compz, n, ilo, ihi, h, ldh, w, z, ldz);
}

inline integer hseqr( char job, char compz, integer n, integer ilo,
                    integer ihi, dcomplex* h, integer ldh, dcomplex* w,
                    dcomplex* z, integer ldz)
{
    return c_zhseqr(job, compz, n, ilo, ihi, h, ldh, w, z, ldz);
}

inline integer hsein( char job, char eigsrc, char initv,
                    logical* select, integer n, const float* h,
                    integer ldh, const scomplex* w, float* vl,
                    integer ldvl, float* vr, integer ldvr,
                    integer mm, integer& m,
                    integer* ifaill, integer* ifailr )
{
    return c_shsein(job, eigsrc, initv, select, n, h, ldh, w, vl, ldvl, vr, ldvr, mm, &m, ifaill, ifailr);
}

inline integer hsein( char job, char eigsrc, char initv,
                    logical* select, integer n, const double* h,
                    integer ldh, const dcomplex* w, double* vl,
                    integer ldvl, double* vr, integer ldvr,
                    integer mm, integer& m,
                    integer* ifaill, integer* ifailr )
{
    return c_dhsein(job, eigsrc, initv, select, n, h, ldh, w, vl, ldvl, vr, ldvr, mm, &m, ifaill, ifailr);
}

inline integer hsein( char job, char eigsrc, char initv,
                    logical* select, integer n, const scomplex* h,
                    integer ldh, scomplex* w, scomplex* vl,
                    integer ldvl, scomplex* vr, integer ldvr,
                    integer mm, integer& m,
                    integer* ifaill, integer* ifailr )
{
    return c_chsein(job, eigsrc, initv, select, n, h, ldh, w, vl, ldvl, vr, ldvr, mm, &m, ifaill, ifailr);
}

inline integer hsein( char job, char eigsrc, char initv,
                    logical* select, integer n, const dcomplex* h,
                    integer ldh, dcomplex* w, dcomplex* vl,
                    integer ldvl, dcomplex* vr, integer ldvr,
                    integer mm, integer& m,
                    integer* ifaill, integer* ifailr )
{
    return c_zhsein(job, eigsrc, initv, select, n, h, ldh, w, vl, ldvl, vr, ldvr, mm, &m, ifaill, ifailr);
}

inline integer trevc( char side, char howmny, const logical* select,
                    integer n, const float* t, integer ldt, float* vl,
                    integer ldvl, float* vr, integer ldvr,
                    integer mm, integer& m)
{
    return c_strevc(side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, &m);
}

inline integer trevc( char side, char howmny, const logical* select,
                    integer n, const double* t, integer ldt, double* vl,
                    integer ldvl, double* vr, integer ldvr,
                    integer mm, integer& m)
{
    return c_dtrevc(side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, &m);
}

inline integer trevc( char side, char howmny, const logical* select,
                    integer n, const scomplex* t, integer ldt, scomplex* vl,
                    integer ldvl, scomplex* vr, integer ldvr,
                    integer mm, integer& m)
{
    return c_ctrevc(side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, &m);
}

inline integer trevc( char side, char howmny, const logical* select,
                    integer n, const dcomplex* t, integer ldt, dcomplex* vl,
                    integer ldvl, dcomplex* vr, integer ldvr,
                    integer mm, integer& m)
{
    return c_ztrevc(side, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, mm, &m);
}

inline integer trsna( char job, char howmny, const logical* select,
                    integer n, const float* t, integer ldt,
                    const float* vl, integer ldvl, const float* vr,
                    integer ldvr, float* s, float* sep, integer mm,
                    integer& m)
{
    return c_strsna(job, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, s, sep, mm, &m);
}

inline integer trsna( char job, char howmny, const logical* select,
                    integer n, const double* t, integer ldt,
                    const double* vl, integer ldvl, const double* vr,
                    integer ldvr, double* s, double* sep, integer mm,
                    integer& m)
{
    return c_dtrsna(job, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, s, sep, mm, &m);
}

inline integer trsna( char job, char howmny, const logical* select,
                    integer n, const scomplex* t, integer ldt,
                    const scomplex* vl, integer ldvl, const scomplex* vr,
                    integer ldvr, float* s, float* sep, integer mm,
                    integer& m)
{
    return c_ctrsna(job, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, s, sep, mm, &m);
}

inline integer trsna( char job, char howmny, const logical* select,
                    integer n, const dcomplex* t, integer ldt,
                    const dcomplex* vl, integer ldvl, const dcomplex* vr,
                    integer ldvr, double* s, double* sep, integer mm,
                    integer& m)
{
    return c_ztrsna(job, howmny, select, n, t, ldt, vl, ldvl, vr, ldvr, s, sep, mm, &m);
}

inline integer trexc( char compq, integer n, float* t, integer ldt,
                    float* q, integer ldq, integer& ifst,
                    integer& ilst)
{
    return c_strexc(compq, n, t, ldt, q, ldq, &ifst, &ilst);
}

inline integer trexc( char compq, integer n, double* t, integer ldt,
                    double* q, integer ldq, integer& ifst,
                    integer& ilst)
{
    return c_dtrexc(compq, n, t, ldt, q, ldq, &ifst, &ilst);
}

inline integer trexc( char compq, integer n, scomplex* t, integer ldt,
                    scomplex* q, integer ldq, integer& ifst,
                    integer& ilst)
{
    return c_ctrexc(compq, n, t, ldt, q, ldq, &ifst, &ilst);
}

inline integer trexc( char compq, integer n, dcomplex* t, integer ldt,
                    dcomplex* q, integer ldq, integer& ifst,
                    integer& ilst)
{
    return c_ztrexc(compq, n, t, ldt, q, ldq, &ifst, &ilst);
}

inline integer trsen( char job, char compq, const logical* select,
                    integer n, float* t, integer ldt, float* q,
                    integer ldq, scomplex* w, integer& m,
                    float& s, float& sep)
{
    return c_strsen(job, compq, select, n, t, ldt, q, ldq, w, &m, &s, &sep);
}

inline integer trsen( char job, char compq, const logical* select,
                    integer n, double* t, integer ldt, double* q,
                    integer ldq, dcomplex* w, integer& m,
                    double& s, double& sep)
{
    return c_dtrsen(job, compq, select, n, t, ldt, q, ldq, w, &m, &s, &sep);
}

inline integer trsen( char job, char compq, const logical* select,
                    integer n, scomplex* t, integer ldt, scomplex* q,
                    integer ldq, scomplex* w, integer& m,
                    float& s, float& sep)
{
    return c_ctrsen(job, compq, select, n, t, ldt, q, ldq, w, &m, &s, &sep);
}

inline integer trsen( char job, char compq, const logical* select,
                    integer n, dcomplex* t, integer ldt, dcomplex* q,
                    integer ldq, dcomplex* w, integer& m,
                    double& s, double& sep)
{
    return c_ztrsen(job, compq, select, n, t, ldt, q, ldq, w, &m, &s, &sep);
}

inline integer trsyl( char trana, char tranb, integer isgn, integer m,
                    integer n, const float* a, integer lda,
                    const float* b, integer ldb, float* c,
                    integer ldc, float& scale )
{
    return c_strsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, &scale);
}

inline integer trsyl( char trana, char tranb, integer isgn, integer m,
                    integer n, const double* a, integer lda,
                    const double* b, integer ldb, double* c,
                    integer ldc, double& scale )
{
    return c_dtrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, &scale);
}

inline integer trsyl( char trana, char tranb, integer isgn, integer m,
                    integer n, const scomplex* a, integer lda,
                    const scomplex* b, integer ldb, scomplex* c,
                    integer ldc, float& scale )
{
    return c_ctrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, &scale);
}

inline integer trsyl( char trana, char tranb, integer isgn, integer m,
                    integer n, const dcomplex* a, integer lda,
                    const dcomplex* b, integer ldb, dcomplex* c,
                    integer ldc, double& scale )
{
    return c_ztrsyl(trana, tranb, isgn, m, n, a, lda, b, ldb, c, ldc, &scale);
}

inline integer gghrd( char compq, char compz, integer n, integer ilo,
                    integer ihi, float* a, integer lda, float* b,
                    integer ldb, float* q, integer ldq, float* z,
                    integer ldz )
{
    return c_sgghrd(compq, compz, n, ilo, ihi, a, lda, b, ldb, q, ldq, z, ldz);
}

inline integer gghrd( char compq, char compz, integer n, integer ilo,
                    integer ihi, double* a, integer lda, double* b,
                    integer ldb, double* q, integer ldq, double* z,
                    integer ldz )
{
    return c_dgghrd(compq, compz, n, ilo, ihi, a, lda, b, ldb, q, ldq, z, ldz);
}

inline integer gghrd( char compq, char compz, integer n, integer ilo,
                    integer ihi, scomplex* a, integer lda, scomplex* b,
                    integer ldb, scomplex* q, integer ldq, scomplex* z,
                    integer ldz )
{
    return c_cgghrd(compq, compz, n, ilo, ihi, a, lda, b, ldb, q, ldq, z, ldz);
}

inline integer gghrd( char compq, char compz, integer n, integer ilo,
                    integer ihi, dcomplex* a, integer lda, dcomplex* b,
                    integer ldb, dcomplex* q, integer ldq, dcomplex* z,
                    integer ldz )
{
    return c_zgghrd(compq, compz, n, ilo, ihi, a, lda, b, ldb, q, ldq, z, ldz);
}

inline integer ggbal( char job, integer n, float* a, integer lda,
                    float* b, integer ldb, integer& ilo,
                    integer& ihi, float* lscale, float* rscale)
{
    return c_sggbal(job, n, a, lda, b, ldb, &ilo, &ihi, lscale, rscale);
}

inline integer ggbal( char job, integer n, double* a, integer lda,
                    double* b, integer ldb, integer& ilo,
                    integer& ihi, double* lscale, double* rscale)
{
    return c_dggbal(job, n, a, lda, b, ldb, &ilo, &ihi, lscale, rscale);
}

inline integer ggbal( char job, integer n, scomplex* a, integer lda,
                    scomplex* b, integer ldb, integer& ilo,
                    integer& ihi, float* lscale, float* rscale)
{
    return c_cggbal(job, n, a, lda, b, ldb, &ilo, &ihi, lscale, rscale);
}

inline integer ggbal( char job, integer n, dcomplex* a, integer lda,
                    dcomplex* b, integer ldb, integer& ilo,
                    integer& ihi, double* lscale, double* rscale)
{
    return c_zggbal(job, n, a, lda, b, ldb, &ilo, &ihi, lscale, rscale);
}

inline integer ggbak( char job, char side, integer n, integer ilo,
                    integer ihi, const float* lscale, const float* rscale,
                    integer m, float* v, integer ldv )
{
    return c_sggbak(job, side, n, ilo, ihi, lscale, rscale, m, v, ldv);
}

inline integer ggbak( char job, char side, integer n, integer ilo,
                    integer ihi, const double* lscale, const double* rscale,
                    integer m, double* v, integer ldv )
{
    return c_dggbak(job, side, n, ilo, ihi, lscale, rscale, m, v, ldv);
}

inline integer ggbak( char job, char side, integer n, integer ilo,
                    integer ihi, const float* lscale, const float* rscale,
                    integer m, scomplex* v, integer ldv )
{
    return c_cggbak(job, side, n, ilo, ihi, lscale, rscale, m, v, ldv);
}

inline integer ggbak( char job, char side, integer n, integer ilo,
                    integer ihi, const double* lscale, const double* rscale,
                    integer m, dcomplex* v, integer ldv )
{
    return c_zggbak(job, side, n, ilo, ihi, lscale, rscale, m, v, ldv);
}

inline integer hgeqz( char job, char compq, char compz, integer n,
                    integer ilo, integer ihi, float* h,
                    integer ldh, float* t, integer ldt, scomplex* alpha,
                    float* beta, float* q, integer ldq,
                    float* z, integer ldz)
{
    return c_shgeqz(job, compq, compz, n, ilo, ihi, h, ldh, t, ldt, alpha, beta, q, ldq, z, ldz);
}

inline integer hgeqz( char job, char compq, char compz, integer n,
                    integer ilo, integer ihi, double* h,
                    integer ldh, double* t, integer ldt, dcomplex* alpha,
                    double* beta, double* q, integer ldq,
                    double* z, integer ldz)
{
    return c_dhgeqz(job, compq, compz, n, ilo, ihi, h, ldh, t, ldt, alpha, beta, q, ldq, z, ldz);
}

inline integer hgeqz( char job, char compq, char compz, integer n,
                    integer ilo, integer ihi, scomplex* h,
                    integer ldh, scomplex* t, integer ldt, scomplex* alpha,
                    scomplex* beta, scomplex* q, integer ldq,
                    scomplex* z, integer ldz)
{
    return c_chgeqz(job, compq, compz, n, ilo, ihi, h, ldh, t, ldt, alpha, beta, q, ldq, z, ldz);
}

inline integer hgeqz( char job, char compq, char compz, integer n,
                    integer ilo, integer ihi, dcomplex* h,
                    integer ldh, dcomplex* t, integer ldt, dcomplex* alpha,
                    dcomplex* beta, dcomplex* q, integer ldq,
                    dcomplex* z, integer ldz)
{
    return c_zhgeqz(job, compq, compz, n, ilo, ihi, h, ldh, t, ldt, alpha, beta, q, ldq, z, ldz);
}

inline integer tgevc( char side, char howmny, const logical* select,
                    integer n, const float* s, integer lds,
                    const float* p, integer ldp, float* vl,
                    integer ldvl, float* vr, integer ldvr,
                    integer mm, integer& m)
{
    return c_stgevc(side, howmny, select, n, s, lds, p, ldp, vl, ldvl, vr, ldvr, mm, &m);
}

inline integer tgevc( char side, char howmny, const logical* select,
                    integer n, const double* s, integer lds,
                    const double* p, integer ldp, double* vl,
                    integer ldvl, double* vr, integer ldvr,
                    integer mm, integer& m)
{
    return c_dtgevc(side, howmny, select, n, s, lds, p, ldp, vl, ldvl, vr, ldvr, mm, &m);
}

inline integer tgevc( char side, char howmny, const logical* select,
                    integer n, const scomplex* s, integer lds,
                    const scomplex* p, integer ldp, scomplex* vl,
                    integer ldvl, scomplex* vr, integer ldvr,
                    integer mm, integer& m)
{
    return c_ctgevc(side, howmny, select, n, s, lds, p, ldp, vl, ldvl, vr, ldvr, mm, &m);
}

inline integer tgevc( char side, char howmny, const logical* select,
                    integer n, const dcomplex* s, integer lds,
                    const dcomplex* p, integer ldp, dcomplex* vl,
                    integer ldvl, dcomplex* vr, integer ldvr,
                    integer mm, integer& m)
{
    return c_ztgevc(side, howmny, select, n, s, lds, p, ldp, vl, ldvl, vr, ldvr, mm, &m);
}

inline integer tgexc( logical wantq, logical wantz, integer n,
                    float* a, integer lda, float* b, integer ldb,
                    float* q, integer ldq, float* z, integer ldz,
                    integer& ifst, integer& ilst)
{
    return c_stgexc(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &ifst, &ilst);
}

inline integer tgexc( logical wantq, logical wantz, integer n,
                    double* a, integer lda, double* b, integer ldb,
                    double* q, integer ldq, double* z, integer ldz,
                    integer& ifst, integer& ilst)
{
    return c_dtgexc(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &ifst, &ilst);
}

inline integer tgexc( logical wantq, logical wantz, integer n,
                    scomplex* a, integer lda, scomplex* b, integer ldb,
                    scomplex* q, integer ldq, scomplex* z, integer ldz,
                    integer& ifst, integer& ilst)
{
    return c_ctgexc(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &ifst, &ilst);
}

inline integer tgexc( logical wantq, logical wantz, integer n,
                    dcomplex* a, integer lda, dcomplex* b, integer ldb,
                    dcomplex* q, integer ldq, dcomplex* z, integer ldz,
                    integer& ifst, integer& ilst)
{
    return c_ztgexc(wantq, wantz, n, a, lda, b, ldb, q, ldq, z, ldz, &ifst, &ilst);
}

inline integer tgsen( integer ijob, logical wantq,
                    logical wantz, const logical* select,
                    integer n, float* a, integer lda, float* b,
                    integer ldb, scomplex* alpha,
                    float* beta, float* q, integer ldq, float* z,
                    integer ldz, integer& m, float& pl, float& pr,
                    float* dif)
{
    return c_stgsen(ijob, wantq, wantz, select, n, a, lda, b, ldb, alpha, beta, q, ldq, z, ldz, &m, &pl, &pr, dif);
}

inline integer tgsen( integer ijob, logical wantq,
                    logical wantz, const logical* select,
                    integer n, double* a, integer lda, double* b,
                    integer ldb, dcomplex* alpha,
                    double* beta, double* q, integer ldq, double* z,
                    integer ldz, integer& m, double& pl, double& pr,
                    double* dif)
{
    return c_dtgsen(ijob, wantq, wantz, select, n, a, lda, b, ldb, alpha, beta, q, ldq, z, ldz, &m, &pl, &pr, dif);
}

inline integer tgsen( integer ijob, logical wantq,
                    logical wantz, const logical* select,
                    integer n, scomplex* a, integer lda, scomplex* b,
                    integer ldb, scomplex* alpha,
                    scomplex* beta, scomplex* q, integer ldq, scomplex* z,
                    integer ldz, integer& m, float& pl, float& pr,
                    float* dif)
{
    return c_ctgsen(ijob, wantq, wantz, select, n, a, lda, b, ldb, alpha, beta, q, ldq, z, ldz, &m, &pl, &pr, dif);
}

inline integer tgsen( integer ijob, logical wantq,
                    logical wantz, const logical* select,
                    integer n, dcomplex* a, integer lda, dcomplex* b,
                    integer ldb, dcomplex* alpha,
                    dcomplex* beta, dcomplex* q, integer ldq, dcomplex* z,
                    integer ldz, integer& m, double& pl, double& pr,
                    double* dif)
{
    return c_ztgsen(ijob, wantq, wantz, select, n, a, lda, b, ldb, alpha, beta, q, ldq, z, ldz, &m, &pl, &pr, dif);
}

inline integer tgsyl( char trans, integer ijob, integer m, integer n,
                    const float* a, integer lda, const float* b,
                    integer ldb, float* c, integer ldc,
                    const float* d, integer ldd, const float* e,
                    integer lde, float* f, integer ldf, float& scale,
                    float& dif)
{
    return c_stgsyl(trans, ijob, m, n, a, lda, b, ldb, c, ldc, d, ldd, e, lde, f, ldf, &scale, &dif);
}

inline integer tgsyl( char trans, integer ijob, integer m, integer n,
                    const double* a, integer lda, const double* b,
                    integer ldb, double* c, integer ldc,
                    const double* d, integer ldd, const double* e,
                    integer lde, double* f, integer ldf, double& scale,
                    double& dif)
{
    return c_dtgsyl(trans, ijob, m, n, a, lda, b, ldb, c, ldc, d, ldd, e, lde, f, ldf, &scale, &dif);
}

inline integer tgsyl( char trans, integer ijob, integer m, integer n,
                    const scomplex* a, integer lda, const scomplex* b,
                    integer ldb, scomplex* c, integer ldc,
                    const scomplex* d, integer ldd, const scomplex* e,
                    integer lde, scomplex* f, integer ldf, float& scale,
                    float& dif)
{
    return c_ctgsyl(trans, ijob, m, n, a, lda, b, ldb, c, ldc, d, ldd, e, lde, f, ldf, &scale, &dif);
}

inline integer tgsyl( char trans, integer ijob, integer m, integer n,
                    const dcomplex* a, integer lda, const dcomplex* b,
                    integer ldb, dcomplex* c, integer ldc,
                    const dcomplex* d, integer ldd, const dcomplex* e,
                    integer lde, dcomplex* f, integer ldf, double& scale,
                    double& dif)
{
    return c_ztgsyl(trans, ijob, m, n, a, lda, b, ldb, c, ldc, d, ldd, e, lde, f, ldf, &scale, &dif);
}

inline integer tgsna( char job, char howmny, const logical* select,
                    integer n, const float* a, integer lda,
                    const float* b, integer ldb, const float* vl,
                    integer ldvl, const float* vr, integer ldvr,
                    float* s, float* dif, integer mm, integer& m)
{
    return c_stgsna(job, howmny, select, n, a, lda, b, ldb, vl, ldvl, vr, ldvr, s, dif, mm, &m);
}

inline integer tgsna( char job, char howmny, const logical* select,
                    integer n, const double* a, integer lda,
                    const double* b, integer ldb, const double* vl,
                    integer ldvl, const double* vr, integer ldvr,
                    double* s, double* dif, integer mm, integer& m)
{
    return c_dtgsna(job, howmny, select, n, a, lda, b, ldb, vl, ldvl, vr, ldvr, s, dif, mm, &m);
}

inline integer tgsna( char job, char howmny, const logical* select,
                    integer n, const scomplex* a, integer lda,
                    const scomplex* b, integer ldb, const scomplex* vl,
                    integer ldvl, const scomplex* vr, integer ldvr,
                    float* s, float* dif, integer mm, integer& m)
{
    return c_ctgsna(job, howmny, select, n, a, lda, b, ldb, vl, ldvl, vr, ldvr, s, dif, mm, &m);
}

inline integer tgsna( char job, char howmny, const logical* select,
                    integer n, const dcomplex* a, integer lda,
                    const dcomplex* b, integer ldb, const dcomplex* vl,
                    integer ldvl, const dcomplex* vr, integer ldvr,
                    double* s, double* dif, integer mm, integer& m)
{
    return c_ztgsna(job, howmny, select, n, a, lda, b, ldb, vl, ldvl, vr, ldvr, s, dif, mm, &m);
}

inline integer ggsvp( char jobu, char jobv, char jobq, integer m,
                    integer p, integer n, float* a, integer lda,
                    float* b, integer ldb, float tola, float tolb,
                    integer* k, integer* l, float* u, integer ldu,
                    float* v, integer ldv, float* q, integer ldq)
{
    return c_sggsvp(jobu, jobv, jobq, m, p, n, a, lda, b, ldb, tola, tolb, k, l, u, ldu, v, ldv, q, ldq);
}

inline integer ggsvp( char jobu, char jobv, char jobq, integer m,
                    integer p, integer n, double* a, integer lda,
                    double* b, integer ldb, double tola, double tolb,
                    integer* k, integer* l, double* u, integer ldu,
                    double* v, integer ldv, double* q, integer ldq)
{
    return c_dggsvp(jobu, jobv, jobq, m, p, n, a, lda, b, ldb, tola, tolb, k, l, u, ldu, v, ldv, q, ldq);
}

inline integer ggsvp( char jobu, char jobv, char jobq, integer m,
                    integer p, integer n, scomplex* a, integer lda,
                    scomplex* b, integer ldb, float tola, float tolb,
                    integer* k, integer* l, scomplex* u, integer ldu,
                    scomplex* v, integer ldv, scomplex* q, integer ldq)
{
    return c_cggsvp(jobu, jobv, jobq, m, p, n, a, lda, b, ldb, tola, tolb, k, l, u, ldu, v, ldv, q, ldq);
}

inline integer ggsvp( char jobu, char jobv, char jobq, integer m,
                    integer p, integer n, dcomplex* a, integer lda,
                    dcomplex* b, integer ldb, double tola, double tolb,
                    integer* k, integer* l, dcomplex* u, integer ldu,
                    dcomplex* v, integer ldv, dcomplex* q, integer ldq)
{
    return c_zggsvp(jobu, jobv, jobq, m, p, n, a, lda, b, ldb, tola, tolb, k, l, u, ldu, v, ldv, q, ldq);
}

inline integer tgsja( char jobu, char jobv, char jobq, integer m,
                    integer p, integer n, integer k, integer l,
                    float* a, integer lda, float* b, integer ldb,
                    float tola, float tolb, float* alpha, float* beta,
                    float* u, integer ldu, float* v, integer ldv,
                    float* q, integer ldq, integer& ncycle )
{
    return c_stgsja(jobu, jobv, jobq, m, p, n, k, l, a, lda, b, ldb, tola, tolb, alpha, beta, u, ldu, v, ldv, q, ldq, &ncycle);
}

inline integer tgsja( char jobu, char jobv, char jobq, integer m,
                    integer p, integer n, integer k, integer l,
                    double* a, integer lda, double* b, integer ldb,
                    double tola, double tolb, double* alpha, double* beta,
                    double* u, integer ldu, double* v, integer ldv,
                    double* q, integer ldq, integer& ncycle )
{
    return c_dtgsja(jobu, jobv, jobq, m, p, n, k, l, a, lda, b, ldb, tola, tolb, alpha, beta, u, ldu, v, ldv, q, ldq, &ncycle);
}

inline integer tgsja( char jobu, char jobv, char jobq, integer m,
                    integer p, integer n, integer k, integer l,
                    scomplex* a, integer lda, scomplex* b, integer ldb,
                    float tola, float tolb, float* alpha, float* beta,
                    scomplex* u, integer ldu, scomplex* v, integer ldv,
                    scomplex* q, integer ldq, integer& ncycle )
{
    return c_ctgsja(jobu, jobv, jobq, m, p, n, k, l, a, lda, b, ldb, tola, tolb, alpha, beta, u, ldu, v, ldv, q, ldq, &ncycle);
}

inline integer tgsja( char jobu, char jobv, char jobq, integer m,
                    integer p, integer n, integer k, integer l,
                    dcomplex* a, integer lda, dcomplex* b, integer ldb,
                    double tola, double tolb, double* alpha, double* beta,
                    dcomplex* u, integer ldu, dcomplex* v, integer ldv,
                    dcomplex* q, integer ldq, integer& ncycle )
{
    return c_ztgsja(jobu, jobv, jobq, m, p, n, k, l, a, lda, b, ldb, tola, tolb, alpha, beta, u, ldu, v, ldv, q, ldq, &ncycle);
}

inline integer gels( char trans, integer m, integer n, integer nrhs,
                   float* a, integer lda, float* b, integer ldb)
{
    return c_sgels(trans, m, n, nrhs, a, lda, b, ldb);
}

inline integer gels( char trans, integer m, integer n, integer nrhs,
                   double* a, integer lda, double* b, integer ldb)
{
    return c_dgels(trans, m, n, nrhs, a, lda, b, ldb);
}

inline integer gels( char trans, integer m, integer n, integer nrhs,
                   scomplex* a, integer lda, scomplex* b, integer ldb)
{
    return c_cgels(trans, m, n, nrhs, a, lda, b, ldb);
}

inline integer gels( char trans, integer m, integer n, integer nrhs,
                   dcomplex* a, integer lda, dcomplex* b, integer ldb)
{
    return c_zgels(trans, m, n, nrhs, a, lda, b, ldb);
}

inline integer gelsy( integer m, integer n, integer nrhs, float* a,
                    integer lda, float* b, integer ldb,
                    integer* jpvt, float rcond, integer& rank)
{
    return c_sgelsy(m, n, nrhs, a, lda, b, ldb, jpvt, rcond, &rank);
}

inline integer gelsy( integer m, integer n, integer nrhs, double* a,
                    integer lda, double* b, integer ldb,
                    integer* jpvt, double rcond, integer& rank)
{
    return c_dgelsy(m, n, nrhs, a, lda, b, ldb, jpvt, rcond, &rank);
}

inline integer gelsy( integer m, integer n, integer nrhs, scomplex* a,
                    integer lda, scomplex* b, integer ldb,
                    integer* jpvt, float rcond, integer& rank)
{
    return c_cgelsy(m, n, nrhs, a, lda, b, ldb, jpvt, rcond, &rank);
}

inline integer gelsy( integer m, integer n, integer nrhs, dcomplex* a,
                    integer lda, dcomplex* b, integer ldb,
                    integer* jpvt, double rcond, integer& rank)
{
    return c_zgelsy(m, n, nrhs, a, lda, b, ldb, jpvt, rcond, &rank);
}

inline integer gelss( integer m, integer n, integer nrhs, float* a,
                    integer lda, float* b, integer ldb, float* s,
                    float rcond, integer& rank)
{
    return c_sgelss(m, n, nrhs, a, lda, b, ldb, s, rcond, &rank);
}

inline integer gelss( integer m, integer n, integer nrhs, double* a,
                    integer lda, double* b, integer ldb, double* s,
                    double rcond, integer& rank)
{
    return c_dgelss(m, n, nrhs, a, lda, b, ldb, s, rcond, &rank);
}

inline integer gelss( integer m, integer n, integer nrhs, scomplex* a,
                    integer lda, scomplex* b, integer ldb, float* s,
                    float rcond, integer& rank)
{
    return c_cgelss(m, n, nrhs, a, lda, b, ldb, s, rcond, &rank);
}

inline integer gelss( integer m, integer n, integer nrhs, dcomplex* a,
                    integer lda, dcomplex* b, integer ldb, double* s,
                    double rcond, integer& rank)
{
    return c_zgelss(m, n, nrhs, a, lda, b, ldb, s, rcond, &rank);
}

inline integer gglse( integer m, integer n, integer p, float* a,
                    integer lda, float* b, integer ldb, float* c,
                    float* d, float* x)
{
    return c_sgglse(m, n, p, a, lda, b, ldb, c, d, x);
}

inline integer gglse( integer m, integer n, integer p, double* a,
                    integer lda, double* b, integer ldb, double* c,
                    double* d, double* x)
{
    return c_dgglse(m, n, p, a, lda, b, ldb, c, d, x);
}

inline integer gglse( integer m, integer n, integer p, scomplex* a,
                    integer lda, scomplex* b, integer ldb, scomplex* c,
                    scomplex* d, scomplex* x)
{
    return c_cgglse(m, n, p, a, lda, b, ldb, c, d, x);
}

inline integer gglse( integer m, integer n, integer p, dcomplex* a,
                    integer lda, dcomplex* b, integer ldb, dcomplex* c,
                    dcomplex* d, dcomplex* x)
{
    return c_zgglse(m, n, p, a, lda, b, ldb, c, d, x);
}

inline integer ggglm( integer n, integer m, integer p, float* a,
                    integer lda, float* b, integer ldb, float* d,
                    float* x, float* y)
{
    return c_sggglm(n, m, p, a, lda, b, ldb, d, x, y);
}

inline integer ggglm( integer n, integer m, integer p, double* a,
                    integer lda, double* b, integer ldb, double* d,
                    double* x, double* y)
{
    return c_dggglm(n, m, p, a, lda, b, ldb, d, x, y);
}

inline integer ggglm( integer n, integer m, integer p, scomplex* a,
                    integer lda, scomplex* b, integer ldb, scomplex* d,
                    scomplex* x, scomplex* y)
{
    return c_cggglm(n, m, p, a, lda, b, ldb, d, x, y);
}

inline integer ggglm( integer n, integer m, integer p, dcomplex* a,
                    integer lda, dcomplex* b, integer ldb, dcomplex* d,
                    dcomplex* x, dcomplex* y)
{
    return c_zggglm(n, m, p, a, lda, b, ldb, d, x, y);
}

inline integer heev( char jobz, char uplo, integer n, float* a,
                   integer lda, float* w)
{
    return c_ssyev(jobz, uplo, n, a, lda, w);
}

inline integer heev( char jobz, char uplo, integer n, double* a,
                   integer lda, double* w)
{
    return c_dsyev(jobz, uplo, n, a, lda, w);
}

inline integer heev( char jobz, char uplo, integer n, scomplex* a,
                   integer lda, float* w)
{
    return c_cheev(jobz, uplo, n, a, lda, w);
}

inline integer heev( char jobz, char uplo, integer n, dcomplex* a,
                   integer lda, double* w)
{
    return c_zheev(jobz, uplo, n, a, lda, w);
}

inline integer heevd( char jobz, char uplo, integer n, float* a,
                    integer lda, float* w)
{
    return c_ssyevd(jobz, uplo, n, a, lda, w);
}

inline integer heevd( char jobz, char uplo, integer n, double* a,
                    integer lda, double* w)
{
    return c_dsyevd(jobz, uplo, n, a, lda, w);
}

inline integer heevd( char jobz, char uplo, integer n, scomplex* a,
                    integer lda, float* w)
{
    return c_cheevd(jobz, uplo, n, a, lda, w);
}

inline integer heevd( char jobz, char uplo, integer n, dcomplex* a,
                    integer lda, double* w)
{
    return c_zheevd(jobz, uplo, n, a, lda, w);
}

inline integer heevx( char jobz, char range, char uplo, integer n,
                    float* a, integer lda, float vl, float vu,
                    integer il, integer iu, float abstol,
                    integer& m, float* w, float* z, integer ldz,
                    integer* ifail )
{
    return c_ssyevx(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer heevx( char jobz, char range, char uplo, integer n,
                    double* a, integer lda, double vl, double vu,
                    integer il, integer iu, double abstol,
                    integer& m, double* w, double* z, integer ldz,
                    integer* ifail )
{
    return c_dsyevx(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer heevx( char jobz, char range, char uplo, integer n,
                    scomplex* a, integer lda, float vl, float vu,
                    integer il, integer iu, float abstol,
                    integer& m, float* w, scomplex* z, integer ldz,
                    integer* ifail )
{
    return c_cheevx(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer heevx( char jobz, char range, char uplo, integer n,
                    dcomplex* a, integer lda, double vl, double vu,
                    integer il, integer iu, double abstol,
                    integer& m, double* w, dcomplex* z, integer ldz,
                    integer* ifail )
{
    return c_zheevx(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer heevr( char jobz, char range, char uplo, integer n,
                    float* a, integer lda, float vl, float vu,
                    integer il, integer iu, float abstol,
                    integer& m, float* w, float* z, integer ldz,
                    integer* isuppz)
{
    return c_ssyevr(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, &m, w, z, ldz, isuppz);
}

inline integer heevr( char jobz, char range, char uplo, integer n,
                    double* a, integer lda, double vl, double vu,
                    integer il, integer iu, double abstol,
                    integer& m, double* w, double* z, integer ldz,
                    integer* isuppz)
{
    return c_dsyevr(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, &m, w, z, ldz, isuppz);
}

inline integer heevr( char jobz, char range, char uplo, integer n,
                    scomplex* a, integer lda, float vl, float vu,
                    integer il, integer iu, float abstol,
                    integer& m, float* w, scomplex* z, integer ldz,
                    integer* isuppz)
{
    return c_cheevr(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, &m, w, z, ldz, isuppz);
}

inline integer heevr( char jobz, char range, char uplo, integer n,
                    dcomplex* a, integer lda, double vl, double vu,
                    integer il, integer iu, double abstol,
                    integer& m, double* w, dcomplex* z, integer ldz,
                    integer* isuppz)
{
    return c_zheevr(jobz, range, uplo, n, a, lda, vl, vu, il, iu, abstol, &m, w, z, ldz, isuppz);
}

inline integer hpev( char jobz, char uplo, integer n, float* ap, float* w,
                   float* z, integer ldz)
{
    return c_sspev(jobz, uplo, n, ap, w, z, ldz);
}

inline integer hpev( char jobz, char uplo, integer n, double* ap, double* w,
                   double* z, integer ldz)
{
    return c_dspev(jobz, uplo, n, ap, w, z, ldz);
}

inline integer hpev( char jobz, char uplo, integer n, scomplex* ap, float* w,
                   scomplex* z, integer ldz)
{
    return c_chpev(jobz, uplo, n, ap, w, z, ldz);
}

inline integer hpev( char jobz, char uplo, integer n, dcomplex* ap, double* w,
                   dcomplex* z, integer ldz)
{
    return c_zhpev(jobz, uplo, n, ap, w, z, ldz);
}

inline integer hpevd( char jobz, char uplo, integer n, float* ap,
                    float* w, float* z, integer ldz)
{
    return c_sspevd(jobz, uplo, n, ap, w, z, ldz);
}

inline integer hpevd( char jobz, char uplo, integer n, double* ap,
                    double* w, double* z, integer ldz)
{
    return c_dspevd(jobz, uplo, n, ap, w, z, ldz);
}

inline integer hpevd( char jobz, char uplo, integer n, scomplex* ap,
                    float* w, scomplex* z, integer ldz)
{
    return c_chpevd(jobz, uplo, n, ap, w, z, ldz);
}

inline integer hpevd( char jobz, char uplo, integer n, dcomplex* ap,
                    double* w, dcomplex* z, integer ldz)
{
    return c_zhpevd(jobz, uplo, n, ap, w, z, ldz);
}

inline integer hpevx( char jobz, char range, char uplo, integer n,
                    float* ap, float vl, float vu, integer il,
                    integer iu, float abstol, integer& m, float* w,
                    float* z, integer ldz, integer* ifail )
{
    return c_sspevx(jobz, range, uplo, n, ap, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hpevx( char jobz, char range, char uplo, integer n,
                    double* ap, double vl, double vu, integer il,
                    integer iu, double abstol, integer& m, double* w,
                    double* z, integer ldz, integer* ifail )
{
    return c_dspevx(jobz, range, uplo, n, ap, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hpevx( char jobz, char range, char uplo, integer n,
                    scomplex* ap, float vl, float vu, integer il,
                    integer iu, float abstol, integer& m, float* w,
                    scomplex* z, integer ldz, integer* ifail )
{
    return c_chpevx(jobz, range, uplo, n, ap, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hpevx( char jobz, char range, char uplo, integer n,
                    dcomplex* ap, double vl, double vu, integer il,
                    integer iu, double abstol, integer& m, double* w,
                    dcomplex* z, integer ldz, integer* ifail )
{
    return c_zhpevx(jobz, range, uplo, n, ap, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hbev( char jobz, char uplo, integer n, integer kd,
                   float* ab, integer ldab, float* w, float* z,
                   integer ldz)
{
    return c_ssbev(jobz, uplo, n, kd, ab, ldab, w, z, ldz);
}

inline integer hbev( char jobz, char uplo, integer n, integer kd,
                   double* ab, integer ldab, double* w, double* z,
                   integer ldz)
{
    return c_dsbev(jobz, uplo, n, kd, ab, ldab, w, z, ldz);
}

inline integer hbev( char jobz, char uplo, integer n, integer kd,
                   scomplex* ab, integer ldab, float* w, scomplex* z,
                   integer ldz)
{
    return c_chbev(jobz, uplo, n, kd, ab, ldab, w, z, ldz);
}

inline integer hbev( char jobz, char uplo, integer n, integer kd,
                   dcomplex* ab, integer ldab, double* w, dcomplex* z,
                   integer ldz)
{
    return c_zhbev(jobz, uplo, n, kd, ab, ldab, w, z, ldz);
}

inline integer hbevd( char jobz, char uplo, integer n, integer kd,
                    float* ab, integer ldab, float* w, float* z,
                    integer ldz)
{
    return c_ssbevd(jobz, uplo, n, kd, ab, ldab, w, z, ldz);
}

inline integer hbevd( char jobz, char uplo, integer n, integer kd,
                    double* ab, integer ldab, double* w, double* z,
                    integer ldz)
{
    return c_dsbevd(jobz, uplo, n, kd, ab, ldab, w, z, ldz);
}

inline integer hbevd( char jobz, char uplo, integer n, integer kd,
                    scomplex* ab, integer ldab, float* w, scomplex* z,
                    integer ldz)
{
    return c_chbevd(jobz, uplo, n, kd, ab, ldab, w, z, ldz);
}

inline integer hbevd( char jobz, char uplo, integer n, integer kd,
                    dcomplex* ab, integer ldab, double* w, dcomplex* z,
                    integer ldz)
{
    return c_zhbevd(jobz, uplo, n, kd, ab, ldab, w, z, ldz);
}

inline integer hbevx( char jobz, char range, char uplo, integer n,
                    integer kd, float* ab, integer ldab, float* q,
                    integer ldq, float vl, float vu, integer il,
                    integer iu, float abstol, integer& m, float* w,
                    float* z, integer ldz, integer* ifail )
{
    return c_ssbevx(jobz, range, uplo, n, kd, ab, ldab, q, ldq, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hbevx( char jobz, char range, char uplo, integer n,
                    integer kd, double* ab, integer ldab, double* q,
                    integer ldq, double vl, double vu, integer il,
                    integer iu, double abstol, integer& m, double* w,
                    double* z, integer ldz, integer* ifail )
{
    return c_dsbevx(jobz, range, uplo, n, kd, ab, ldab, q, ldq, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hbevx( char jobz, char range, char uplo, integer n,
                    integer kd, scomplex* ab, integer ldab, scomplex* q,
                    integer ldq, float vl, float vu, integer il,
                    integer iu, float abstol, integer& m, float* w,
                    scomplex* z, integer ldz, integer* ifail )
{
    return c_chbevx(jobz, range, uplo, n, kd, ab, ldab, q, ldq, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hbevx( char jobz, char range, char uplo, integer n,
                    integer kd, dcomplex* ab, integer ldab, dcomplex* q,
                    integer ldq, double vl, double vu, integer il,
                    integer iu, double abstol, integer& m, double* w,
                    dcomplex* z, integer ldz, integer* ifail )
{
    return c_zhbevx(jobz, range, uplo, n, kd, ab, ldab, q, ldq, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer stev( char jobz, integer n, float* d, float* e, float* z,
                   integer ldz)
{
    return c_sstev(jobz, n, d, e, z, ldz);
}

inline integer stev( char jobz, integer n, double* d, double* e, double* z,
                   integer ldz)
{
    return c_dstev(jobz, n, d, e, z, ldz);
}

inline integer stevd( char jobz, integer n, float* d, float* e, float* z,
                    integer ldz)
{
    return c_sstevd(jobz, n, d, e, z, ldz);
}

inline integer stevd( char jobz, integer n, double* d, double* e, double* z,
                    integer ldz)
{
    return c_dstevd(jobz, n, d, e, z, ldz);
}

inline integer stevx( char jobz, char range, integer n, float* d,
                    float* e, float vl, float vu, integer il,
                    integer iu, float abstol, integer& m, float* w,
                    float* z, integer ldz, integer* ifail )
{
    return c_sstevx(jobz, range, n, d, e, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer stevx( char jobz, char range, integer n, double* d,
                    double* e, double vl, double vu, integer il,
                    integer iu, double abstol, integer& m, double* w,
                    double* z, integer ldz, integer* ifail )
{
    return c_dstevx(jobz, range, n, d, e, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer stevr( char jobz, char range, integer n, float* d,
                    float* e, float vl, float vu, integer il,
                    integer iu, float abstol, integer& m, float* w,
                    float* z, integer ldz, integer* isuppz)
{
    return c_sstevr(jobz, range, n, d, e, vl, vu, il, iu, abstol, &m, w, z, ldz, isuppz);
}

inline integer stevr( char jobz, char range, integer n, double* d,
                    double* e, double vl, double vu, integer il,
                    integer iu, double abstol, integer& m, double* w,
                    double* z, integer ldz, integer* isuppz)
{
    return c_dstevr(jobz, range, n, d, e, vl, vu, il, iu, abstol, &m, w, z, ldz, isuppz);
}

inline integer geev( char jobvl, char jobvr, integer n, float* a,
                   integer lda, scomplex* w, float* vl,
                   integer ldvl, float* vr, integer ldvr)
{
    return c_sgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

inline integer geev( char jobvl, char jobvr, integer n, double* a,
                   integer lda, dcomplex* w, double* vl,
                   integer ldvl, double* vr, integer ldvr)
{
    return c_dgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

inline integer geev( char jobvl, char jobvr, integer n, scomplex* a,
                   integer lda, scomplex* w, scomplex* vl,
                   integer ldvl, scomplex* vr, integer ldvr)
{
    return c_cgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

inline integer geev( char jobvl, char jobvr, integer n, dcomplex* a,
                   integer lda, dcomplex* w, dcomplex* vl,
                   integer ldvl, dcomplex* vr, integer ldvr)
{
    return c_zgeev(jobvl, jobvr, n, a, lda, w, vl, ldvl, vr, ldvr);
}

inline integer geevx( char balanc, char jobvl, char jobvr, char sense,
                    integer n, float* a, integer lda, scomplex* w,
                    float* vl, integer ldvl, float* vr,
                    integer ldvr, integer& ilo, integer& ihi,
                    float* scale, float& abnrm, float* rconde,
                    float* rcondv)
{
    return c_sgeevx(balanc, jobvl, jobvr, sense, n, a, lda, w, vl, ldvl, vr, ldvr, &ilo, &ihi, scale, &abnrm, rconde, rcondv);
}

inline integer geevx( char balanc, char jobvl, char jobvr, char sense,
                    integer n, double* a, integer lda, dcomplex* w,
                    double* vl, integer ldvl, double* vr,
                    integer ldvr, integer& ilo, integer& ihi,
                    double* scale, double& abnrm, double* rconde,
                    double* rcondv)
{
    return c_dgeevx(balanc, jobvl, jobvr, sense, n, a, lda, w, vl, ldvl, vr, ldvr, &ilo, &ihi, scale, &abnrm, rconde, rcondv);
}

inline integer geevx( char balanc, char jobvl, char jobvr, char sense,
                    integer n, scomplex* a, integer lda, scomplex* w,
                    scomplex* vl, integer ldvl, scomplex* vr,
                    integer ldvr, integer& ilo, integer& ihi,
                    float* scale, float& abnrm, float* rconde,
                    float* rcondv)
{
    return c_cgeevx(balanc, jobvl, jobvr, sense, n, a, lda, w, vl, ldvl, vr, ldvr, &ilo, &ihi, scale, &abnrm, rconde, rcondv);
}

inline integer geevx( char balanc, char jobvl, char jobvr, char sense,
                    integer n, dcomplex* a, integer lda, dcomplex* w,
                    dcomplex* vl, integer ldvl, dcomplex* vr,
                    integer ldvr, integer& ilo, integer& ihi,
                    double* scale, double& abnrm, double* rconde,
                    double* rcondv)
{
    return c_zgeevx(balanc, jobvl, jobvr, sense, n, a, lda, w, vl, ldvl, vr, ldvr, &ilo, &ihi, scale, &abnrm, rconde, rcondv);
}

inline integer gesvd( char jobu, char jobvt, integer m, integer n,
                    float* a, integer lda, float* s, float* u,
                    integer ldu, float* vt, integer ldvt)
{
    return c_sgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt);
}

inline integer gesvd( char jobu, char jobvt, integer m, integer n,
                    double* a, integer lda, double* s, double* u,
                    integer ldu, double* vt, integer ldvt)
{
    return c_dgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt);
}

inline integer gesvd( char jobu, char jobvt, integer m, integer n,
                    scomplex* a, integer lda, float* s, scomplex* u,
                    integer ldu, scomplex* vt, integer ldvt)
{
    return c_cgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt);
}

inline integer gesvd( char jobu, char jobvt, integer m, integer n,
                    dcomplex* a, integer lda, double* s, dcomplex* u,
                    integer ldu, dcomplex* vt, integer ldvt)
{
    return c_zgesvd(jobu, jobvt, m, n, a, lda, s, u, ldu, vt, ldvt);
}

inline integer gesdd( char jobz, integer m, integer n, float* a,
                    integer lda, float* s, float* u, integer ldu,
                    float* vt, integer ldvt)
{
    return c_sgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

inline integer gesdd( char jobz, integer m, integer n, double* a,
                    integer lda, double* s, double* u, integer ldu,
                    double* vt, integer ldvt)
{
    return c_dgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

inline integer gesdd( char jobz, integer m, integer n, scomplex* a,
                    integer lda, float* s, scomplex* u, integer ldu,
                    scomplex* vt, integer ldvt)
{
    return c_cgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

inline integer gesdd( char jobz, integer m, integer n, dcomplex* a,
                    integer lda, double* s, dcomplex* u, integer ldu,
                    dcomplex* vt, integer ldvt)
{
    return c_zgesdd(jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

inline integer gejsv( char joba, char jobu, char jobv, char jobr, char jobt,
                    char jobp, integer m, integer n, float* a,
                    integer lda, float* sva, float* u, integer ldu,
                    float* v, integer ldv)
{
    return c_sgejsv(joba, jobu, jobv, jobr, jobt, jobp, m, n, a, lda, sva, u, ldu, v, ldv);
}

inline integer gejsv( char joba, char jobu, char jobv, char jobr, char jobt,
                    char jobp, integer m, integer n, double* a,
                    integer lda, double* sva, double* u, integer ldu,
                    double* v, integer ldv)
{
    return c_dgejsv(joba, jobu, jobv, jobr, jobt, jobp, m, n, a, lda, sva, u, ldu, v, ldv);
}

inline integer gesvj( char joba, char jobu, char jobv, integer m,
                    integer n, float* a, integer lda, float* sva,
                    integer mv, float* v, integer ldv)
{
    return c_sgesvj(joba, jobu, jobv, m, n, a, lda, sva, mv, v, ldv);
}

inline integer gesvj( char joba, char jobu, char jobv, integer m,
                    integer n, double* a, integer lda, double* sva,
                    integer mv, double* v, integer ldv)
{
    return c_dgesvj(joba, jobu, jobv, m, n, a, lda, sva, mv, v, ldv);
}

inline integer ggsvd( char jobu, char jobv, char jobq, integer m,
                    integer n, integer p, integer& k, integer& l,
                    float* a, integer lda, float* b, integer ldb,
                    float* alpha, float* beta, float* u, integer ldu,
                    float* v, integer ldv, float* q, integer ldq)
{
    return c_sggsvd(jobu, jobv, jobq, m, n, p, &k, &l, a, lda, b, ldb, alpha, beta, u, ldu, v, ldv, q, ldq);
}

inline integer ggsvd( char jobu, char jobv, char jobq, integer m,
                    integer n, integer p, integer& k, integer& l,
                    double* a, integer lda, double* b, integer ldb,
                    double* alpha, double* beta, double* u, integer ldu,
                    double* v, integer ldv, double* q, integer ldq)
{
    return c_dggsvd(jobu, jobv, jobq, m, n, p, &k, &l, a, lda, b, ldb, alpha, beta, u, ldu, v, ldv, q, ldq);
}

inline integer ggsvd( char jobu, char jobv, char jobq, integer m,
                    integer n, integer p, integer& k, integer& l,
                    scomplex* a, integer lda, scomplex* b, integer ldb,
                    float* alpha, float* beta, scomplex* u, integer ldu,
                    scomplex* v, integer ldv, scomplex* q, integer ldq)
{
    return c_cggsvd(jobu, jobv, jobq, m, n, p, &k, &l, a, lda, b, ldb, alpha, beta, u, ldu, v, ldv, q, ldq);
}

inline integer ggsvd( char jobu, char jobv, char jobq, integer m,
                    integer n, integer p, integer& k, integer& l,
                    dcomplex* a, integer lda, dcomplex* b, integer ldb,
                    double* alpha, double* beta, dcomplex* u, integer ldu,
                    dcomplex* v, integer ldv, dcomplex* q, integer ldq)
{
    return c_zggsvd(jobu, jobv, jobq, m, n, p, &k, &l, a, lda, b, ldb, alpha, beta, u, ldu, v, ldv, q, ldq);
}

inline integer hegv( integer itype, char jobz, char uplo, integer n,
                   float* a, integer lda, float* b, integer ldb,
                   float* w)
{
    return c_ssygv(itype, jobz, uplo, n, a, lda, b, ldb, w);
}

inline integer hegv( integer itype, char jobz, char uplo, integer n,
                   double* a, integer lda, double* b, integer ldb,
                   double* w)
{
    return c_dsygv(itype, jobz, uplo, n, a, lda, b, ldb, w);
}

inline integer hegv( integer itype, char jobz, char uplo, integer n,
                   scomplex* a, integer lda, scomplex* b, integer ldb,
                   float* w)
{
    return c_chegv(itype, jobz, uplo, n, a, lda, b, ldb, w);
}

inline integer hegv( integer itype, char jobz, char uplo, integer n,
                   dcomplex* a, integer lda, dcomplex* b, integer ldb,
                   double* w)
{
    return c_zhegv(itype, jobz, uplo, n, a, lda, b, ldb, w);
}

inline integer hegvd( integer itype, char jobz, char uplo, integer n,
                    float* a, integer lda, float* b, integer ldb,
                    float* w)
{
    return c_ssygvd(itype, jobz, uplo, n, a, lda, b, ldb, w);
}

inline integer hegvd( integer itype, char jobz, char uplo, integer n,
                    double* a, integer lda, double* b, integer ldb,
                    double* w)
{
    return c_dsygvd(itype, jobz, uplo, n, a, lda, b, ldb, w);
}

inline integer hegvd( integer itype, char jobz, char uplo, integer n,
                    scomplex* a, integer lda, scomplex* b, integer ldb,
                    float* w)
{
    return c_chegvd(itype, jobz, uplo, n, a, lda, b, ldb, w);
}

inline integer hegvd( integer itype, char jobz, char uplo, integer n,
                    dcomplex* a, integer lda, dcomplex* b, integer ldb,
                    double* w)
{
    return c_zhegvd(itype, jobz, uplo, n, a, lda, b, ldb, w);
}

inline integer hegvx( integer itype, char jobz, char range, char uplo,
                    integer n, float* a, integer lda, float* b,
                    integer ldb, float vl, float vu, integer il,
                    integer iu, float abstol, integer& m, float* w,
                    float* z, integer ldz, integer* ifail )
{
    return c_ssygvx(itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hegvx( integer itype, char jobz, char range, char uplo,
                    integer n, double* a, integer lda, double* b,
                    integer ldb, double vl, double vu, integer il,
                    integer iu, double abstol, integer& m, double* w,
                    double* z, integer ldz, integer* ifail )
{
    return c_dsygvx(itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hegvx( integer itype, char jobz, char range, char uplo,
                    integer n, scomplex* a, integer lda, scomplex* b,
                    integer ldb, float vl, float vu, integer il,
                    integer iu, float abstol, integer& m, float* w,
                    scomplex* z, integer ldz, integer* ifail )
{
    return c_chegvx(itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hegvx( integer itype, char jobz, char range, char uplo,
                    integer n, dcomplex* a, integer lda, dcomplex* b,
                    integer ldb, double vl, double vu, integer il,
                    integer iu, double abstol, integer& m, double* w,
                    dcomplex* z, integer ldz, integer* ifail )
{
    return c_zhegvx(itype, jobz, range, uplo, n, a, lda, b, ldb, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hpgv( integer itype, char jobz, char uplo, integer n,
                    float* ap, float* bp, float* w, float* z,
                   integer ldz)
{
    return c_sspgv(itype, jobz, uplo, n, ap, bp, w, z, ldz);
}

inline integer hpgv( integer itype, char jobz, char uplo, integer n,
                   double* ap, double* bp, double* w, double* z,
                   integer ldz)
{
    return c_dspgv(itype, jobz, uplo, n, ap, bp, w, z, ldz);
}

inline integer hpgv( integer itype, char jobz, char uplo, integer n,
                   scomplex* ap, scomplex* bp, float* w, scomplex* z,
                   integer ldz)
{
    return c_chpgv(itype, jobz, uplo, n, ap, bp, w, z, ldz);
}

inline integer hpgv( integer itype, char jobz, char uplo, integer n,
                   dcomplex* ap, dcomplex* bp, double* w, dcomplex* z,
                   integer ldz)
{
    return c_zhpgv(itype, jobz, uplo, n, ap, bp, w, z, ldz);
}

inline integer hpgvd( integer itype, char jobz, char uplo, integer n,
                    float* ap, float* bp, float* w, float* z,
                    integer ldz)
{
    return c_sspgvd(itype, jobz, uplo, n, ap, bp, w, z, ldz);
}

inline integer hpgvd( integer itype, char jobz, char uplo, integer n,
                    double* ap, double* bp, double* w, double* z,
                    integer ldz)
{
    return c_dspgvd(itype, jobz, uplo, n, ap, bp, w, z, ldz);
}

inline integer hpgvd( integer itype, char jobz, char uplo, integer n,
                    scomplex* ap, scomplex* bp, float* w, scomplex* z,
                    integer ldz)
{
    return c_chpgvd(itype, jobz, uplo, n, ap, bp, w, z, ldz);
}

inline integer hpgvd( integer itype, char jobz, char uplo, integer n,
                    dcomplex* ap, dcomplex* bp, double* w, dcomplex* z,
                    integer ldz)
{
    return c_zhpgvd(itype, jobz, uplo, n, ap, bp, w, z, ldz);
}

inline integer hpgvx( integer itype, char jobz, char range, char uplo,
                    integer n, float* ap, float* bp, float vl,
                    float vu, integer il, integer iu, float abstol,
                    integer& m, float* w, float* z, integer ldz,
                    integer* ifail )
{
    return c_sspgvx(itype, jobz, range, uplo, n, ap, bp, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hpgvx( integer itype, char jobz, char range, char uplo,
                    integer n, double* ap, double* bp, double vl,
                    double vu, integer il, integer iu, double abstol,
                    integer& m, double* w, double* z, integer ldz,
                    integer* ifail )
{
    return c_dspgvx(itype, jobz, range, uplo, n, ap, bp, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hpgvx( integer itype, char jobz, char range, char uplo,
                    integer n, scomplex* ap, scomplex* bp, float vl,
                    float vu, integer il, integer iu, float abstol,
                    integer& m, float* w, scomplex* z, integer ldz,
                    integer* ifail )
{
    return c_chpgvx(itype, jobz, range, uplo, n, ap, bp, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hpgvx( integer itype, char jobz, char range, char uplo,
                    integer n, dcomplex* ap, dcomplex* bp, double vl,
                    double vu, integer il, integer iu, double abstol,
                    integer& m, double* w, dcomplex* z, integer ldz,
                    integer* ifail )
{
    return c_zhpgvx(itype, jobz, range, uplo, n, ap, bp, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hbgv( char jobz, char uplo, integer n, integer ka,
                   integer kb, float* ab, integer ldab, float* bb,
                   integer ldbb, float* w, float* z, integer ldz)
{
    return c_ssbgv(jobz, uplo, n, ka, kb, ab, ldab, bb, ldbb, w, z, ldz);
}

inline integer hbgv( char jobz, char uplo, integer n, integer ka,
                   integer kb, double* ab, integer ldab, double* bb,
                   integer ldbb, double* w, double* z, integer ldz)
{
    return c_dsbgv(jobz, uplo, n, ka, kb, ab, ldab, bb, ldbb, w, z, ldz);
}

inline integer hbgv( char jobz, char uplo, integer n, integer ka,
                   integer kb, scomplex* ab, integer ldab, scomplex* bb,
                   integer ldbb, float* w, scomplex* z, integer ldz)
{
    return c_chbgv(jobz, uplo, n, ka, kb, ab, ldab, bb, ldbb, w, z, ldz);
}

inline integer hbgv( char jobz, char uplo, integer n, integer ka,
                   integer kb, dcomplex* ab, integer ldab, dcomplex* bb,
                   integer ldbb, double* w, dcomplex* z, integer ldz)
{
    return c_zhbgv(jobz, uplo, n, ka, kb, ab, ldab, bb, ldbb, w, z, ldz);
}

inline integer hbgvd( char jobz, char uplo, integer n, integer ka,
                    integer kb, float* ab, integer ldab, float* bb,
                    integer ldbb, float* w, float* z, integer ldz)
{
    return c_ssbgvd(jobz, uplo, n, ka, kb, ab, ldab, bb, ldbb, w, z, ldz);
}

inline integer hbgvd( char jobz, char uplo, integer n, integer ka,
                    integer kb, double* ab, integer ldab, double* bb,
                    integer ldbb, double* w, double* z, integer ldz)
{
    return c_dsbgvd(jobz, uplo, n, ka, kb, ab, ldab, bb, ldbb, w, z, ldz);
}

inline integer hbgvd( char jobz, char uplo, integer n, integer ka,
                    integer kb, scomplex* ab, integer ldab, scomplex* bb,
                    integer ldbb, float* w, scomplex* z, integer ldz)
{
    return c_chbgvd(jobz, uplo, n, ka, kb, ab, ldab, bb, ldbb, w, z, ldz);
}

inline integer hbgvd( char jobz, char uplo, integer n, integer ka,
                    integer kb, dcomplex* ab, integer ldab, dcomplex* bb,
                    integer ldbb, double* w, dcomplex* z, integer ldz)
{
    return c_zhbgvd(jobz, uplo, n, ka, kb, ab, ldab, bb, ldbb, w, z, ldz);
}

inline integer hbgvx( char jobz, char range, char uplo, integer n,
                    integer ka, integer kb, float* ab,
                    integer ldab, float* bb, integer ldbb, float* q,
                    integer ldq, float vl, float vu, integer il,
                    integer iu, float abstol, integer& m, float* w,
                    float* z, integer ldz, integer* ifail )
{
    return c_ssbgvx(jobz, range, uplo, n, ka, kb, ab, ldab, bb, ldbb, q, ldq, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hbgvx( char jobz, char range, char uplo, integer n,
                    integer ka, integer kb, double* ab,
                    integer ldab, double* bb, integer ldbb, double* q,
                    integer ldq, double vl, double vu, integer il,
                    integer iu, double abstol, integer& m, double* w,
                    double* z, integer ldz, integer* ifail )
{
    return c_dsbgvx(jobz, range, uplo, n, ka, kb, ab, ldab, bb, ldbb, q, ldq, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hbgvx( char jobz, char range, char uplo, integer n,
                    integer ka, integer kb, scomplex* ab,
                    integer ldab, scomplex* bb, integer ldbb, scomplex* q,
                    integer ldq, float vl, float vu, integer il,
                    integer iu, float abstol, integer& m, float* w,
                    scomplex* z, integer ldz, integer* ifail )
{
    return c_chbgvx(jobz, range, uplo, n, ka, kb, ab, ldab, bb, ldbb, q, ldq, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer hbgvx( char jobz, char range, char uplo, integer n,
                    integer ka, integer kb, dcomplex* ab,
                    integer ldab, dcomplex* bb, integer ldbb, dcomplex* q,
                    integer ldq, double vl, double vu, integer il,
                    integer iu, double abstol, integer& m, double* w,
                    dcomplex* z, integer ldz, integer* ifail )
{
    return c_zhbgvx(jobz, range, uplo, n, ka, kb, ab, ldab, bb, ldbb, q, ldq, vl, vu, il, iu, abstol, &m, w, z, ldz, ifail);
}

inline integer ggev( char jobvl, char jobvr, integer n, float* a,
                   integer lda, float* b, integer ldb, scomplex* alpha,
                   float* beta, float* vl, integer ldvl,
                   float* vr, integer ldvr)
{
    return c_sggev(jobvl, jobvr, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr);
}

inline integer ggev( char jobvl, char jobvr, integer n, double* a,
                   integer lda, double* b, integer ldb, dcomplex* alpha,
                   double* beta, double* vl, integer ldvl,
                   double* vr, integer ldvr)
{
    return c_dggev(jobvl, jobvr, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr);
}

inline integer ggev( char jobvl, char jobvr, integer n, scomplex* a,
                   integer lda, scomplex* b, integer ldb, scomplex* alpha,
                   scomplex* beta, scomplex* vl, integer ldvl,
                   scomplex* vr, integer ldvr)
{
    return c_cggev(jobvl, jobvr, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr);
}

inline integer ggev( char jobvl, char jobvr, integer n, dcomplex* a,
                   integer lda, dcomplex* b, integer ldb, dcomplex* alpha,
                   dcomplex* beta, dcomplex* vl, integer ldvl,
                   dcomplex* vr, integer ldvr)
{
    return c_zggev(jobvl, jobvr, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr);
}

inline integer ggevx( char balanc, char jobvl, char jobvr, char sense,
                    integer n, float* a, integer lda, float* b,
                    integer ldb, scomplex* alpha,
                    float* beta, float* vl, integer ldvl, float* vr,
                    integer ldvr, integer& ilo, integer& ihi,
                    float* lscale, float* rscale, float& abnrm,
                    float& bbnrm, float* rconde, float* rcondv)
{
    return c_sggevx(balanc, jobvl, jobvr, sense, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr,
                  &ilo, &ihi, lscale, rscale, &abnrm, &bbnrm, rconde, rcondv);
}

inline integer ggevx( char balanc, char jobvl, char jobvr, char sense,
                    integer n, double* a, integer lda, double* b,
                    integer ldb, dcomplex* alpha,
                    double* beta, double* vl, integer ldvl, double* vr,
                    integer ldvr, integer& ilo, integer& ihi,
                    double* lscale, double* rscale, double& abnrm,
                    double& bbnrm, double* rconde, double* rcondv)
{
    return c_dggevx(balanc, jobvl, jobvr, sense, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr,
                  &ilo, &ihi, lscale, rscale, &abnrm, &bbnrm, rconde, rcondv);
}

inline integer ggevx( char balanc, char jobvl, char jobvr, char sense,
                    integer n, scomplex* a, integer lda, scomplex* b,
                    integer ldb, scomplex* alpha,
                    scomplex* beta, scomplex* vl, integer ldvl, scomplex* vr,
                    integer ldvr, integer& ilo, integer& ihi,
                    float* lscale, float* rscale, float& abnrm,
                    float& bbnrm, float* rconde, float* rcondv)
{
    return c_cggevx(balanc, jobvl, jobvr, sense, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr,
                  &ilo, &ihi, lscale, rscale, &abnrm, &bbnrm, rconde, rcondv);
}

inline integer ggevx( char balanc, char jobvl, char jobvr, char sense,
                    integer n, dcomplex* a, integer lda, dcomplex* b,
                    integer ldb, dcomplex* alpha,
                    dcomplex* beta, dcomplex* vl, integer ldvl, dcomplex* vr,
                    integer ldvr, integer& ilo, integer& ihi,
                    double* lscale, double* rscale, double& abnrm,
                    double& bbnrm, double* rconde, double* rcondv)
{
    return c_zggevx(balanc, jobvl, jobvr, sense, n, a, lda, b, ldb, alpha, beta, vl, ldvl, vr, ldvr,
                  &ilo, &ihi, lscale, rscale, &abnrm, &bbnrm, rconde, rcondv);
}

inline void hfrk( char transr, char uplo, char trans, integer n,
                   integer k, float alpha, const float* a,
                   integer lda, float beta, float* c )
{
    c_ssfrk(transr, uplo, trans, n, k, alpha, a, lda, beta, c);
}

inline void hfrk( char transr, char uplo, char trans, integer n,
                   integer k, double alpha, const double* a,
                   integer lda, double beta, double* c )
{
    c_dsfrk(transr, uplo, trans, n, k, alpha, a, lda, beta, c);
}

inline void hfrk( char transr, char uplo, char trans, integer n,
                   integer k, float alpha, const scomplex* a,
                   integer lda, float beta, scomplex* c )
{
    c_chfrk(transr, uplo, trans, n, k, alpha, a, lda, beta, c);
}

inline void hfrk( char transr, char uplo, char trans, integer n,
                   integer k, double alpha, const dcomplex* a,
                   integer lda, double beta, dcomplex* c )
{
    c_zhfrk(transr, uplo, trans, n, k, alpha, a, lda, beta, c);
}

inline void tfsm( char transr, char side, char uplo, char trans,
                   char diag, integer m, integer n, float alpha,
                   const float* a, float* b, integer ldb )
{
    c_stfsm(transr, side, uplo, trans, diag, m, n, alpha, a, b, ldb);
}

inline void tfsm( char transr, char side, char uplo, char trans,
                   char diag, integer m, integer n, double alpha,
                   const double* a, double* b, integer ldb )
{
    c_dtfsm(transr, side, uplo, trans, diag, m, n, alpha, a, b, ldb);
}

inline void tfsm( char transr, char side, char uplo, char trans,
                   char diag, integer m, integer n, scomplex alpha,
                   const scomplex* a, scomplex* b, integer ldb )
{
    c_ctfsm(transr, side, uplo, trans, diag, m, n, alpha, a, b, ldb);
}

inline void tfsm( char transr, char side, char uplo, char trans,
                   char diag, integer m, integer n, dcomplex alpha,
                   const dcomplex* a, dcomplex* b, integer ldb )
{
    c_ztfsm(transr, side, uplo, trans, diag, m, n, alpha, a, b, ldb);
}

inline integer tfttp( char transr, char uplo, integer n, const float* arf,
                    float* ap )
{
    return c_stfttp(transr, uplo, n, arf, ap);
}

inline integer tfttp( char transr, char uplo, integer n, const double* arf,
                    double* ap )
{
    return c_dtfttp(transr, uplo, n, arf, ap);
}

inline integer tfttp( char transr, char uplo, integer n, const scomplex* arf,
                    scomplex* ap )
{
    return c_ctfttp(transr, uplo, n, arf, ap);
}

inline integer tfttp( char transr, char uplo, integer n, const dcomplex* arf,
                    dcomplex* ap )
{
    return c_ztfttp(transr, uplo, n, arf, ap);
}

inline integer tfttr( char transr, char uplo, integer n, const float* arf,
                    float* a, integer lda )
{
    return c_stfttr(transr, uplo, n, arf, a, lda);
}

inline integer tfttr( char transr, char uplo, integer n, const double* arf,
                    double* a, integer lda )
{
    return c_dtfttr(transr, uplo, n, arf, a, lda);
}

inline integer tfttr( char transr, char uplo, integer n, const scomplex* arf,
                    scomplex* a, integer lda )
{
    return c_ctfttr(transr, uplo, n, arf, a, lda);
}

inline integer tfttr( char transr, char uplo, integer n, const dcomplex* arf,
                    dcomplex* a, integer lda )
{
    return c_ztfttr(transr, uplo, n, arf, a, lda);
}

inline integer tpttf( char transr, char uplo, integer n, const float* ap,
                    float* arf )
{
    return c_stpttf(transr, uplo, n, ap, arf);
}

inline integer tpttf( char transr, char uplo, integer n, const double* ap,
                    double* arf )
{
    return c_dtpttf(transr, uplo, n, ap, arf);
}

inline integer tpttf( char transr, char uplo, integer n, const scomplex* ap,
                    scomplex* arf )
{
    return c_ctpttf(transr, uplo, n, ap, arf);
}

inline integer tpttf( char transr, char uplo, integer n, const dcomplex* ap,
                    dcomplex* arf )
{
    return c_ztpttf(transr, uplo, n, ap, arf);
}

inline integer tpttr( char uplo, integer n, const float* ap, float* a,
                    integer lda )
{
    return c_stpttr(uplo, n, ap, a, lda);
}

inline integer tpttr( char uplo, integer n, const double* ap, double* a,
                    integer lda )
{
    return c_dtpttr(uplo, n, ap, a, lda);
}

inline integer tpttr( char uplo, integer n, const scomplex* ap, scomplex* a,
                    integer lda )
{
    return c_ctpttr(uplo, n, ap, a, lda);
}

inline integer tpttr( char uplo, integer n, const dcomplex* ap, dcomplex* a,
                    integer lda )
{
    return c_ztpttr(uplo, n, ap, a, lda);
}

inline integer trttf( char transr, char uplo, integer n, const float* a,
                    integer lda, float* arf )
{
    return c_strttf(transr, uplo, n, a, lda, arf);
}

inline integer trttf( char transr, char uplo, integer n, const double* a,
                    integer lda, double* arf )
{
    return c_dtrttf(transr, uplo, n, a, lda, arf);
}

inline integer trttf( char transr, char uplo, integer n, const scomplex* a,
                    integer lda, scomplex* arf )
{
    return c_ctrttf(transr, uplo, n, a, lda, arf);
}

inline integer trttf( char transr, char uplo, integer n, const dcomplex* a,
                    integer lda, dcomplex* arf )
{
    return c_ztrttf(transr, uplo, n, a, lda, arf);
}

inline integer trttp( char uplo, integer n, const float* a, integer lda,
                    float* ap )
{
    return c_strttp(uplo, n, a, lda, ap);
}

inline integer trttp( char uplo, integer n, const double* a, integer lda,
                    double* ap )
{
    return c_dtrttp(uplo, n, a, lda, ap);
}

inline integer trttp( char uplo, integer n, const scomplex* a, integer lda,
                    scomplex* ap )
{
    return c_ctrttp(uplo, n, a, lda, ap);
}

inline integer trttp( char uplo, integer n, const dcomplex* a, integer lda,
                    dcomplex* ap )
{
    return c_ztrttp(uplo, n, a, lda, ap);
}

inline integer geqrfp( integer m, integer n, float* a, integer lda,
                     float* tau)
{
    return c_sgeqrfp(m, n, a, lda, tau);
}

inline integer geqrfp( integer m, integer n, double* a, integer lda,
                     double* tau)
{
    return c_dgeqrfp(m, n, a, lda, tau);
}

inline integer geqrfp( integer m, integer n, scomplex* a, integer lda,
                     scomplex* tau)
{
    return c_cgeqrfp(m, n, a, lda, tau);
}

inline integer geqrfp( integer m, integer n, dcomplex* a, integer lda,
                     dcomplex* tau)
{
    return c_zgeqrfp(m, n, a, lda, tau);
}

}

#endif
