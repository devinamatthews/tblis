#ifndef _LAWRAP_INTERNAL_LAPACK_F77_H_
#define _LAWRAP_INTERNAL_LAPACK_F77_H_

#ifndef _LAWRAP_LAPACK_H_
#error "This file must only be included through lapack.h."
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/*
 * FORTRAN prototypes
 */
void FC_FUNC(sgetrf,SGETRF)( const integer* m, const integer* n, float* a, const integer* lda,
                    integer* ipiv, integer *info );

void FC_FUNC(dgetrf,DGETRF)( const integer* m, const integer* n, double* a, const integer* lda,
                    integer* ipiv, integer *info );

void FC_FUNC(cgetrf,CGETRF)( const integer* m, const integer* n, scomplex* a, const integer* lda,
                    integer* ipiv, integer *info );

void FC_FUNC(zgetrf,ZGETRF)( const integer* m, const integer* n, dcomplex* a, const integer* lda,
                    integer* ipiv, integer *info );

void FC_FUNC(sgbtrf,SGBTRF)( const integer* m, const integer* n, const integer* kl,
                    const integer* ku, float* ab, const integer* ldab,
                    integer* ipiv, integer *info );

void FC_FUNC(dgbtrf,DGBTRF)( const integer* m, const integer* n, const integer* kl,
                    const integer* ku, double* ab, const integer* ldab,
                    integer* ipiv, integer *info );

void FC_FUNC(cgbtrf,CGBTRF)( const integer* m, const integer* n, const integer* kl,
                    const integer* ku, scomplex* ab, const integer* ldab,
                    integer* ipiv, integer *info );

void FC_FUNC(zgbtrf,ZGBTRF)( const integer* m, const integer* n, const integer* kl,
                    const integer* ku, dcomplex* ab, const integer* ldab,
                    integer* ipiv, integer *info );

void FC_FUNC(sgttrf,SGTTRF)( const integer* n, float* dl, float* d, float* du,
                    float* du2, integer* ipiv, integer *info );

void FC_FUNC(dgttrf,DGTTRF)( const integer* n, double* dl, double* d, double* du,
                    double* du2, integer* ipiv, integer *info );

void FC_FUNC(cgttrf,CGTTRF)( const integer* n, scomplex* dl, scomplex* d, scomplex* du,
                    scomplex* du2, integer* ipiv, integer *info );

void FC_FUNC(zgttrf,ZGTTRF)( const integer* n, dcomplex* dl, dcomplex* d, dcomplex* du,
                    dcomplex* du2, integer* ipiv, integer *info );

void FC_FUNC(spotrf,SPOTRF)( const char* uplo, const integer* n, float* a, const integer* lda,
                    integer *info );

void FC_FUNC(dpotrf,DPOTRF)( const char* uplo, const integer* n, double* a, const integer* lda,
                    integer *info );

void FC_FUNC(cpotrf,CPOTRF)( const char* uplo, const integer* n, scomplex* a, const integer* lda,
                    integer *info );

void FC_FUNC(zpotrf,ZPOTRF)( const char* uplo, const integer* n, dcomplex* a, const integer* lda,
                    integer *info );

void FC_FUNC(spstrf,SPSTRF)( const char* uplo, const integer* n, float* a, const integer* lda,
                    integer* piv, integer* rank, float* tol,
                    float* work, integer *info );

void FC_FUNC(dpstrf,DPSTRF)( const char* uplo, const integer* n, double* a, const integer* lda,
                    integer* piv, integer* rank, double* tol,
                    double* work, integer *info );

void FC_FUNC(cpstrf,CPSTRF)( const char* uplo, const integer* n, scomplex* a, const integer* lda,
                    integer* piv, integer* rank, float* tol,
                    float* work, integer *info );

void FC_FUNC(zpstrf,ZPSTRF)( const char* uplo, const integer* n, dcomplex* a, const integer* lda,
                    integer* piv, integer* rank, double* tol,
                    double* work, integer *info );

void FC_FUNC(spftrf,SPFTRF)( const char* transr, const char* uplo, const integer* n, float* a,
                    integer *info );

void FC_FUNC(dpftrf,DPFTRF)( const char* transr, const char* uplo, const integer* n, double* a,
                    integer *info );

void FC_FUNC(cpftrf,CPFTRF)( const char* transr, const char* uplo, const integer* n, scomplex* a,
                    integer *info );

void FC_FUNC(zpftrf,ZPFTRF)( const char* transr, const char* uplo, const integer* n, dcomplex* a,
                    integer *info );

void FC_FUNC(spptrf,SPPTRF)( const char* uplo, const integer* n, float* ap, integer *info );

void FC_FUNC(dpptrf,DPPTRF)( const char* uplo, const integer* n, double* ap, integer *info );

void FC_FUNC(cpptrf,CPPTRF)( const char* uplo, const integer* n, scomplex* ap, integer *info );

void FC_FUNC(zpptrf,ZPPTRF)( const char* uplo, const integer* n, dcomplex* ap, integer *info );

void FC_FUNC(spbtrf,SPBTRF)( const char* uplo, const integer* n, const integer* kd, float* ab,
                    const integer* ldab, integer *info );

void FC_FUNC(dpbtrf,DPBTRF)( const char* uplo, const integer* n, const integer* kd, double* ab,
                    const integer* ldab, integer *info );

void FC_FUNC(cpbtrf,CPBTRF)( const char* uplo, const integer* n, const integer* kd, scomplex* ab,
                    const integer* ldab, integer *info );

void FC_FUNC(zpbtrf,ZPBTRF)( const char* uplo, const integer* n, const integer* kd, dcomplex* ab,
                    const integer* ldab, integer *info );

void FC_FUNC(spttrf,SPTTRF)( const integer* n, float* d, float* e, integer *info );

void FC_FUNC(dpttrf,DPTTRF)( const integer* n, double* d, double* e, integer *info );

void FC_FUNC(cpttrf,CPTTRF)( const integer* n, float* d, scomplex* e, integer *info );

void FC_FUNC(zpttrf,ZPTTRF)( const integer* n, double* d, dcomplex* e, integer *info );

void FC_FUNC(ssytrf,SSYTRF)( const char* uplo, const integer* n, float* a, const integer* lda,
                    integer* ipiv, float* work, integer* lwork,
                    integer *info );

void FC_FUNC(dsytrf,DSYTRF)( const char* uplo, const integer* n, double* a, const integer* lda,
                    integer* ipiv, double* work, integer* lwork,
                    integer *info );

void FC_FUNC(csytrf,CSYTRF)( const char* uplo, const integer* n, scomplex* a, const integer* lda,
                    integer* ipiv, scomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(zsytrf,ZSYTRF)( const char* uplo, const integer* n, dcomplex* a, const integer* lda,
                    integer* ipiv, dcomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(chetrf,CHETRF)( const char* uplo, const integer* n, scomplex* a, const integer* lda,
                    integer* ipiv, scomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(zhetrf,ZHETRF)( const char* uplo, const integer* n, dcomplex* a, const integer* lda,
                    integer* ipiv, dcomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(ssptrf,SSPTRF)( const char* uplo, const integer* n, float* ap, integer* ipiv,
                    integer *info );

void FC_FUNC(dsptrf,DSPTRF)( const char* uplo, const integer* n, double* ap, integer* ipiv,
                    integer *info );

void FC_FUNC(csptrf,CSPTRF)( const char* uplo, const integer* n, scomplex* ap, integer* ipiv,
                    integer *info );

void FC_FUNC(zsptrf,ZSPTRF)( const char* uplo, const integer* n, dcomplex* ap, integer* ipiv,
                    integer *info );

void FC_FUNC(chptrf,CHPTRF)( const char* uplo, const integer* n, scomplex* ap, integer* ipiv,
                    integer *info );

void FC_FUNC(zhptrf,ZHPTRF)( const char* uplo, const integer* n, dcomplex* ap, integer* ipiv,
                    integer *info );

void FC_FUNC(sgetrs,SGETRS)( const char* trans, const integer* n, const integer* nrhs,
                    const float* a, const integer* lda, const integer* ipiv,
                    float* b, const integer* ldb, integer *info );

void FC_FUNC(dgetrs,DGETRS)( const char* trans, const integer* n, const integer* nrhs,
                    const double* a, const integer* lda, const integer* ipiv,
                    double* b, const integer* ldb, integer *info );

void FC_FUNC(cgetrs,CGETRS)( const char* trans, const integer* n, const integer* nrhs,
                    const scomplex* a, const integer* lda, const integer* ipiv,
                    scomplex* b, const integer* ldb, integer *info );

void FC_FUNC(zgetrs,ZGETRS)( const char* trans, const integer* n, const integer* nrhs,
                    const dcomplex* a, const integer* lda, const integer* ipiv,
                    dcomplex* b, const integer* ldb, integer *info );

void FC_FUNC(sgbtrs,SGBTRS)( const char* trans, const integer* n, const integer* kl, const integer* ku,
                    const integer* nrhs, const float* ab, const integer* ldab,
                    const integer* ipiv, float* b, const integer* ldb,
                    integer *info );

void FC_FUNC(dgbtrs,DGBTRS)( const char* trans, const integer* n, const integer* kl, const integer* ku,
                    const integer* nrhs, const double* ab, const integer* ldab,
                    const integer* ipiv, double* b, const integer* ldb,
                    integer *info );

void FC_FUNC(cgbtrs,CGBTRS)( const char* trans, const integer* n, const integer* kl, const integer* ku,
                    const integer* nrhs, const scomplex* ab, const integer* ldab,
                    const integer* ipiv, scomplex* b, const integer* ldb,
                    integer *info );

void FC_FUNC(zgbtrs,ZGBTRS)( const char* trans, const integer* n, const integer* kl, const integer* ku,
                    const integer* nrhs, const dcomplex* ab, const integer* ldab,
                    const integer* ipiv, dcomplex* b, const integer* ldb,
                    integer *info );

void FC_FUNC(sgttrs,SGTTRS)( const char* trans, const integer* n, const integer* nrhs,
                    const float* dl, const float* d, const float* du,
                    const float* du2, const integer* ipiv, float* b,
                    const integer* ldb, integer *info );

void FC_FUNC(dgttrs,DGTTRS)( const char* trans, const integer* n, const integer* nrhs,
                    const double* dl, const double* d, const double* du,
                    const double* du2, const integer* ipiv, double* b,
                    const integer* ldb, integer *info );

void FC_FUNC(cgttrs,CGTTRS)( const char* trans, const integer* n, const integer* nrhs,
                    const scomplex* dl, const scomplex* d, const scomplex* du,
                    const scomplex* du2, const integer* ipiv, scomplex* b,
                    const integer* ldb, integer *info );

void FC_FUNC(zgttrs,ZGTTRS)( const char* trans, const integer* n, const integer* nrhs,
                    const dcomplex* dl, const dcomplex* d, const dcomplex* du,
                    const dcomplex* du2, const integer* ipiv, dcomplex* b,
                    const integer* ldb, integer *info );

void FC_FUNC(spotrs,SPOTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const float* a, const integer* lda, float* b,
                    const integer* ldb, integer *info );

void FC_FUNC(dpotrs,DPOTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const double* a, const integer* lda, double* b,
                    const integer* ldb, integer *info );

void FC_FUNC(cpotrs,CPOTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* a, const integer* lda, scomplex* b,
                    const integer* ldb, integer *info );

void FC_FUNC(zpotrs,ZPOTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* a, const integer* lda, dcomplex* b,
                    const integer* ldb, integer *info );

void FC_FUNC(spftrs,SPFTRS)( const char* transr, const char* uplo, const integer* n, const integer* nrhs,
                    const float* a, float* b, const integer* ldb,
                    integer *info );

void FC_FUNC(dpftrs,DPFTRS)( const char* transr, const char* uplo, const integer* n, const integer* nrhs,
                    const double* a, double* b, const integer* ldb,
                    integer *info );

void FC_FUNC(cpftrs,CPFTRS)( const char* transr, const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* a, scomplex* b, const integer* ldb,
                    integer *info );

void FC_FUNC(zpftrs,ZPFTRS)( const char* transr, const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* a, dcomplex* b, const integer* ldb,
                    integer *info );

void FC_FUNC(spptrs,SPPTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const float* ap, float* b, const integer* ldb,
                    integer *info );

void FC_FUNC(dpptrs,DPPTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const double* ap, double* b, const integer* ldb,
                    integer *info );

void FC_FUNC(cpptrs,CPPTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* ap, scomplex* b, const integer* ldb,
                    integer *info );

void FC_FUNC(zpptrs,ZPPTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* ap, dcomplex* b, const integer* ldb,
                    integer *info );

void FC_FUNC(spbtrs,SPBTRS)( const char* uplo, const integer* n, const integer* kd, const integer* nrhs,
                    const float* ab, const integer* ldab, float* b,
                    const integer* ldb, integer *info );

void FC_FUNC(dpbtrs,DPBTRS)( const char* uplo, const integer* n, const integer* kd, const integer* nrhs,
                    const double* ab, const integer* ldab, double* b,
                    const integer* ldb, integer *info );

void FC_FUNC(cpbtrs,CPBTRS)( const char* uplo, const integer* n, const integer* kd, const integer* nrhs,
                    const scomplex* ab, const integer* ldab, scomplex* b,
                    const integer* ldb, integer *info );

void FC_FUNC(zpbtrs,ZPBTRS)( const char* uplo, const integer* n, const integer* kd, const integer* nrhs,
                    const dcomplex* ab, const integer* ldab, dcomplex* b,
                    const integer* ldb, integer *info );

void FC_FUNC(spttrs,SPTTRS)( const integer* n, const integer* nrhs, const float* d,
                    const float* e, float* b, const integer* ldb,
                    integer *info );

void FC_FUNC(dpttrs,DPTTRS)( const integer* n, const integer* nrhs, const double* d,
                    const double* e, double* b, const integer* ldb,
                    integer *info );

void FC_FUNC(cpttrs,CPTTRS)( const integer* n, const integer* nrhs, const float* d,
                    const scomplex* e, scomplex* b, const integer* ldb,
                    integer *info );

void FC_FUNC(zpttrs,ZPTTRS)( const integer* n, const integer* nrhs, const double* d,
                    const dcomplex* e, dcomplex* b, const integer* ldb,
                    integer *info );

void FC_FUNC(ssytrs,SSYTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const float* a, const integer* lda, const integer* ipiv,
                    float* b, const integer* ldb, integer *info );

void FC_FUNC(dsytrs,DSYTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const double* a, const integer* lda, const integer* ipiv,
                    double* b, const integer* ldb, integer *info );

void FC_FUNC(csytrs,CSYTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* a, const integer* lda, const integer* ipiv,
                    scomplex* b, const integer* ldb, integer *info );

void FC_FUNC(zsytrs,ZSYTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* a, const integer* lda, const integer* ipiv,
                    dcomplex* b, const integer* ldb, integer *info );

void FC_FUNC(chetrs,CHETRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* a, const integer* lda, const integer* ipiv,
                    scomplex* b, const integer* ldb, integer *info );

void FC_FUNC(zhetrs,ZHETRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* a, const integer* lda, const integer* ipiv,
                    dcomplex* b, const integer* ldb, integer *info );

void FC_FUNC(ssptrs,SSPTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const float* ap, const integer* ipiv, float* b,
                    const integer* ldb, integer *info );

void FC_FUNC(dsptrs,DSPTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const double* ap, const integer* ipiv, double* b,
                    const integer* ldb, integer *info );

void FC_FUNC(csptrs,CSPTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* ap, const integer* ipiv, scomplex* b,
                    const integer* ldb, integer *info );

void FC_FUNC(zsptrs,ZSPTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* ap, const integer* ipiv, dcomplex* b,
                    const integer* ldb, integer *info );

void FC_FUNC(chptrs,CHPTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* ap, const integer* ipiv, scomplex* b,
                    const integer* ldb, integer *info );

void FC_FUNC(zhptrs,ZHPTRS)( const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* ap, const integer* ipiv, dcomplex* b,
                    const integer* ldb, integer *info );

void FC_FUNC(strtrs,STRTRS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* nrhs, const float* a, const integer* lda,
                    float* b, const integer* ldb, integer *info );

void FC_FUNC(dtrtrs,DTRTRS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* nrhs, const double* a, const integer* lda,
                    double* b, const integer* ldb, integer *info );

void FC_FUNC(ctrtrs,CTRTRS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* nrhs, const scomplex* a, const integer* lda,
                    scomplex* b, const integer* ldb, integer *info );

void FC_FUNC(ztrtrs,ZTRTRS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* nrhs, const dcomplex* a, const integer* lda,
                    dcomplex* b, const integer* ldb, integer *info );

void FC_FUNC(stptrs,STPTRS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* nrhs, const float* ap, float* b,
                    const integer* ldb, integer *info );

void FC_FUNC(dtptrs,DTPTRS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* nrhs, const double* ap, double* b,
                    const integer* ldb, integer *info );

void FC_FUNC(ctptrs,CTPTRS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* nrhs, const scomplex* ap, scomplex* b,
                    const integer* ldb, integer *info );

void FC_FUNC(ztptrs,ZTPTRS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* nrhs, const dcomplex* ap, dcomplex* b,
                    const integer* ldb, integer *info );

void FC_FUNC(stbtrs,STBTRS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* kd, const integer* nrhs, const float* ab,
                    const integer* ldab, float* b, const integer* ldb,
                    integer *info );

void FC_FUNC(dtbtrs,DTBTRS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* kd, const integer* nrhs, const double* ab,
                    const integer* ldab, double* b, const integer* ldb,
                    integer *info );

void FC_FUNC(ctbtrs,CTBTRS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* kd, const integer* nrhs, const scomplex* ab,
                    const integer* ldab, scomplex* b, const integer* ldb,
                    integer *info );

void FC_FUNC(ztbtrs,ZTBTRS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* kd, const integer* nrhs, const dcomplex* ab,
                    const integer* ldab, dcomplex* b, const integer* ldb,
                    integer *info );

void FC_FUNC(sgecon,SGECON)( char* norm, const integer* n, const float* a, const integer* lda,
                    float* anorm, float* rcond, float* work,
                    integer* iwork, integer *info );

void FC_FUNC(dgecon,DGECON)( char* norm, const integer* n, const double* a, const integer* lda,
                    double* anorm, double* rcond, double* work,
                    integer* iwork, integer *info );

void FC_FUNC(cgecon,CGECON)( char* norm, const integer* n, const scomplex* a, const integer* lda,
                    float* anorm, float* rcond, scomplex* work,
                    float* rwork, integer *info );

void FC_FUNC(zgecon,ZGECON)( char* norm, const integer* n, const dcomplex* a, const integer* lda,
                    double* anorm, double* rcond, dcomplex* work,
                    double* rwork, integer *info );

void FC_FUNC(sgbcon,SGBCON)( char* norm, const integer* n, const integer* kl, const integer* ku,
                    const float* ab, const integer* ldab, const integer* ipiv,
                    float* anorm, float* rcond, float* work,
                    integer* iwork, integer *info );

void FC_FUNC(dgbcon,DGBCON)( char* norm, const integer* n, const integer* kl, const integer* ku,
                    const double* ab, const integer* ldab, const integer* ipiv,
                    double* anorm, double* rcond, double* work,
                    integer* iwork, integer *info );

void FC_FUNC(cgbcon,CGBCON)( char* norm, const integer* n, const integer* kl, const integer* ku,
                    const scomplex* ab, const integer* ldab, const integer* ipiv,
                    float* anorm, float* rcond, scomplex* work,
                    float* rwork, integer *info );

void FC_FUNC(zgbcon,ZGBCON)( char* norm, const integer* n, const integer* kl, const integer* ku,
                    const dcomplex* ab, const integer* ldab, const integer* ipiv,
                    double* anorm, double* rcond, dcomplex* work,
                    double* rwork, integer *info );

void FC_FUNC(sgtcon,SGTCON)( char* norm, const integer* n, const float* dl,
                    const float* d, const float* du, const float* du2,
                    const integer* ipiv, float* anorm, float* rcond,
                    float* work, integer* iwork, integer *info );

void FC_FUNC(dgtcon,DGTCON)( char* norm, const integer* n, const double* dl,
                    const double* d, const double* du, const double* du2,
                    const integer* ipiv, double* anorm, double* rcond,
                    double* work, integer* iwork, integer *info );

void FC_FUNC(cgtcon,CGTCON)( char* norm, const integer* n, const scomplex* dl,
                    const scomplex* d, const scomplex* du, const scomplex* du2,
                    const integer* ipiv, float* anorm, float* rcond,
                    scomplex* work, integer *info );

void FC_FUNC(zgtcon,ZGTCON)( char* norm, const integer* n, const dcomplex* dl,
                    const dcomplex* d, const dcomplex* du, const dcomplex* du2,
                    const integer* ipiv, double* anorm, double* rcond,
                    dcomplex* work, integer *info );

void FC_FUNC(spocon,SPOCON)( const char* uplo, const integer* n, const float* a, const integer* lda,
                    float* anorm, float* rcond, float* work,
                    integer* iwork, integer *info );

void FC_FUNC(dpocon,DPOCON)( const char* uplo, const integer* n, const double* a, const integer* lda,
                    double* anorm, double* rcond, double* work,
                    integer* iwork, integer *info );

void FC_FUNC(cpocon,CPOCON)( const char* uplo, const integer* n, const scomplex* a, const integer* lda,
                    float* anorm, float* rcond, scomplex* work,
                    float* rwork, integer *info );

void FC_FUNC(zpocon,ZPOCON)( const char* uplo, const integer* n, const dcomplex* a, const integer* lda,
                    double* anorm, double* rcond, dcomplex* work,
                    double* rwork, integer *info );

void FC_FUNC(sppcon,SPPCON)( const char* uplo, const integer* n, const float* ap, float* anorm,
                    float* rcond, float* work, integer* iwork,
                    integer *info );

void FC_FUNC(dppcon,DPPCON)( const char* uplo, const integer* n, const double* ap, double* anorm,
                    double* rcond, double* work, integer* iwork,
                    integer *info );

void FC_FUNC(cppcon,CPPCON)( const char* uplo, const integer* n, const scomplex* ap, float* anorm,
                    float* rcond, scomplex* work, float* rwork,
                    integer *info );

void FC_FUNC(zppcon,ZPPCON)( const char* uplo, const integer* n, const dcomplex* ap, double* anorm,
                    double* rcond, dcomplex* work, double* rwork,
                    integer *info );

void FC_FUNC(spbcon,SPBCON)( const char* uplo, const integer* n, const integer* kd, const float* ab,
                    const integer* ldab, float* anorm, float* rcond,
                    float* work, integer* iwork, integer *info );

void FC_FUNC(dpbcon,DPBCON)( const char* uplo, const integer* n, const integer* kd, const double* ab,
                    const integer* ldab, double* anorm, double* rcond,
                    double* work, integer* iwork, integer *info );

void FC_FUNC(cpbcon,CPBCON)( const char* uplo, const integer* n, const integer* kd, const scomplex* ab,
                    const integer* ldab, float* anorm, float* rcond,
                    scomplex* work, float* iwork, integer *info );

void FC_FUNC(zpbcon,ZPBCON)( const char* uplo, const integer* n, const integer* kd, const dcomplex* ab,
                    const integer* ldab, double* anorm, double* rcond,
                    dcomplex* work, double* iwork, integer *info );

void FC_FUNC(sptcon,SPTCON)( const integer* n, const float* d, const float* e,
                    float* anorm, float* rcond, float* work,
                    integer *info );

void FC_FUNC(dptcon,DPTCON)( const integer* n, const double* d, const double* e,
                    double* anorm, double* rcond, double* work,
                    integer *info );

void FC_FUNC(cptcon,CPTCON)( const integer* n, const float* d, const scomplex* e,
                    float* anorm, float* rcond, float* work,
                    integer *info );

void FC_FUNC(zptcon,ZPTCON)( const integer* n, const double* d, const dcomplex* e,
                    double* anorm, double* rcond, double* work,
                    integer *info );

void FC_FUNC(ssycon,SSYCON)( const char* uplo, const integer* n, const float* a, const integer* lda,
                    const integer* ipiv, float* anorm, float* rcond,
                    float* work, integer* iwork, integer *info );

void FC_FUNC(dsycon,DSYCON)( const char* uplo, const integer* n, const double* a, const integer* lda,
                    const integer* ipiv, double* anorm, double* rcond,
                    double* work, integer* iwork, integer *info );

void FC_FUNC(csycon,CSYCON)( const char* uplo, const integer* n, const scomplex* a, const integer* lda,
                    const integer* ipiv, float* anorm, float* rcond,
                    scomplex* work, integer *info );

void FC_FUNC(zsycon,ZSYCON)( const char* uplo, const integer* n, const dcomplex* a, const integer* lda,
                    const integer* ipiv, double* anorm, double* rcond,
                    dcomplex* work, integer *info );

void FC_FUNC(checon,CHECON)( const char* uplo, const integer* n, const scomplex* a, const integer* lda,
                    const integer* ipiv, float* anorm, float* rcond,
                    scomplex* work, integer *info );

void FC_FUNC(zhecon,ZHECON)( const char* uplo, const integer* n, const dcomplex* a, const integer* lda,
                    const integer* ipiv, double* anorm, double* rcond,
                    dcomplex* work, integer *info );

void FC_FUNC(sspcon,SSPCON)( const char* uplo, const integer* n, const float* ap,
                    const integer* ipiv, float* anorm, float* rcond,
                    float* work, integer* iwork, integer *info );

void FC_FUNC(dspcon,DSPCON)( const char* uplo, const integer* n, const double* ap,
                    const integer* ipiv, double* anorm, double* rcond,
                    double* work, integer* iwork, integer *info );

void FC_FUNC(cspcon,CSPCON)( const char* uplo, const integer* n, const scomplex* ap,
                    const integer* ipiv, float* anorm, float* rcond,
                    scomplex* work, integer *info );

void FC_FUNC(zspcon,ZSPCON)( const char* uplo, const integer* n, const dcomplex* ap,
                    const integer* ipiv, double* anorm, double* rcond,
                    dcomplex* work, integer *info );

void FC_FUNC(chpcon,CHPCON)( const char* uplo, const integer* n, const scomplex* ap,
                    const integer* ipiv, float* anorm, float* rcond,
                    scomplex* work, integer *info );

void FC_FUNC(zhpcon,ZHPCON)( const char* uplo, const integer* n, const dcomplex* ap,
                    const integer* ipiv, double* anorm, double* rcond,
                    dcomplex* work, integer *info );

void FC_FUNC(strcon,STRCON)( char* norm, const char* uplo, char* diag, const integer* n,
                    const float* a, const integer* lda, float* rcond,
                    float* work, integer* iwork, integer *info );

void FC_FUNC(dtrcon,DTRCON)( char* norm, const char* uplo, char* diag, const integer* n,
                    const double* a, const integer* lda, double* rcond,
                    double* work, integer* iwork, integer *info );

void FC_FUNC(ctrcon,CTRCON)( char* norm, const char* uplo, char* diag, const integer* n,
                    const scomplex* a, const integer* lda, float* rcond,
                    scomplex* work, float* rwork, integer *info );

void FC_FUNC(ztrcon,ZTRCON)( char* norm, const char* uplo, char* diag, const integer* n,
                    const dcomplex* a, const integer* lda, double* rcond,
                    dcomplex* work, double* rwork, integer *info );

void FC_FUNC(stpcon,STPCON)( char* norm, const char* uplo, char* diag, const integer* n,
                    const float* ap, float* rcond, float* work,
                    integer* iwork, integer *info );

void FC_FUNC(dtpcon,DTPCON)( char* norm, const char* uplo, char* diag, const integer* n,
                    const double* ap, double* rcond, double* work,
                    integer* iwork, integer *info );

void FC_FUNC(ctpcon,CTPCON)( char* norm, const char* uplo, char* diag, const integer* n,
                    const scomplex* ap, float* rcond, scomplex* work,
                    float* rwork, integer *info );

void FC_FUNC(ztpcon,ZTPCON)( char* norm, const char* uplo, char* diag, const integer* n,
                    const dcomplex* ap, double* rcond, dcomplex* work,
                    double* rwork, integer *info );

void FC_FUNC(stbcon,STBCON)( char* norm, const char* uplo, char* diag, const integer* n,
                    const integer* kd, const float* ab, const integer* ldab,
                    float* rcond, float* work, integer* iwork,
                    integer *info );

void FC_FUNC(dtbcon,DTBCON)( char* norm, const char* uplo, char* diag, const integer* n,
                    const integer* kd, const double* ab, const integer* ldab,
                    double* rcond, double* work, integer* iwork,
                    integer *info );

void FC_FUNC(ctbcon,CTBCON)( char* norm, const char* uplo, char* diag, const integer* n,
                    const integer* kd, const scomplex* ab, const integer* ldab,
                    float* rcond, scomplex* work, float* rwork,
                    integer *info );

void FC_FUNC(ztbcon,ZTBCON)( char* norm, const char* uplo, char* diag, const integer* n,
                    const integer* kd, const dcomplex* ab, const integer* ldab,
                    double* rcond, dcomplex* work, double* rwork,
                    integer *info );

void FC_FUNC(sgerfs,SGERFS)( const char* trans, const integer* n, const integer* nrhs,
                    const float* a, const integer* lda, const float* af,
                    const integer* ldaf, const integer* ipiv, const float* b,
                    const integer* ldb, float* x, const integer* ldx, float* ferr,
                    float* berr, float* work, integer* iwork,
                    integer *info );

void FC_FUNC(dgerfs,DGERFS)( const char* trans, const integer* n, const integer* nrhs,
                    const double* a, const integer* lda, const double* af,
                    const integer* ldaf, const integer* ipiv, const double* b,
                    const integer* ldb, double* x, const integer* ldx, double* ferr,
                    double* berr, double* work, integer* iwork,
                    integer *info );

void FC_FUNC(cgerfs,CGERFS)( const char* trans, const integer* n, const integer* nrhs,
                    const scomplex* a, const integer* lda, const scomplex* af,
                    const integer* ldaf, const integer* ipiv, const scomplex* b,
                    const integer* ldb, scomplex* x, const integer* ldx, float* ferr,
                    float* berr, scomplex* work, float* rwork,
                    integer *info );

void FC_FUNC(zgerfs,ZGERFS)( const char* trans, const integer* n, const integer* nrhs,
                    const dcomplex* a, const integer* lda, const dcomplex* af,
                    const integer* ldaf, const integer* ipiv, const dcomplex* b,
                    const integer* ldb, dcomplex* x, const integer* ldx, double* ferr,
                    double* berr, dcomplex* work, double* rwork,
                    integer *info );

void FC_FUNC(sgerfsx,SGERFSX)( const char* trans, char* equed, const integer* n, const integer* nrhs,
                     const float* a, const integer* lda, const float* af,
                     const integer* ldaf, const integer* ipiv, const float* r,
                     const float* c, const float* b, const integer* ldb,
                     float* x, const integer* ldx, float* rcond, float* berr,
                     const integer* n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, const integer* nparams, float* params,
                     float* work, integer* iwork, integer *info );

void FC_FUNC(dgerfsx,DGERFSX)( const char* trans, char* equed, const integer* n, const integer* nrhs,
                     const double* a, const integer* lda, const double* af,
                     const integer* ldaf, const integer* ipiv, const double* r,
                     const double* c, const double* b, const integer* ldb,
                     double* x, const integer* ldx, double* rcond, double* berr,
                     const integer* n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, const integer* nparams, double* params,
                     double* work, integer* iwork, integer *info );

void FC_FUNC(cgerfsx,CGERFSX)( const char* trans, char* equed, const integer* n, const integer* nrhs,
                     const scomplex* a, const integer* lda, const scomplex* af,
                     const integer* ldaf, const integer* ipiv, const float* r,
                     const float* c, const scomplex* b, const integer* ldb,
                     scomplex* x, const integer* ldx, float* rcond, float* berr,
                     const integer* n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, const integer* nparams, float* params,
                     scomplex* work, float* rwork, integer *info );

void FC_FUNC(zgerfsx,ZGERFSX)( const char* trans, char* equed, const integer* n, const integer* nrhs,
                     const dcomplex* a, const integer* lda, const dcomplex* af,
                     const integer* ldaf, const integer* ipiv, const double* r,
                     const double* c, const dcomplex* b, const integer* ldb,
                     dcomplex* x, const integer* ldx, double* rcond, double* berr,
                     const integer* n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, const integer* nparams, double* params,
                     dcomplex* work, double* rwork, integer *info );

void FC_FUNC(sgbrfs,SGBRFS)( const char* trans, const integer* n, const integer* kl, const integer* ku,
                    const integer* nrhs, const float* ab, const integer* ldab,
                    const float* afb, const integer* ldafb,
                    const integer* ipiv, const float* b, const integer* ldb,
                    float* x, const integer* ldx, float* ferr, float* berr,
                    float* work, integer* iwork, integer *info );

void FC_FUNC(dgbrfs,DGBRFS)( const char* trans, const integer* n, const integer* kl, const integer* ku,
                    const integer* nrhs, const double* ab, const integer* ldab,
                    const double* afb, const integer* ldafb,
                    const integer* ipiv, const double* b, const integer* ldb,
                    double* x, const integer* ldx, double* ferr, double* berr,
                    double* work, integer* iwork, integer *info );

void FC_FUNC(cgbrfs,CGBRFS)( const char* trans, const integer* n, const integer* kl, const integer* ku,
                    const integer* nrhs, const scomplex* ab, const integer* ldab,
                    const scomplex* afb, const integer* ldafb,
                    const integer* ipiv, const scomplex* b, const integer* ldb,
                    scomplex* x, const integer* ldx, float* ferr, float* berr,
                    scomplex* work, float* rwork, integer *info );

void FC_FUNC(zgbrfs,ZGBRFS)( const char* trans, const integer* n, const integer* kl, const integer* ku,
                    const integer* nrhs, const dcomplex* ab, const integer* ldab,
                    const dcomplex* afb, const integer* ldafb,
                    const integer* ipiv, const dcomplex* b, const integer* ldb,
                    dcomplex* x, const integer* ldx, double* ferr, double* berr,
                    dcomplex* work, double* rwork, integer *info );

void FC_FUNC(sgbrfsx,SGBRFSX)( const char* trans, char* equed, const integer* n, const integer* kl,
                     const integer* ku, const integer* nrhs, const float* ab,
                     const integer* ldab, const float* afb, const integer* ldafb,
                     const integer* ipiv, const float* r, const float* c,
                     const float* b, const integer* ldb, float* x,
                     const integer* ldx, float* rcond, float* berr,
                     const integer* n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, const integer* nparams, float* params,
                     float* work, integer* iwork, integer *info );

void FC_FUNC(dgbrfsx,DGBRFSX)( const char* trans, char* equed, const integer* n, const integer* kl,
                     const integer* ku, const integer* nrhs, const double* ab,
                     const integer* ldab, const double* afb, const integer* ldafb,
                     const integer* ipiv, const double* r, const double* c,
                     const double* b, const integer* ldb, double* x,
                     const integer* ldx, double* rcond, double* berr,
                     const integer* n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, const integer* nparams, double* params,
                     double* work, integer* iwork, integer *info );

void FC_FUNC(cgbrfsx,CGBRFSX)( const char* trans, char* equed, const integer* n, const integer* kl,
                     const integer* ku, const integer* nrhs, const scomplex* ab,
                     const integer* ldab, const scomplex* afb, const integer* ldafb,
                     const integer* ipiv, const float* r, const float* c,
                     const scomplex* b, const integer* ldb, scomplex* x,
                     const integer* ldx, float* rcond, float* berr,
                     const integer* n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, const integer* nparams, float* params,
                     scomplex* work, float* rwork, integer *info );

void FC_FUNC(zgbrfsx,ZGBRFSX)( const char* trans, char* equed, const integer* n, const integer* kl,
                     const integer* ku, const integer* nrhs, const dcomplex* ab,
                     const integer* ldab, const dcomplex* afb, const integer* ldafb,
                     const integer* ipiv, const double* r, const double* c,
                     const dcomplex* b, const integer* ldb, dcomplex* x,
                     const integer* ldx, double* rcond, double* berr,
                     const integer* n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, const integer* nparams, double* params,
                     dcomplex* work, double* rwork, integer *info );

void FC_FUNC(sgtrfs,SGTRFS)( const char* trans, const integer* n, const integer* nrhs,
                    const float* dl, const float* d, const float* du,
                    const float* dlf, const float* df, const float* duf,
                    const float* du2, const integer* ipiv, const float* b,
                    const integer* ldb, float* x, const integer* ldx, float* ferr,
                    float* berr, float* work, integer* iwork,
                    integer *info );

void FC_FUNC(dgtrfs,DGTRFS)( const char* trans, const integer* n, const integer* nrhs,
                    const double* dl, const double* d, const double* du,
                    const double* dlf, const double* df, const double* duf,
                    const double* du2, const integer* ipiv, const double* b,
                    const integer* ldb, double* x, const integer* ldx, double* ferr,
                    double* berr, double* work, integer* iwork,
                    integer *info );

void FC_FUNC(cgtrfs,CGTRFS)( const char* trans, const integer* n, const integer* nrhs,
                    const scomplex* dl, const scomplex* d, const scomplex* du,
                    const scomplex* dlf, const scomplex* df, const scomplex* duf,
                    const scomplex* du2, const integer* ipiv, const scomplex* b,
                    const integer* ldb, scomplex* x, const integer* ldx, float* ferr,
                    float* berr, scomplex* work, float* rwork,
                    integer *info );

void FC_FUNC(zgtrfs,ZGTRFS)( const char* trans, const integer* n, const integer* nrhs,
                    const dcomplex* dl, const dcomplex* d, const dcomplex* du,
                    const dcomplex* dlf, const dcomplex* df, const dcomplex* duf,
                    const dcomplex* du2, const integer* ipiv, const dcomplex* b,
                    const integer* ldb, dcomplex* x, const integer* ldx, double* ferr,
                    double* berr, dcomplex* work, double* rwork,
                    integer *info );

void FC_FUNC(sporfs,SPORFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const float* a, const integer* lda, const float* af,
                    const integer* ldaf, const float* b, const integer* ldb,
                    float* x, const integer* ldx, float* ferr, float* berr,
                    float* work, integer* iwork, integer *info );

void FC_FUNC(dporfs,DPORFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const double* a, const integer* lda, const double* af,
                    const integer* ldaf, const double* b, const integer* ldb,
                    double* x, const integer* ldx, double* ferr, double* berr,
                    double* work, integer* iwork, integer *info );

void FC_FUNC(cporfs,CPORFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* a, const integer* lda, const scomplex* af,
                    const integer* ldaf, const scomplex* b, const integer* ldb,
                    scomplex* x, const integer* ldx, float* ferr, float* berr,
                    scomplex* work, float* rwork, integer *info );

void FC_FUNC(zporfs,ZPORFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* a, const integer* lda, const dcomplex* af,
                    const integer* ldaf, const dcomplex* b, const integer* ldb,
                    dcomplex* x, const integer* ldx, double* ferr, double* berr,
                    dcomplex* work, double* rwork, integer *info );

void FC_FUNC(sporfsx,SPORFSX)( const char* uplo, char* equed, const integer* n, const integer* nrhs,
                     const float* a, const integer* lda, const float* af,
                     const integer* ldaf, const float* s, const float* b,
                     const integer* ldb, float* x, const integer* ldx, float* rcond,
                     float* berr, const integer* n_err_bnds,
                     float* err_bnds_norm, float* err_bnds_comp,
                     const integer* nparams, float* params, float* work,
                     integer* iwork, integer *info );

void FC_FUNC(dporfsx,DPORFSX)( const char* uplo, char* equed, const integer* n, const integer* nrhs,
                     const double* a, const integer* lda, const double* af,
                     const integer* ldaf, const double* s, const double* b,
                     const integer* ldb, double* x, const integer* ldx, double* rcond,
                     double* berr, const integer* n_err_bnds,
                     double* err_bnds_norm, double* err_bnds_comp,
                     const integer* nparams, double* params, double* work,
                     integer* iwork, integer *info );

void FC_FUNC(cporfsx,CPORFSX)( const char* uplo, char* equed, const integer* n, const integer* nrhs,
                     const scomplex* a, const integer* lda, const scomplex* af,
                     const integer* ldaf, const float* s, const scomplex* b,
                     const integer* ldb, scomplex* x, const integer* ldx, float* rcond,
                     float* berr, const integer* n_err_bnds,
                     float* err_bnds_norm, float* err_bnds_comp,
                     const integer* nparams, float* params, scomplex* work,
                     float* rwork, integer *info );

void FC_FUNC(zporfsx,ZPORFSX)( const char* uplo, char* equed, const integer* n, const integer* nrhs,
                     const dcomplex* a, const integer* lda, const dcomplex* af,
                     const integer* ldaf, const double* s, const dcomplex* b,
                     const integer* ldb, dcomplex* x, const integer* ldx, double* rcond,
                     double* berr, const integer* n_err_bnds,
                     double* err_bnds_norm, double* err_bnds_comp,
                     const integer* nparams, double* params, dcomplex* work,
                     double* rwork, integer *info );

void FC_FUNC(spprfs,SPPRFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const float* ap, const float* afp, const float* b,
                    const integer* ldb, float* x, const integer* ldx, float* ferr,
                    float* berr, float* work, integer* iwork,
                    integer *info );

void FC_FUNC(dpprfs,DPPRFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const double* ap, const double* afp, const double* b,
                    const integer* ldb, double* x, const integer* ldx, double* ferr,
                    double* berr, double* work, integer* iwork,
                    integer *info );

void FC_FUNC(cpprfs,CPPRFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* ap, const scomplex* afp, const scomplex* b,
                    const integer* ldb, scomplex* x, const integer* ldx, float* ferr,
                    float* berr, scomplex* work, float* rwork,
                    integer *info );

void FC_FUNC(zpprfs,ZPPRFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* ap, const dcomplex* afp, const dcomplex* b,
                    const integer* ldb, dcomplex* x, const integer* ldx, double* ferr,
                    double* berr, dcomplex* work, double* rwork,
                    integer *info );

void FC_FUNC(spbrfs,SPBRFS)( const char* uplo, const integer* n, const integer* kd, const integer* nrhs,
                    const float* ab, const integer* ldab, const float* afb,
                    const integer* ldafb, const float* b, const integer* ldb,
                    float* x, const integer* ldx, float* ferr, float* berr,
                    float* work, integer* iwork, integer *info );

void FC_FUNC(dpbrfs,DPBRFS)( const char* uplo, const integer* n, const integer* kd, const integer* nrhs,
                    const double* ab, const integer* ldab, const double* afb,
                    const integer* ldafb, const double* b, const integer* ldb,
                    double* x, const integer* ldx, double* ferr, double* berr,
                    double* work, integer* iwork, integer *info );

void FC_FUNC(cpbrfs,CPBRFS)( const char* uplo, const integer* n, const integer* kd, const integer* nrhs,
                    const scomplex* ab, const integer* ldab, const scomplex* afb,
                    const integer* ldafb, const scomplex* b, const integer* ldb,
                    scomplex* x, const integer* ldx, float* ferr, float* berr,
                    scomplex* work, float* rwork, integer *info );

void FC_FUNC(zpbrfs,ZPBRFS)( const char* uplo, const integer* n, const integer* kd, const integer* nrhs,
                    const dcomplex* ab, const integer* ldab, const dcomplex* afb,
                    const integer* ldafb, const dcomplex* b, const integer* ldb,
                    dcomplex* x, const integer* ldx, double* ferr, double* berr,
                    dcomplex* work, double* rwork, integer *info );

void FC_FUNC(sptrfs,SPTRFS)( const integer* n, const integer* nrhs, const float* d,
                    const float* e, const float* df, const float* ef,
                    const float* b, const integer* ldb, float* x,
                    const integer* ldx, float* ferr, float* berr, float* work,
                    integer *info );

void FC_FUNC(dptrfs,DPTRFS)( const integer* n, const integer* nrhs, const double* d,
                    const double* e, const double* df, const double* ef,
                    const double* b, const integer* ldb, double* x,
                    const integer* ldx, double* ferr, double* berr, double* work,
                    integer *info );

void FC_FUNC(cptrfs,CPTRFS)( const integer* n, const integer* nrhs, const float* d,
                    const scomplex* e, const float* df, const scomplex* ef,
                    const scomplex* b, const integer* ldb, scomplex* x,
                    const integer* ldx, float* ferr, float* berr, scomplex* work,
                    float* rwork, integer *info );

void FC_FUNC(zptrfs,ZPTRFS)( const integer* n, const integer* nrhs, const double* d,
                    const dcomplex* e, const double* df, const dcomplex* ef,
                    const dcomplex* b, const integer* ldb, dcomplex* x,
                    const integer* ldx, double* ferr, double* berr, dcomplex* work,
                    double* rwork, integer *info );

void FC_FUNC(ssyrfs,SSYRFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const float* a, const integer* lda, const float* af,
                    const integer* ldaf, const integer* ipiv, const float* b,
                    const integer* ldb, float* x, const integer* ldx, float* ferr,
                    float* berr, float* work, integer* iwork,
                    integer *info );

void FC_FUNC(dsyrfs,DSYRFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const double* a, const integer* lda, const double* af,
                    const integer* ldaf, const integer* ipiv, const double* b,
                    const integer* ldb, double* x, const integer* ldx, double* ferr,
                    double* berr, double* work, integer* iwork,
                    integer *info );

void FC_FUNC(csyrfs,CSYRFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* a, const integer* lda, const scomplex* af,
                    const integer* ldaf, const integer* ipiv, const scomplex* b,
                    const integer* ldb, scomplex* x, const integer* ldx, float* ferr,
                    float* berr, scomplex* work, float* rwork,
                    integer *info );

void FC_FUNC(zsyrfs,ZSYRFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* a, const integer* lda, const dcomplex* af,
                    const integer* ldaf, const integer* ipiv, const dcomplex* b,
                    const integer* ldb, dcomplex* x, const integer* ldx, double* ferr,
                    double* berr, dcomplex* work, double* rwork,
                    integer *info );

void FC_FUNC(cherfs,CHERFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* a, const integer* lda, const scomplex* af,
                    const integer* ldaf, const integer* ipiv, const scomplex* b,
                    const integer* ldb, scomplex* x, const integer* ldx, float* ferr,
                    float* berr, scomplex* work, float* rwork,
                    integer *info );

void FC_FUNC(zherfs,ZHERFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* a, const integer* lda, const dcomplex* af,
                    const integer* ldaf, const integer* ipiv, const dcomplex* b,
                    const integer* ldb, dcomplex* x, const integer* ldx, double* ferr,
                    double* berr, dcomplex* work, double* rwork,
                    integer *info );

void FC_FUNC(ssyrfsx,SSYRFSX)( const char* uplo, char* equed, const integer* n, const integer* nrhs,
                     const float* a, const integer* lda, const float* af,
                     const integer* ldaf, const integer* ipiv, const float* s,
                     const float* b, const integer* ldb, float* x,
                     const integer* ldx, float* rcond, float* berr,
                     const integer* n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, const integer* nparams, float* params,
                     float* work, integer* iwork, integer *info );

void FC_FUNC(dsyrfsx,DSYRFSX)( const char* uplo, char* equed, const integer* n, const integer* nrhs,
                     const double* a, const integer* lda, const double* af,
                     const integer* ldaf, const integer* ipiv, const double* s,
                     const double* b, const integer* ldb, double* x,
                     const integer* ldx, double* rcond, double* berr,
                     const integer* n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, const integer* nparams, double* params,
                     double* work, integer* iwork, integer *info );

void FC_FUNC(csyrfsx,CSYRFSX)( const char* uplo, char* equed, const integer* n, const integer* nrhs,
                     const scomplex* a, const integer* lda, const scomplex* af,
                     const integer* ldaf, const integer* ipiv, const float* s,
                     const scomplex* b, const integer* ldb, scomplex* x,
                     const integer* ldx, float* rcond, float* berr,
                     const integer* n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, const integer* nparams, float* params,
                     scomplex* work, float* rwork, integer *info );

void FC_FUNC(zsyrfsx,ZSYRFSX)( const char* uplo, char* equed, const integer* n, const integer* nrhs,
                     const dcomplex* a, const integer* lda, const dcomplex* af,
                     const integer* ldaf, const integer* ipiv, const double* s,
                     const dcomplex* b, const integer* ldb, dcomplex* x,
                     const integer* ldx, double* rcond, double* berr,
                     const integer* n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, const integer* nparams, double* params,
                     dcomplex* work, double* rwork, integer *info );

void FC_FUNC(cherfsx,CHERFSX)( const char* uplo, char* equed, const integer* n, const integer* nrhs,
                     const scomplex* a, const integer* lda, const scomplex* af,
                     const integer* ldaf, const integer* ipiv, const float* s,
                     const scomplex* b, const integer* ldb, scomplex* x,
                     const integer* ldx, float* rcond, float* berr,
                     const integer* n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, const integer* nparams, float* params,
                     scomplex* work, float* rwork, integer *info );

void FC_FUNC(zherfsx,ZHERFSX)( const char* uplo, char* equed, const integer* n, const integer* nrhs,
                     const dcomplex* a, const integer* lda, const dcomplex* af,
                     const integer* ldaf, const integer* ipiv, const double* s,
                     const dcomplex* b, const integer* ldb, dcomplex* x,
                     const integer* ldx, double* rcond, double* berr,
                     const integer* n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, const integer* nparams, double* params,
                     dcomplex* work, double* rwork, integer *info );

void FC_FUNC(ssprfs,SSPRFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const float* ap, const float* afp, const integer* ipiv,
                    const float* b, const integer* ldb, float* x,
                    const integer* ldx, float* ferr, float* berr, float* work,
                    integer* iwork, integer *info );

void FC_FUNC(dsprfs,DSPRFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const double* ap, const double* afp, const integer* ipiv,
                    const double* b, const integer* ldb, double* x,
                    const integer* ldx, double* ferr, double* berr, double* work,
                    integer* iwork, integer *info );

void FC_FUNC(csprfs,CSPRFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* ap, const scomplex* afp, const integer* ipiv,
                    const scomplex* b, const integer* ldb, scomplex* x,
                    const integer* ldx, float* ferr, float* berr, scomplex* work,
                    float* rwork, integer *info );

void FC_FUNC(zsprfs,ZSPRFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* ap, const dcomplex* afp, const integer* ipiv,
                    const dcomplex* b, const integer* ldb, dcomplex* x,
                    const integer* ldx, double* ferr, double* berr, dcomplex* work,
                    double* rwork, integer *info );

void FC_FUNC(chprfs,CHPRFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* ap, const scomplex* afp, const integer* ipiv,
                    const scomplex* b, const integer* ldb, scomplex* x,
                    const integer* ldx, float* ferr, float* berr, scomplex* work,
                    float* rwork, integer *info );

void FC_FUNC(zhprfs,ZHPRFS)( const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* ap, const dcomplex* afp, const integer* ipiv,
                    const dcomplex* b, const integer* ldb, dcomplex* x,
                    const integer* ldx, double* ferr, double* berr, dcomplex* work,
                    double* rwork, integer *info );

void FC_FUNC(strrfs,STRRFS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* nrhs, const float* a, const integer* lda,
                    const float* b, const integer* ldb, const float* x,
                    const integer* ldx, float* ferr, float* berr, float* work,
                    integer* iwork, integer *info );

void FC_FUNC(dtrrfs,DTRRFS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* nrhs, const double* a, const integer* lda,
                    const double* b, const integer* ldb, const double* x,
                    const integer* ldx, double* ferr, double* berr, double* work,
                    integer* iwork, integer *info );

void FC_FUNC(ctrrfs,CTRRFS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* nrhs, const scomplex* a, const integer* lda,
                    const scomplex* b, const integer* ldb, const scomplex* x,
                    const integer* ldx, float* ferr, float* berr, scomplex* work,
                    float* rwork, integer *info );

void FC_FUNC(ztrrfs,ZTRRFS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* nrhs, const dcomplex* a, const integer* lda,
                    const dcomplex* b, const integer* ldb, const dcomplex* x,
                    const integer* ldx, double* ferr, double* berr, dcomplex* work,
                    double* rwork, integer *info );

void FC_FUNC(stprfs,STPRFS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* nrhs, const float* ap, const float* b,
                    const integer* ldb, const float* x, const integer* ldx,
                    float* ferr, float* berr, float* work, integer* iwork,
                    integer *info );

void FC_FUNC(dtprfs,DTPRFS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* nrhs, const double* ap, const double* b,
                    const integer* ldb, const double* x, const integer* ldx,
                    double* ferr, double* berr, double* work, integer* iwork,
                    integer *info );

void FC_FUNC(ctprfs,CTPRFS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* nrhs, const scomplex* ap, const scomplex* b,
                    const integer* ldb, const scomplex* x, const integer* ldx,
                    float* ferr, float* berr, scomplex* work, float* rwork,
                    integer *info );

void FC_FUNC(ztprfs,ZTPRFS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* nrhs, const dcomplex* ap, const dcomplex* b,
                    const integer* ldb, const dcomplex* x, const integer* ldx,
                    double* ferr, double* berr, dcomplex* work, double* rwork,
                    integer *info );

void FC_FUNC(stbrfs,STBRFS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* kd, const integer* nrhs, const float* ab,
                    const integer* ldab, const float* b, const integer* ldb,
                    const float* x, const integer* ldx, float* ferr,
                    float* berr, float* work, integer* iwork,
                    integer *info );

void FC_FUNC(dtbrfs,DTBRFS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* kd, const integer* nrhs, const double* ab,
                    const integer* ldab, const double* b, const integer* ldb,
                    const double* x, const integer* ldx, double* ferr,
                    double* berr, double* work, integer* iwork,
                    integer *info );

void FC_FUNC(ctbrfs,CTBRFS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* kd, const integer* nrhs, const scomplex* ab,
                    const integer* ldab, const scomplex* b, const integer* ldb,
                    const scomplex* x, const integer* ldx, float* ferr,
                    float* berr, scomplex* work, float* rwork,
                    integer *info );

void FC_FUNC(ztbrfs,ZTBRFS)( const char* uplo, const char* trans, char* diag, const integer* n,
                    const integer* kd, const integer* nrhs, const dcomplex* ab,
                    const integer* ldab, const dcomplex* b, const integer* ldb,
                    const dcomplex* x, const integer* ldx, double* ferr,
                    double* berr, dcomplex* work, double* rwork,
                    integer *info );

void FC_FUNC(sgetri,SGETRI)( const integer* n, float* a, const integer* lda,
                    const integer* ipiv, float* work, integer* lwork,
                    integer *info );

void FC_FUNC(dgetri,DGETRI)( const integer* n, double* a, const integer* lda,
                    const integer* ipiv, double* work, integer* lwork,
                    integer *info );

void FC_FUNC(cgetri,CGETRI)( const integer* n, scomplex* a, const integer* lda,
                    const integer* ipiv, scomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(zgetri,ZGETRI)( const integer* n, dcomplex* a, const integer* lda,
                    const integer* ipiv, dcomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(spotri,SPOTRI)( const char* uplo, const integer* n, float* a, const integer* lda,
                    integer *info );

void FC_FUNC(dpotri,DPOTRI)( const char* uplo, const integer* n, double* a, const integer* lda,
                    integer *info );

void FC_FUNC(cpotri,CPOTRI)( const char* uplo, const integer* n, scomplex* a, const integer* lda,
                    integer *info );

void FC_FUNC(zpotri,ZPOTRI)( const char* uplo, const integer* n, dcomplex* a, const integer* lda,
                    integer *info );

void FC_FUNC(spftri,SPFTRI)( const char* transr, const char* uplo, const integer* n, float* a,
                    integer *info );

void FC_FUNC(dpftri,DPFTRI)( const char* transr, const char* uplo, const integer* n, double* a,
                    integer *info );

void FC_FUNC(cpftri,CPFTRI)( const char* transr, const char* uplo, const integer* n, scomplex* a,
                    integer *info );

void FC_FUNC(zpftri,ZPFTRI)( const char* transr, const char* uplo, const integer* n, dcomplex* a,
                    integer *info );

void FC_FUNC(spptri,SPPTRI)( const char* uplo, const integer* n, float* ap, integer *info );

void FC_FUNC(dpptri,DPPTRI)( const char* uplo, const integer* n, double* ap, integer *info );

void FC_FUNC(cpptri,CPPTRI)( const char* uplo, const integer* n, scomplex* ap, integer *info );

void FC_FUNC(zpptri,ZPPTRI)( const char* uplo, const integer* n, dcomplex* ap, integer *info );

void FC_FUNC(ssytri,SSYTRI)( const char* uplo, const integer* n, float* a, const integer* lda,
                    const integer* ipiv, float* work, integer *info );

void FC_FUNC(dsytri,DSYTRI)( const char* uplo, const integer* n, double* a, const integer* lda,
                    const integer* ipiv, double* work, integer *info );

void FC_FUNC(csytri,CSYTRI)( const char* uplo, const integer* n, scomplex* a, const integer* lda,
                    const integer* ipiv, scomplex* work, integer *info );

void FC_FUNC(zsytri,ZSYTRI)( const char* uplo, const integer* n, dcomplex* a, const integer* lda,
                    const integer* ipiv, dcomplex* work, integer *info );

void FC_FUNC(chetri,CHETRI)( const char* uplo, const integer* n, scomplex* a, const integer* lda,
                    const integer* ipiv, scomplex* work, integer *info );

void FC_FUNC(zhetri,ZHETRI)( const char* uplo, const integer* n, dcomplex* a, const integer* lda,
                    const integer* ipiv, dcomplex* work, integer *info );

void FC_FUNC(ssptri,SSPTRI)( const char* uplo, const integer* n, float* ap,
                    const integer* ipiv, float* work, integer *info );

void FC_FUNC(dsptri,DSPTRI)( const char* uplo, const integer* n, double* ap,
                    const integer* ipiv, double* work, integer *info );

void FC_FUNC(csptri,CSPTRI)( const char* uplo, const integer* n, scomplex* ap,
                    const integer* ipiv, scomplex* work, integer *info );

void FC_FUNC(zsptri,ZSPTRI)( const char* uplo, const integer* n, dcomplex* ap,
                    const integer* ipiv, dcomplex* work, integer *info );

void FC_FUNC(chptri,CHPTRI)( const char* uplo, const integer* n, scomplex* ap,
                    const integer* ipiv, scomplex* work, integer *info );

void FC_FUNC(zhptri,ZHPTRI)( const char* uplo, const integer* n, dcomplex* ap,
                    const integer* ipiv, dcomplex* work, integer *info );

void FC_FUNC(strtri,STRTRI)( const char* uplo, char* diag, const integer* n, float* a,
                    const integer* lda, integer *info );

void FC_FUNC(dtrtri,DTRTRI)( const char* uplo, char* diag, const integer* n, double* a,
                    const integer* lda, integer *info );

void FC_FUNC(ctrtri,CTRTRI)( const char* uplo, char* diag, const integer* n, scomplex* a,
                    const integer* lda, integer *info );

void FC_FUNC(ztrtri,ZTRTRI)( const char* uplo, char* diag, const integer* n, dcomplex* a,
                    const integer* lda, integer *info );

void FC_FUNC(stftri,STFTRI)( const char* transr, const char* uplo, char* diag, const integer* n,
                    float* a, integer *info );

void FC_FUNC(dtftri,DTFTRI)( const char* transr, const char* uplo, char* diag, const integer* n,
                    double* a, integer *info );

void FC_FUNC(ctftri,CTFTRI)( const char* transr, const char* uplo, char* diag, const integer* n,
                    scomplex* a, integer *info );

void FC_FUNC(ztftri,ZTFTRI)( const char* transr, const char* uplo, char* diag, const integer* n,
                    dcomplex* a, integer *info );

void FC_FUNC(stptri,STPTRI)( const char* uplo, char* diag, const integer* n, float* ap,
                    integer *info );

void FC_FUNC(dtptri,DTPTRI)( const char* uplo, char* diag, const integer* n, double* ap,
                    integer *info );

void FC_FUNC(ctptri,CTPTRI)( const char* uplo, char* diag, const integer* n, scomplex* ap,
                    integer *info );

void FC_FUNC(ztptri,ZTPTRI)( const char* uplo, char* diag, const integer* n, dcomplex* ap,
                    integer *info );

void FC_FUNC(sgeequ,SGEEQU)( const integer* m, const integer* n, const float* a,
                    const integer* lda, float* r, float* c, float* rowcnd,
                    float* colcnd, float* amax, integer *info );

void FC_FUNC(dgeequ,DGEEQU)( const integer* m, const integer* n, const double* a,
                    const integer* lda, double* r, double* c, double* rowcnd,
                    double* colcnd, double* amax, integer *info );

void FC_FUNC(cgeequ,CGEEQU)( const integer* m, const integer* n, const scomplex* a,
                    const integer* lda, float* r, float* c, float* rowcnd,
                    float* colcnd, float* amax, integer *info );

void FC_FUNC(zgeequ,ZGEEQU)( const integer* m, const integer* n, const dcomplex* a,
                    const integer* lda, double* r, double* c, double* rowcnd,
                    double* colcnd, double* amax, integer *info );

void FC_FUNC(sgeequb,SGEEQUB)( const integer* m, const integer* n, const float* a,
                    const integer* lda, float* r, float* c, float* rowcnd,
                    float* colcnd, float* amax, integer *info );

void FC_FUNC(dgeequb,DGEEQUB)( const integer* m, const integer* n, const double* a,
                    const integer* lda, double* r, double* c, double* rowcnd,
                    double* colcnd, double* amax, integer *info );

void FC_FUNC(cgeequb,CGEEQUB)( const integer* m, const integer* n, const scomplex* a,
                    const integer* lda, float* r, float* c, float* rowcnd,
                    float* colcnd, float* amax, integer *info );

void FC_FUNC(zgeequb,ZGEEQUB)( const integer* m, const integer* n, const dcomplex* a,
                    const integer* lda, double* r, double* c, double* rowcnd,
                    double* colcnd, double* amax, integer *info );

void FC_FUNC(sgbequ,SGBEQU)( const integer* m, const integer* n, const integer* kl,
                    const integer* ku, const float* ab, const integer* ldab,
                    float* r, float* c, float* rowcnd, float* colcnd,
                    float* amax, integer *info );

void FC_FUNC(dgbequ,DGBEQU)( const integer* m, const integer* n, const integer* kl,
                    const integer* ku, const double* ab, const integer* ldab,
                    double* r, double* c, double* rowcnd, double* colcnd,
                    double* amax, integer *info );

void FC_FUNC(cgbequ,CGBEQU)( const integer* m, const integer* n, const integer* kl,
                    const integer* ku, const scomplex* ab, const integer* ldab,
                    float* r, float* c, float* rowcnd, float* colcnd,
                    float* amax, integer *info );

void FC_FUNC(zgbequ,ZGBEQU)( const integer* m, const integer* n, const integer* kl,
                    const integer* ku, const dcomplex* ab, const integer* ldab,
                    double* r, double* c, double* rowcnd, double* colcnd,
                    double* amax, integer *info );

void FC_FUNC(sgbequb,SGBEQUB)( const integer* m, const integer* n, const integer* kl,
                    const integer* ku, const float* ab, const integer* ldab,
                    float* r, float* c, float* rowcnd, float* colcnd,
                    float* amax, integer *info );

void FC_FUNC(dgbequb,DGBEQUB)( const integer* m, const integer* n, const integer* kl,
                    const integer* ku, const double* ab, const integer* ldab,
                    double* r, double* c, double* rowcnd, double* colcnd,
                    double* amax, integer *info );

void FC_FUNC(cgbequb,CGBEQUB)( const integer* m, const integer* n, const integer* kl,
                    const integer* ku, const scomplex* ab, const integer* ldab,
                    float* r, float* c, float* rowcnd, float* colcnd,
                    float* amax, integer *info );

void FC_FUNC(zgbequb,ZGBEQUB)( const integer* m, const integer* n, const integer* kl,
                    const integer* ku, const dcomplex* ab, const integer* ldab,
                    double* r, double* c, double* rowcnd, double* colcnd,
                    double* amax, integer *info );

void FC_FUNC(spoequ,SPOEQU)( const integer* n, const float* a, const integer* lda, float* s,
                    float* scond, float* amax, integer *info );

void FC_FUNC(dpoequ,DPOEQU)( const integer* n, const double* a, const integer* lda, double* s,
                    double* scond, double* amax, integer *info );

void FC_FUNC(cpoequ,CPOEQU)( const integer* n, const scomplex* a, const integer* lda, float* s,
                    float* scond, float* amax, integer *info );

void FC_FUNC(zpoequ,ZPOEQU)( const integer* n, const dcomplex* a, const integer* lda, double* s,
                    double* scond, double* amax, integer *info );

void FC_FUNC(spoequb,SPOEQUB)( const integer* n, const float* a, const integer* lda, float* s,
                    float* scond, float* amax, integer *info );

void FC_FUNC(dpoequb,DPOEQUB)( const integer* n, const double* a, const integer* lda, double* s,
                    double* scond, double* amax, integer *info );

void FC_FUNC(cpoequb,CPOEQUB)( const integer* n, const scomplex* a, const integer* lda, float* s,
                    float* scond, float* amax, integer *info );

void FC_FUNC(zpoequb,ZPOEQUB)( const integer* n, const dcomplex* a, const integer* lda, double* s,
                    double* scond, double* amax, integer *info );

void FC_FUNC(sppequ,SPPEQU)( const char* uplo, const integer* n, const float* ap, float* s,
                    float* scond, float* amax, integer *info );

void FC_FUNC(dppequ,DPPEQU)( const char* uplo, const integer* n, const double* ap, double* s,
                    double* scond, double* amax, integer *info );

void FC_FUNC(cppequ,CPPEQU)( const char* uplo, const integer* n, const scomplex* ap, float* s,
                    float* scond, float* amax, integer *info );

void FC_FUNC(zppequ,ZPPEQU)( const char* uplo, const integer* n, const dcomplex* ap, double* s,
                    double* scond, double* amax, integer *info );

void FC_FUNC(spbequ,SPBEQU)( const char* uplo, const integer* n, const integer* kd, const float* ab,
                    const integer* ldab, float* s, float* scond, float* amax,
                    integer *info );

void FC_FUNC(dpbequ,DPBEQU)( const char* uplo, const integer* n, const integer* kd, const double* ab,
                    const integer* ldab, double* s, double* scond, double* amax,
                    integer *info );

void FC_FUNC(cpbequ,CPBEQU)( const char* uplo, const integer* n, const integer* kd, const scomplex* ab,
                    const integer* ldab, float* s, float* scond, float* amax,
                    integer *info );

void FC_FUNC(zpbequ,ZPBEQU)( const char* uplo, const integer* n, const integer* kd, const dcomplex* ab,
                    const integer* ldab, double* s, double* scond, double* amax,
                    integer *info );

void FC_FUNC(ssyequb,SSYEQUB)( const char* uplo, const integer* n, const float* a,
                     const integer* lda, float* s, float* scond, float* amax,
                     float* work, integer *info );

void FC_FUNC(dsyequb,DSYEQUB)( const char* uplo, const integer* n, const double* a,
                     const integer* lda, double* s, double* scond, double* amax,
                     double* work, integer *info );

void FC_FUNC(csyequb,CSYEQUB)( const char* uplo, const integer* n, const scomplex* a,
                     const integer* lda, float* s, float* scond, float* amax,
                     scomplex* work, integer *info );

void FC_FUNC(zsyequb,ZSYEQUB)( const char* uplo, const integer* n, const dcomplex* a,
                     const integer* lda, double* s, double* scond, double* amax,
                     dcomplex* work, integer *info );

void FC_FUNC(cheequb,CHEEQUB)( const char* uplo, const integer* n, const scomplex* a,
                     const integer* lda, float* s, float* scond, float* amax,
                     scomplex* work, integer *info );

void FC_FUNC(zheequb,ZHEEQUB)( const char* uplo, const integer* n, const dcomplex* a,
                     const integer* lda, double* s, double* scond, double* amax,
                     dcomplex* work, integer *info );

void FC_FUNC(sgesv,SGESV)( const integer* n, const integer* nrhs, float* a, const integer* lda,
                   integer* ipiv, float* b, const integer* ldb,
                   integer *info );

void FC_FUNC(dgesv,DGESV)( const integer* n, const integer* nrhs, double* a, const integer* lda,
                   integer* ipiv, double* b, const integer* ldb,
                   integer *info );

void FC_FUNC(cgesv,CGESV)( const integer* n, const integer* nrhs, scomplex* a, const integer* lda,
                   integer* ipiv, scomplex* b, const integer* ldb,
                   integer *info );

void FC_FUNC(zgesv,ZGESV)( const integer* n, const integer* nrhs, dcomplex* a, const integer* lda,
                   integer* ipiv, dcomplex* b, const integer* ldb,
                   integer *info );

void FC_FUNC(dsgesv,DSGESV)( const integer* n, const integer* nrhs, double* a, const integer* lda,
                    integer* ipiv, double* b, const integer* ldb, double* x,
                    const integer* ldx, double* work, float* swork,
                    integer* iter, integer *info );

void FC_FUNC(zcgesv,ZCGESV)( const integer* n, const integer* nrhs, dcomplex* a, const integer* lda,
                    integer* ipiv, dcomplex* b, const integer* ldb, dcomplex* x,
                    const integer* ldx, dcomplex* work, scomplex* swork, double* rwork,
                    integer* iter, integer *info );

void FC_FUNC(sgesvx,SGESVX)( char* fact, const char* trans, const integer* n, const integer* nrhs,
                    float* a, const integer* lda, float* af, const integer* ldaf,
                    integer* ipiv, char* equed, float* r, float* c,
                    float* b, const integer* ldb, float* x, const integer* ldx,
                    float* rcond, float* ferr, float* berr, float* work,
                    integer* iwork, integer *info );

void FC_FUNC(dgesvx,DGESVX)( char* fact, const char* trans, const integer* n, const integer* nrhs,
                    double* a, const integer* lda, double* af, const integer* ldaf,
                    integer* ipiv, char* equed, double* r, double* c,
                    double* b, const integer* ldb, double* x, const integer* ldx,
                    double* rcond, double* ferr, double* berr, double* work,
                    integer* iwork, integer *info );

void FC_FUNC(cgesvx,CGESVX)( char* fact, const char* trans, const integer* n, const integer* nrhs,
                    scomplex* a, const integer* lda, scomplex* af, const integer* ldaf,
                    integer* ipiv, char* equed, float* r, float* c,
                    scomplex* b, const integer* ldb, scomplex* x, const integer* ldx,
                    float* rcond, float* ferr, float* berr, scomplex* work,
                    float* rwork, integer *info );

void FC_FUNC(zgesvx,ZGESVX)( char* fact, const char* trans, const integer* n, const integer* nrhs,
                    dcomplex* a, const integer* lda, dcomplex* af, const integer* ldaf,
                    integer* ipiv, char* equed, double* r, double* c,
                    dcomplex* b, const integer* ldb, dcomplex* x, const integer* ldx,
                    double* rcond, double* ferr, double* berr, dcomplex* work,
                    double* rwork, integer *info );

void FC_FUNC(sgesvxx,SGESVXX)( char* fact, const char* trans, const integer* n, const integer* nrhs,
                     float* a, const integer* lda, float* af, const integer* ldaf,
                     integer* ipiv, char* equed, float* r, float* c,
                     float* b, const integer* ldb, float* x, const integer* ldx,
                     float* rcond, float* rpvgrw, float* berr,
                     const integer* n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, const integer* nparams, float* params,
                     float* work, integer* iwork, integer *info );

void FC_FUNC(dgesvxx,DGESVXX)( char* fact, const char* trans, const integer* n, const integer* nrhs,
                     double* a, const integer* lda, double* af, const integer* ldaf,
                     integer* ipiv, char* equed, double* r, double* c,
                     double* b, const integer* ldb, double* x, const integer* ldx,
                     double* rcond, double* rpvgrw, double* berr,
                     const integer* n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, const integer* nparams, double* params,
                     double* work, integer* iwork, integer *info );

void FC_FUNC(cgesvxx,CGESVXX)( char* fact, const char* trans, const integer* n, const integer* nrhs,
                     scomplex* a, const integer* lda, scomplex* af, const integer* ldaf,
                     integer* ipiv, char* equed, float* r, float* c,
                     scomplex* b, const integer* ldb, scomplex* x, const integer* ldx,
                     float* rcond, float* rpvgrw, float* berr,
                     const integer* n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, const integer* nparams, float* params,
                     scomplex* work, float* rwork, integer *info );

void FC_FUNC(zgesvxx,ZGESVXX)( char* fact, const char* trans, const integer* n, const integer* nrhs,
                     dcomplex* a, const integer* lda, dcomplex* af, const integer* ldaf,
                     integer* ipiv, char* equed, double* r, double* c,
                     dcomplex* b, const integer* ldb, dcomplex* x, const integer* ldx,
                     double* rcond, double* rpvgrw, double* berr,
                     const integer* n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, const integer* nparams, double* params,
                     dcomplex* work, double* rwork, integer *info );

void FC_FUNC(sgbsv,SGBSV)( const integer* n, const integer* kl, const integer* ku,
                   const integer* nrhs, float* ab, const integer* ldab,
                   integer* ipiv, float* b, const integer* ldb,
                   integer *info );

void FC_FUNC(dgbsv,DGBSV)( const integer* n, const integer* kl, const integer* ku,
                   const integer* nrhs, double* ab, const integer* ldab,
                   integer* ipiv, double* b, const integer* ldb,
                   integer *info );

void FC_FUNC(cgbsv,CGBSV)( const integer* n, const integer* kl, const integer* ku,
                   const integer* nrhs, scomplex* ab, const integer* ldab,
                   integer* ipiv, scomplex* b, const integer* ldb,
                   integer *info );

void FC_FUNC(zgbsv,ZGBSV)( const integer* n, const integer* kl, const integer* ku,
                   const integer* nrhs, dcomplex* ab, const integer* ldab,
                   integer* ipiv, dcomplex* b, const integer* ldb,
                   integer *info );

void FC_FUNC(sgbsvx,SGBSVX)( char* fact, const char* trans, const integer* n, const integer* kl,
                    const integer* ku, const integer* nrhs, float* ab,
                    const integer* ldab, float* afb, const integer* ldafb,
                    integer* ipiv, char* equed, float* r, float* c,
                    float* b, const integer* ldb, float* x, const integer* ldx,
                    float* rcond, float* ferr, float* berr, float* work,
                    integer* iwork, integer *info );

void FC_FUNC(dgbsvx,DGBSVX)( char* fact, const char* trans, const integer* n, const integer* kl,
                    const integer* ku, const integer* nrhs, double* ab,
                    const integer* ldab, double* afb, const integer* ldafb,
                    integer* ipiv, char* equed, double* r, double* c,
                    double* b, const integer* ldb, double* x, const integer* ldx,
                    double* rcond, double* ferr, double* berr, double* work,
                    integer* iwork, integer *info );

void FC_FUNC(cgbsvx,CGBSVX)( char* fact, const char* trans, const integer* n, const integer* kl,
                    const integer* ku, const integer* nrhs, scomplex* ab,
                    const integer* ldab, scomplex* afb, const integer* ldafb,
                    integer* ipiv, char* equed, float* r, float* c,
                    scomplex* b, const integer* ldb, scomplex* x, const integer* ldx,
                    float* rcond, float* ferr, float* berr, scomplex* work,
                    float* rwork, integer *info );

void FC_FUNC(zgbsvx,ZGBSVX)( char* fact, const char* trans, const integer* n, const integer* kl,
                    const integer* ku, const integer* nrhs, dcomplex* ab,
                    const integer* ldab, dcomplex* afb, const integer* ldafb,
                    integer* ipiv, char* equed, double* r, double* c,
                    dcomplex* b, const integer* ldb, dcomplex* x, const integer* ldx,
                    double* rcond, double* ferr, double* berr, dcomplex* work,
                    double* rwork, integer *info );

void FC_FUNC(sgbsvxx,SGBSVXX)( char* fact, const char* trans, const integer* n, const integer* kl,
                     const integer* ku, const integer* nrhs, float* ab,
                     const integer* ldab, float* afb, const integer* ldafb,
                     integer* ipiv, char* equed, float* r, float* c,
                     float* b, const integer* ldb, float* x, const integer* ldx,
                     float* rcond, float* rpvgrw, float* berr,
                     const integer* n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, const integer* nparams, float* params,
                     float* work, integer* iwork, integer *info );

void FC_FUNC(dgbsvxx,DGBSVXX)( char* fact, const char* trans, const integer* n, const integer* kl,
                     const integer* ku, const integer* nrhs, double* ab,
                     const integer* ldab, double* afb, const integer* ldafb,
                     integer* ipiv, char* equed, double* r, double* c,
                     double* b, const integer* ldb, double* x, const integer* ldx,
                     double* rcond, double* rpvgrw, double* berr,
                     const integer* n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, const integer* nparams, double* params,
                     double* work, integer* iwork, integer *info );

void FC_FUNC(cgbsvxx,CGBSVXX)( char* fact, const char* trans, const integer* n, const integer* kl,
                     const integer* ku, const integer* nrhs, scomplex* ab,
                     const integer* ldab, scomplex* afb, const integer* ldafb,
                     integer* ipiv, char* equed, float* r, float* c,
                     scomplex* b, const integer* ldb, scomplex* x, const integer* ldx,
                     float* rcond, float* rpvgrw, float* berr,
                     const integer* n_err_bnds, float* err_bnds_norm,
                     float* err_bnds_comp, const integer* nparams, float* params,
                     scomplex* work, float* rwork, integer *info );

void FC_FUNC(zgbsvxx,ZGBSVXX)( char* fact, const char* trans, const integer* n, const integer* kl,
                     const integer* ku, const integer* nrhs, dcomplex* ab,
                     const integer* ldab, dcomplex* afb, const integer* ldafb,
                     integer* ipiv, char* equed, double* r, double* c,
                     dcomplex* b, const integer* ldb, dcomplex* x, const integer* ldx,
                     double* rcond, double* rpvgrw, double* berr,
                     const integer* n_err_bnds, double* err_bnds_norm,
                     double* err_bnds_comp, const integer* nparams, double* params,
                     dcomplex* work, double* rwork, integer *info );

void FC_FUNC(sgtsv,SGTSV)( const integer* n, const integer* nrhs, float* dl, float* d,
                   float* du, float* b, const integer* ldb, integer *info );

void FC_FUNC(dgtsv,DGTSV)( const integer* n, const integer* nrhs, double* dl, double* d,
                   double* du, double* b, const integer* ldb, integer *info );

void FC_FUNC(cgtsv,CGTSV)( const integer* n, const integer* nrhs, scomplex* dl, scomplex* d,
                    scomplex* du, scomplex* b, const integer* ldb, integer *info );

void FC_FUNC(zgtsv,ZGTSV)( const integer* n, const integer* nrhs, dcomplex* dl, dcomplex* d,
                    dcomplex* du, dcomplex* b, const integer* ldb, integer *info );

void FC_FUNC(sgtsvx,SGTSVX)( char* fact, const char* trans, const integer* n, const integer* nrhs,
                    const float* dl, const float* d, const float* du,
                    float* dlf, float* df, float* duf, float* du2,
                    integer* ipiv, const float* b, const integer* ldb,
                    float* x, const integer* ldx, float* rcond, float* ferr,
                    float* berr, float* work, integer* iwork,
                    integer *info );

void FC_FUNC(dgtsvx,DGTSVX)( char* fact, const char* trans, const integer* n, const integer* nrhs,
                    const double* dl, const double* d, const double* du,
                    double* dlf, double* df, double* duf, double* du2,
                    integer* ipiv, const double* b, const integer* ldb,
                    double* x, const integer* ldx, double* rcond, double* ferr,
                    double* berr, double* work, integer* iwork,
                    integer *info );

void FC_FUNC(cgtsvx,CGTSVX)( char* fact, const char* trans, const integer* n, const integer* nrhs,
                    const scomplex* dl, const scomplex* d, const scomplex* du,
                    scomplex* dlf, scomplex* df, scomplex* duf, scomplex* du2,
                    integer* ipiv, const scomplex* b, const integer* ldb,
                    scomplex* x, const integer* ldx, float* rcond, float* ferr,
                    float* berr, scomplex* work, float* rwork,
                    integer *info );

void FC_FUNC(zgtsvx,ZGTSVX)( char* fact, const char* trans, const integer* n, const integer* nrhs,
                    const dcomplex* dl, const dcomplex* d, const dcomplex* du,
                    dcomplex* dlf, dcomplex* df, dcomplex* duf, dcomplex* du2,
                    integer* ipiv, const dcomplex* b, const integer* ldb,
                    dcomplex* x, const integer* ldx, double* rcond, double* ferr,
                    double* berr, dcomplex* work, double* rwork,
                    integer *info );

void FC_FUNC(sposv,SPOSV)( const char* uplo, const integer* n, const integer* nrhs, float* a,
                   const integer* lda, float* b, const integer* ldb,
                   integer *info );

void FC_FUNC(dposv,DPOSV)( const char* uplo, const integer* n, const integer* nrhs, double* a,
                   const integer* lda, double* b, const integer* ldb,
                   integer *info );

void FC_FUNC(cposv,CPOSV)( const char* uplo, const integer* n, const integer* nrhs, scomplex* a,
                   const integer* lda, scomplex* b, const integer* ldb,
                   integer *info );

void FC_FUNC(zposv,ZPOSV)( const char* uplo, const integer* n, const integer* nrhs, dcomplex* a,
                   const integer* lda, dcomplex* b, const integer* ldb,
                   integer *info );

void FC_FUNC(dsposv,DSPOSV)( const char* uplo, const integer* n, const integer* nrhs, double* a,
                    const integer* lda, double* b, const integer* ldb, double* x,
                    const integer* ldx, double* work, float* swork,
                    integer* iter, integer *info );

void FC_FUNC(zcposv,ZCPOSV)( const char* uplo, const integer* n, const integer* nrhs, dcomplex* a,
                    const integer* lda, dcomplex* b, const integer* ldb, dcomplex* x,
                    const integer* ldx, dcomplex* work, scomplex* swork, double* rwork,
                    integer* iter, integer *info );

void FC_FUNC(sposvx,SPOSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    float* a, const integer* lda, float* af, const integer* ldaf,
                    char* equed, float* s, float* b, const integer* ldb,
                    float* x, const integer* ldx, float* rcond, float* ferr,
                    float* berr, float* work, integer* iwork,
                    integer *info );

void FC_FUNC(dposvx,DPOSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    double* a, const integer* lda, double* af, const integer* ldaf,
                    char* equed, double* s, double* b, const integer* ldb,
                    double* x, const integer* ldx, double* rcond, double* ferr,
                    double* berr, double* work, integer* iwork,
                    integer *info );

void FC_FUNC(cposvx,CPOSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    scomplex* a, const integer* lda, scomplex* af, const integer* ldaf,
                    char* equed, float* s, scomplex* b, const integer* ldb,
                    scomplex* x, const integer* ldx, float* rcond, float* ferr,
                    float* berr, scomplex* work, float* rwork,
                    integer *info );

void FC_FUNC(zposvx,ZPOSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    dcomplex* a, const integer* lda, dcomplex* af, const integer* ldaf,
                    char* equed, double* s, dcomplex* b, const integer* ldb,
                    dcomplex* x, const integer* ldx, double* rcond, double* ferr,
                    double* berr, dcomplex* work, double* rwork,
                    integer *info );

void FC_FUNC(sposvxx,SPOSVXX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                     float* a, const integer* lda, float* af, const integer* ldaf,
                     char* equed, float* s, float* b, const integer* ldb,
                     float* x, const integer* ldx, float* rcond, float* rpvgrw,
                     float* berr, const integer* n_err_bnds,
                     float* err_bnds_norm, float* err_bnds_comp,
                     const integer* nparams, float* params, float* work,
                     integer* iwork, integer *info );

void FC_FUNC(dposvxx,DPOSVXX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                     double* a, const integer* lda, double* af, const integer* ldaf,
                     char* equed, double* s, double* b, const integer* ldb,
                     double* x, const integer* ldx, double* rcond, double* rpvgrw,
                     double* berr, const integer* n_err_bnds,
                     double* err_bnds_norm, double* err_bnds_comp,
                     const integer* nparams, double* params, double* work,
                     integer* iwork, integer *info );

void FC_FUNC(cposvxx,CPOSVXX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    scomplex* a, const integer* lda, scomplex* af, const integer* ldaf,
                     char* equed, float* s, scomplex* b, const integer* ldb,
                     scomplex* x, const integer* ldx, float* rcond, float* rpvgrw,
                     float* berr, const integer* n_err_bnds,
                     float* err_bnds_norm, float* err_bnds_comp,
                     const integer* nparams, float* params, scomplex* work,
                     float* rwork, integer *info );

void FC_FUNC(zposvxx,ZPOSVXX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                     dcomplex* a, const integer* lda, dcomplex* af, const integer* ldaf,
                     char* equed, double* s, dcomplex* b, const integer* ldb,
                     dcomplex* x, const integer* ldx, double* rcond, double* rpvgrw,
                     double* berr, const integer* n_err_bnds,
                     double* err_bnds_norm, double* err_bnds_comp,
                     const integer* nparams, double* params, dcomplex* work,
                     double* rwork, integer *info );

void FC_FUNC(sppsv,SPPSV)( const char* uplo, const integer* n, const integer* nrhs, float* ap,
                   float* b, const integer* ldb, integer *info );

void FC_FUNC(dppsv,DPPSV)( const char* uplo, const integer* n, const integer* nrhs, double* ap,
                   double* b, const integer* ldb, integer *info );

void FC_FUNC(cppsv,CPPSV)( const char* uplo, const integer* n, const integer* nrhs, scomplex* ap,
                   scomplex* b, const integer* ldb, integer *info );

void FC_FUNC(zppsv,ZPPSV)( const char* uplo, const integer* n, const integer* nrhs, dcomplex* ap,
                   dcomplex* b, const integer* ldb, integer *info );

void FC_FUNC(sppsvx,SPPSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    float* ap, float* afp, char* equed, float* s, float* b,
                    const integer* ldb, float* x, const integer* ldx, float* rcond,
                    float* ferr, float* berr, float* work, integer* iwork,
                    integer *info );

void FC_FUNC(dppsvx,DPPSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    double* ap, double* afp, char* equed, double* s, double* b,
                    const integer* ldb, double* x, const integer* ldx, double* rcond,
                    double* ferr, double* berr, double* work, integer* iwork,
                    integer *info );

void FC_FUNC(cppsvx,CPPSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    scomplex* ap, scomplex* afp, char* equed, float* s, scomplex* b,
                    const integer* ldb, scomplex* x, const integer* ldx, float* rcond,
                    float* ferr, float* berr, scomplex* work, float* rwork,
                    integer *info );

void FC_FUNC(zppsvx,ZPPSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    dcomplex* ap, dcomplex* afp, char* equed, double* s, dcomplex* b,
                    const integer* ldb, dcomplex* x, const integer* ldx, double* rcond,
                    double* ferr, double* berr, dcomplex* work, double* rwork,
                    integer *info );

void FC_FUNC(spbsv,SPBSV)( const char* uplo, const integer* n, const integer* kd, const integer* nrhs,
                   float* ab, const integer* ldab, float* b, const integer* ldb,
                   integer *info );

void FC_FUNC(dpbsv,DPBSV)( const char* uplo, const integer* n, const integer* kd, const integer* nrhs,
                   double* ab, const integer* ldab, double* b, const integer* ldb,
                   integer *info );

void FC_FUNC(cpbsv,CPBSV)( const char* uplo, const integer* n, const integer* kd, const integer* nrhs,
                   scomplex* ab, const integer* ldab, scomplex* b, const integer* ldb,
                   integer *info );

void FC_FUNC(zpbsv,ZPBSV)( const char* uplo, const integer* n, const integer* kd, const integer* nrhs,
                   dcomplex* ab, const integer* ldab, dcomplex* b, const integer* ldb,
                   integer *info );

void FC_FUNC(spbsvx,SPBSVX)( char* fact, const char* uplo, const integer* n, const integer* kd,
                    const integer* nrhs, float* ab, const integer* ldab, float* afb,
                    const integer* ldafb, char* equed, float* s, float* b,
                    const integer* ldb, float* x, const integer* ldx, float* rcond,
                    float* ferr, float* berr, float* work, integer* iwork,
                    integer *info );

void FC_FUNC(dpbsvx,DPBSVX)( char* fact, const char* uplo, const integer* n, const integer* kd,
                    const integer* nrhs, double* ab, const integer* ldab, double* afb,
                    const integer* ldafb, char* equed, double* s, double* b,
                    const integer* ldb, double* x, const integer* ldx, double* rcond,
                    double* ferr, double* berr, double* work, integer* iwork,
                    integer *info );

void FC_FUNC(cpbsvx,CPBSVX)( char* fact, const char* uplo, const integer* n, const integer* kd,
                    const integer* nrhs, scomplex* ab, const integer* ldab, scomplex* afb,
                    const integer* ldafb, char* equed, float* s, scomplex* b,
                    const integer* ldb, scomplex* x, const integer* ldx, float* rcond,
                    float* ferr, float* berr, scomplex* work, float* rwork,
                    integer *info );

void FC_FUNC(zpbsvx,ZPBSVX)( char* fact, const char* uplo, const integer* n, const integer* kd,
                    const integer* nrhs, dcomplex* ab, const integer* ldab, dcomplex* afb,
                    const integer* ldafb, char* equed, double* s, dcomplex* b,
                    const integer* ldb, dcomplex* x, const integer* ldx, double* rcond,
                    double* ferr, double* berr, dcomplex* work, double* rwork,
                    integer *info );

void FC_FUNC(sptsv,SPTSV)( const integer* n, const integer* nrhs, float* d, float* e,
                    float* b, const integer* ldb, integer *info );

void FC_FUNC(dptsv,DPTSV)( const integer* n, const integer* nrhs, double* d, double* e,
                   double* b, const integer* ldb, integer *info );

void FC_FUNC(cptsv,CPTSV)( const integer* n, const integer* nrhs, float* d, scomplex* e,
                   scomplex* b, const integer* ldb, integer *info );

void FC_FUNC(zptsv,ZPTSV)( const integer* n, const integer* nrhs, double* d, dcomplex* e,
                   dcomplex* b, const integer* ldb, integer *info );

void FC_FUNC(sptsvx,SPTSVX)( char* fact, const integer* n, const integer* nrhs,
                    const float* d, const float* e, float* df, float* ef,
                    const float* b, const integer* ldb, float* x,
                    const integer* ldx, float* rcond, float* ferr, float* berr,
                    float* work, integer *info );

void FC_FUNC(dptsvx,DPTSVX)( char* fact, const integer* n, const integer* nrhs,
                    const double* d, const double* e, double* df, double* ef,
                    const double* b, const integer* ldb, double* x,
                    const integer* ldx, double* rcond, double* ferr, double* berr,
                    double* work, integer *info );

void FC_FUNC(cptsvx,CPTSVX)( char* fact, const integer* n, const integer* nrhs,
                    const float* d, const scomplex* e, float* df, scomplex* ef,
                    const scomplex* b, const integer* ldb, scomplex* x,
                    const integer* ldx, float* rcond, float* ferr, float* berr,
                    scomplex* work, float* rwork, integer *info );

void FC_FUNC(zptsvx,ZPTSVX)( char* fact, const integer* n, const integer* nrhs,
                    const double* d, const dcomplex* e, double* df, dcomplex* ef,
                    const dcomplex* b, const integer* ldb, dcomplex* x,
                    const integer* ldx, double* rcond, double* ferr, double* berr,
                    dcomplex* work, double* rwork, integer *info );

void FC_FUNC(ssysv,SSYSV)( const char* uplo, const integer* n, const integer* nrhs, float* a,
                   const integer* lda, integer* ipiv, float* b,
                   const integer* ldb, float* work, integer* lwork,
                   integer *info );

void FC_FUNC(dsysv,DSYSV)( const char* uplo, const integer* n, const integer* nrhs, double* a,
                   const integer* lda, integer* ipiv, double* b,
                   const integer* ldb, double* work, integer* lwork,
                   integer *info );

void FC_FUNC(csysv,CSYSV)( const char* uplo, const integer* n, const integer* nrhs, scomplex* a,
                   const integer* lda, integer* ipiv, scomplex* b,
                   const integer* ldb, scomplex* work, integer* lwork,
                   integer *info );

void FC_FUNC(zsysv,ZSYSV)( const char* uplo, const integer* n, const integer* nrhs, dcomplex* a,
                   const integer* lda, integer* ipiv, dcomplex* b,
                   const integer* ldb, dcomplex* work, integer* lwork,
                   integer *info );

void FC_FUNC(chesv,CHESV)( const char* uplo, const integer* n, const integer* nrhs, scomplex* a,
                   const integer* lda, integer* ipiv, scomplex* b,
                   const integer* ldb, scomplex* work, integer* lwork,
                   integer *info );

void FC_FUNC(zhesv,ZHESV)( const char* uplo, const integer* n, const integer* nrhs, dcomplex* a,
                   const integer* lda, integer* ipiv, dcomplex* b,
                   const integer* ldb, dcomplex* work, integer* lwork,
                   integer *info );

void FC_FUNC(ssysvx,SSYSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    const float* a, const integer* lda, float* af,
                    const integer* ldaf, integer* ipiv, const float* b,
                    const integer* ldb, float* x, const integer* ldx, float* rcond,
                    float* ferr, float* berr, float* work, integer* lwork,
                    integer* iwork, integer *info );

void FC_FUNC(dsysvx,DSYSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    const double* a, const integer* lda, double* af,
                    const integer* ldaf, integer* ipiv, const double* b,
                    const integer* ldb, double* x, const integer* ldx, double* rcond,
                    double* ferr, double* berr, double* work, integer* lwork,
                    integer* iwork, integer *info );

void FC_FUNC(csysvx,CSYSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* a, const integer* lda, scomplex* af,
                    const integer* ldaf, integer* ipiv, const scomplex* b,
                    const integer* ldb, scomplex* x, const integer* ldx, float* rcond,
                    float* ferr, float* berr, scomplex* work, integer* lwork,
                    float* rwork, integer *info );

void FC_FUNC(zsysvx,ZSYSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* a, const integer* lda, dcomplex* af,
                    const integer* ldaf, integer* ipiv, const dcomplex* b,
                    const integer* ldb, dcomplex* x, const integer* ldx, double* rcond,
                    double* ferr, double* berr, dcomplex* work, integer* lwork,
                    double* rwork, integer *info );

void FC_FUNC(chesvx,CHESVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* a, const integer* lda, scomplex* af,
                    const integer* ldaf, integer* ipiv, const scomplex* b,
                    const integer* ldb, scomplex* x, const integer* ldx, float* rcond,
                    float* ferr, float* berr, scomplex* work, integer* lwork,
                    float* rwork, integer *info );

void FC_FUNC(zhesvx,ZHESVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* a, const integer* lda, dcomplex* af,
                    const integer* ldaf, integer* ipiv, const dcomplex* b,
                    const integer* ldb, dcomplex* x, const integer* ldx, double* rcond,
                    double* ferr, double* berr, dcomplex* work, integer* lwork,
                    double* rwork, integer *info );

void FC_FUNC(ssysvxx,SSYSVXX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    float* a, const integer* lda, float* af, const integer* ldaf,
                     integer* ipiv, char* equed, float* s, float* b,
                     const integer* ldb, float* x, const integer* ldx, float* rcond,
                     float* rpvgrw, float* berr, const integer* n_err_bnds,
                     float* err_bnds_norm, float* err_bnds_comp,
                     const integer* nparams, float* params, float* work,
                     integer* iwork, integer *info );

void FC_FUNC(dsysvxx,DSYSVXX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                     double* a, const integer* lda, double* af, const integer* ldaf,
                     integer* ipiv, char* equed, double* s, double* b,
                     const integer* ldb, double* x, const integer* ldx, double* rcond,
                     double* rpvgrw, double* berr, const integer* n_err_bnds,
                     double* err_bnds_norm, double* err_bnds_comp,
                     const integer* nparams, double* params, double* work,
                     integer* iwork, integer *info );

void FC_FUNC(csysvxx,CSYSVXX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                     scomplex* a, const integer* lda, scomplex* af, const integer* ldaf,
                     integer* ipiv, char* equed, float* s, scomplex* b,
                     const integer* ldb, scomplex* x, const integer* ldx, float* rcond,
                     float* rpvgrw, float* berr, const integer* n_err_bnds,
                     float* err_bnds_norm, float* err_bnds_comp,
                     const integer* nparams, float* params, scomplex* work,
                     float* rwork, integer *info );

void FC_FUNC(zsysvxx,ZSYSVXX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                     dcomplex* a, const integer* lda, dcomplex* af, const integer* ldaf,
                     integer* ipiv, char* equed, double* s, dcomplex* b,
                     const integer* ldb, dcomplex* x, const integer* ldx, double* rcond,
                     double* rpvgrw, double* berr, const integer* n_err_bnds,
                     double* err_bnds_norm, double* err_bnds_comp,
                     const integer* nparams, double* params, dcomplex* work,
                     double* rwork, integer *info );

void FC_FUNC(chesvxx,CHESVXX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                     scomplex* a, const integer* lda, scomplex* af, const integer* ldaf,
                     integer* ipiv, char* equed, float* s, scomplex* b,
                     const integer* ldb, scomplex* x, const integer* ldx, float* rcond,
                     float* rpvgrw, float* berr, const integer* n_err_bnds,
                     float* err_bnds_norm, float* err_bnds_comp,
                     const integer* nparams, float* params, scomplex* work,
                     float* rwork, integer *info );

void FC_FUNC(zhesvxx,ZHESVXX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                     dcomplex* a, const integer* lda, dcomplex* af, const integer* ldaf,
                     integer* ipiv, char* equed, double* s, dcomplex* b,
                     const integer* ldb, dcomplex* x, const integer* ldx, double* rcond,
                     double* rpvgrw, double* berr, const integer* n_err_bnds,
                     double* err_bnds_norm, double* err_bnds_comp,
                     const integer* nparams, double* params, dcomplex* work,
                     double* rwork, integer *info );

void FC_FUNC(sspsv,SSPSV)( const char* uplo, const integer* n, const integer* nrhs, float* ap,
                   integer* ipiv, float* b, const integer* ldb,
                   integer *info );

void FC_FUNC(dspsv,DSPSV)( const char* uplo, const integer* n, const integer* nrhs, double* ap,
                   integer* ipiv, double* b, const integer* ldb,
                   integer *info );

void FC_FUNC(cspsv,CSPSV)( const char* uplo, const integer* n, const integer* nrhs, scomplex* ap,
                   integer* ipiv, scomplex* b, const integer* ldb,
                   integer *info );

void FC_FUNC(zspsv,ZSPSV)( const char* uplo, const integer* n, const integer* nrhs, dcomplex* ap,
                   integer* ipiv, dcomplex* b, const integer* ldb,
                   integer *info );

void FC_FUNC(chpsv,CHPSV)( const char* uplo, const integer* n, const integer* nrhs, scomplex* ap,
                   integer* ipiv, scomplex* b, const integer* ldb,
                   integer *info );

void FC_FUNC(zhpsv,ZHPSV)( const char* uplo, const integer* n, const integer* nrhs, dcomplex* ap,
                   integer* ipiv, dcomplex* b, const integer* ldb,
                   integer *info );

void FC_FUNC(sspsvx,SSPSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    const float* ap, float* afp, integer* ipiv,
                    const float* b, const integer* ldb, float* x,
                    const integer* ldx, float* rcond, float* ferr, float* berr,
                    float* work, integer* iwork, integer *info );

void FC_FUNC(dspsvx,DSPSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    const double* ap, double* afp, integer* ipiv,
                    const double* b, const integer* ldb, double* x,
                    const integer* ldx, double* rcond, double* ferr, double* berr,
                    double* work, integer* iwork, integer *info );

void FC_FUNC(cspsvx,CSPSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* ap, scomplex* afp, integer* ipiv,
                    const scomplex* b, const integer* ldb, scomplex* x,
                    const integer* ldx, float* rcond, float* ferr, float* berr,
                    scomplex* work, float* rwork, integer *info );

void FC_FUNC(zspsvx,ZSPSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* ap, dcomplex* afp, integer* ipiv,
                    const dcomplex* b, const integer* ldb, dcomplex* x,
                    const integer* ldx, double* rcond, double* ferr, double* berr,
                    dcomplex* work, double* rwork, integer *info );

void FC_FUNC(chpsvx,CHPSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    const scomplex* ap, scomplex* afp, integer* ipiv,
                    const scomplex* b, const integer* ldb, scomplex* x,
                    const integer* ldx, float* rcond, float* ferr, float* berr,
                    scomplex* work, float* rwork, integer *info );

void FC_FUNC(zhpsvx,ZHPSVX)( char* fact, const char* uplo, const integer* n, const integer* nrhs,
                    const dcomplex* ap, dcomplex* afp, integer* ipiv,
                    const dcomplex* b, const integer* ldb, dcomplex* x,
                    const integer* ldx, double* rcond, double* ferr, double* berr,
                    dcomplex* work, double* rwork, integer *info );

void FC_FUNC(sgeqrf,SGEQRF)( const integer* m, const integer* n, float* a, const integer* lda,
                    float* tau, float* work, integer* lwork,
                    integer *info );

void FC_FUNC(dgeqrf,DGEQRF)( const integer* m, const integer* n, double* a, const integer* lda,
                    double* tau, double* work, integer* lwork,
                    integer *info );

void FC_FUNC(cgeqrf,CGEQRF)( const integer* m, const integer* n, scomplex* a, const integer* lda,
                    scomplex* tau, scomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(zgeqrf,ZGEQRF)( const integer* m, const integer* n, dcomplex* a, const integer* lda,
                    dcomplex* tau, dcomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(sgeqpf,SGEQPF)( const integer* m, const integer* n, float* a, const integer* lda,
                    integer* jpvt, float* tau, float* work,
                    integer *info );

void FC_FUNC(dgeqpf,DGEQPF)( const integer* m, const integer* n, double* a, const integer* lda,
                    integer* jpvt, double* tau, double* work,
                    integer *info );

void FC_FUNC(cgeqpf,CGEQPF)( const integer* m, const integer* n, scomplex* a, const integer* lda,
                    integer* jpvt, scomplex* tau, scomplex* work, float* rwork,
                    integer *info );

void FC_FUNC(zgeqpf,ZGEQPF)( const integer* m, const integer* n, dcomplex* a, const integer* lda,
                    integer* jpvt, dcomplex* tau, dcomplex* work, double* rwork,
                    integer *info );

void FC_FUNC(sgeqp3,SGEQP3)( const integer* m, const integer* n, float* a, const integer* lda,
                    integer* jpvt, float* tau, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dgeqp3,DGEQP3)( const integer* m, const integer* n, double* a, const integer* lda,
                    integer* jpvt, double* tau, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(cgeqp3,CGEQP3)( const integer* m, const integer* n, scomplex* a, const integer* lda,
                    integer* jpvt, scomplex* tau, scomplex* work,
                    integer* lwork, float* rwork, integer *info );

void FC_FUNC(zgeqp3,ZGEQP3)( const integer* m, const integer* n, dcomplex* a, const integer* lda,
                    integer* jpvt, dcomplex* tau, dcomplex* work,
                    integer* lwork, double* rwork, integer *info );

void FC_FUNC(sorgqr,SORGQR)( const integer* m, const integer* n, const integer* k, float* a,
                    const integer* lda, const float* tau, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dorgqr,DORGQR)( const integer* m, const integer* n, const integer* k, double* a,
                    const integer* lda, const double* tau, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(cungqr,CUNGQR)( const integer* m, const integer* n, const integer* k, scomplex* a,
                    const integer* lda, const scomplex* tau, scomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(zungqr,ZUNGQR)( const integer* m, const integer* n, const integer* k, dcomplex* a,
                    const integer* lda, const dcomplex* tau, dcomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(sormqr,SORMQR)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, const float* a, const integer* lda,
                    const float* tau, float* c, const integer* ldc, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dormqr,DORMQR)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, const double* a, const integer* lda,
                    const double* tau, double* c, const integer* ldc, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(cunmqr,CUNMQR)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, const scomplex* a, const integer* lda,
                    const scomplex* tau, scomplex* c, const integer* ldc, scomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(zunmqr,ZUNMQR)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, const dcomplex* a, const integer* lda,
                    const dcomplex* tau, dcomplex* c, const integer* ldc, dcomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(sgelqf,SGELQF)( const integer* m, const integer* n, float* a, const integer* lda,
                    float* tau, float* work, integer* lwork,
                    integer *info );

void FC_FUNC(dgelqf,DGELQF)( const integer* m, const integer* n, double* a, const integer* lda,
                    double* tau, double* work, integer* lwork,
                    integer *info );

void FC_FUNC(cgelqf,CGELQF)( const integer* m, const integer* n, scomplex* a, const integer* lda,
                    scomplex* tau, scomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(zgelqf,ZGELQF)( const integer* m, const integer* n, dcomplex* a, const integer* lda,
                    dcomplex* tau, dcomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(sorglq,SORGLQ)( const integer* m, const integer* n, const integer* k, float* a,
                    const integer* lda, const float* tau, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dorglq,DORGLQ)( const integer* m, const integer* n, const integer* k, double* a,
                    const integer* lda, const double* tau, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(cunglq,CUNGLQ)( const integer* m, const integer* n, const integer* k, scomplex* a,
                    const integer* lda, const scomplex* tau, scomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(zunglq,ZUNGLQ)( const integer* m, const integer* n, const integer* k, dcomplex* a,
                    const integer* lda, const dcomplex* tau, dcomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(sormlq,SORMLQ)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, const float* a, const integer* lda,
                    const float* tau, float* c, const integer* ldc, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dormlq,DORMLQ)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, const double* a, const integer* lda,
                    const double* tau, double* c, const integer* ldc, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(cunmlq,CUNMLQ)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, const scomplex* a, const integer* lda,
                    const scomplex* tau, scomplex* c, const integer* ldc, scomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(zunmlq,ZUNMLQ)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, const dcomplex* a, const integer* lda,
                    const dcomplex* tau, dcomplex* c, const integer* ldc, dcomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(sgeqlf,SGEQLF)( const integer* m, const integer* n, float* a, const integer* lda,
                    float* tau, float* work, integer* lwork,
                    integer *info );

void FC_FUNC(dgeqlf,DGEQLF)( const integer* m, const integer* n, double* a, const integer* lda,
                    double* tau, double* work, integer* lwork,
                    integer *info );

void FC_FUNC(cgeqlf,CGEQLF)( const integer* m, const integer* n, scomplex* a, const integer* lda,
                    scomplex* tau, scomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(zgeqlf,ZGEQLF)( const integer* m, const integer* n, dcomplex* a, const integer* lda,
                    dcomplex* tau, dcomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(sorgql,SORGQL)( const integer* m, const integer* n, const integer* k, float* a,
                    const integer* lda, const float* tau, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dorgql,DORGQL)( const integer* m, const integer* n, const integer* k, double* a,
                    const integer* lda, const double* tau, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(cungql,CUNGQL)( const integer* m, const integer* n, const integer* k, scomplex* a,
                    const integer* lda, const scomplex* tau, scomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(zungql,ZUNGQL)( const integer* m, const integer* n, const integer* k, dcomplex* a,
                    const integer* lda, const dcomplex* tau, dcomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(sormql,SORMQL)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, const float* a, const integer* lda,
                    const float* tau, float* c, const integer* ldc, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dormql,DORMQL)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, const double* a, const integer* lda,
                    const double* tau, double* c, const integer* ldc, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(cunmql,CUNMQL)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, const scomplex* a, const integer* lda,
                    const scomplex* tau, scomplex* c, const integer* ldc, scomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(zunmql,ZUNMQL)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, const dcomplex* a, const integer* lda,
                    const dcomplex* tau, dcomplex* c, const integer* ldc, dcomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(sgerqf,SGERQF)( const integer* m, const integer* n, float* a, const integer* lda,
                    float* tau, float* work, integer* lwork,
                    integer *info );

void FC_FUNC(dgerqf,DGERQF)( const integer* m, const integer* n, double* a, const integer* lda,
                    double* tau, double* work, integer* lwork,
                    integer *info );

void FC_FUNC(cgerqf,CGERQF)( const integer* m, const integer* n, scomplex* a, const integer* lda,
                    scomplex* tau, scomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(zgerqf,ZGERQF)( const integer* m, const integer* n, dcomplex* a, const integer* lda,
                    dcomplex* tau, dcomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(sorgrq,SORGRQ)( const integer* m, const integer* n, const integer* k, float* a,
                    const integer* lda, const float* tau, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dorgrq,DORGRQ)( const integer* m, const integer* n, const integer* k, double* a,
                    const integer* lda, const double* tau, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(cungrq,CUNGRQ)( const integer* m, const integer* n, const integer* k, scomplex* a,
                    const integer* lda, const scomplex* tau, scomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(zungrq,ZUNGRQ)( const integer* m, const integer* n, const integer* k, dcomplex* a,
                    const integer* lda, const dcomplex* tau, dcomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(sormrq,SORMRQ)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, const float* a, const integer* lda,
                    const float* tau, float* c, const integer* ldc, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dormrq,DORMRQ)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, const double* a, const integer* lda,
                    const double* tau, double* c, const integer* ldc, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(cunmrq,CUNMRQ)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, const scomplex* a, const integer* lda,
                    const scomplex* tau, scomplex* c, const integer* ldc, scomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(zunmrq,ZUNMRQ)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, const dcomplex* a, const integer* lda,
                    const dcomplex* tau, dcomplex* c, const integer* ldc, dcomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(stzrzf,STZRZF)( const integer* m, const integer* n, float* a, const integer* lda,
                    float* tau, float* work, integer* lwork,
                    integer *info );

void FC_FUNC(dtzrzf,DTZRZF)( const integer* m, const integer* n, double* a, const integer* lda,
                    double* tau, double* work, integer* lwork,
                    integer *info );

void FC_FUNC(ctzrzf,CTZRZF)( const integer* m, const integer* n, scomplex* a, const integer* lda,
                    scomplex* tau, scomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(ztzrzf,ZTZRZF)( const integer* m, const integer* n, dcomplex* a, const integer* lda,
                    dcomplex* tau, dcomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(sormrz,SORMRZ)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, integer* l, const float* a,
                    const integer* lda, const float* tau, float* c,
                    const integer* ldc, float* work, integer* lwork,
                    integer *info );

void FC_FUNC(dormrz,DORMRZ)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, integer* l, const double* a,
                    const integer* lda, const double* tau, double* c,
                    const integer* ldc, double* work, integer* lwork,
                    integer *info );

void FC_FUNC(cunmrz,CUNMRZ)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, integer* l, const scomplex* a,
                    const integer* lda, const scomplex* tau, scomplex* c,
                    const integer* ldc, scomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(zunmrz,ZUNMRZ)( char* side, const char* trans, const integer* m, const integer* n,
                    const integer* k, integer* l, const dcomplex* a,
                    const integer* lda, const dcomplex* tau, dcomplex* c,
                    const integer* ldc, dcomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(sggqrf,SGGQRF)( const integer* n, const integer* m, integer* p, float* a,
                    const integer* lda, float* taua, float* b, const integer* ldb,
                    float* taub, float* work, integer* lwork,
                    integer *info );

void FC_FUNC(dggqrf,DGGQRF)( const integer* n, const integer* m, integer* p, double* a,
                    const integer* lda, double* taua, double* b, const integer* ldb,
                    double* taub, double* work, integer* lwork,
                    integer *info );

void FC_FUNC(cggqrf,CGGQRF)( const integer* n, const integer* m, integer* p, scomplex* a,
                    const integer* lda, scomplex* taua, scomplex* b, const integer* ldb,
                    scomplex* taub, scomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(zggqrf,ZGGQRF)( const integer* n, const integer* m, integer* p, dcomplex* a,
                    const integer* lda, dcomplex* taua, dcomplex* b, const integer* ldb,
                    dcomplex* taub, dcomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(sggrqf,SGGRQF)( const integer* m, integer* p, const integer* n, float* a,
                    const integer* lda, float* taua, float* b, const integer* ldb,
                    float* taub, float* work, integer* lwork,
                    integer *info );

void FC_FUNC(dggrqf,DGGRQF)( const integer* m, integer* p, const integer* n, double* a,
                    const integer* lda, double* taua, double* b, const integer* ldb,
                    double* taub, double* work, integer* lwork,
                    integer *info );

void FC_FUNC(cggrqf,CGGRQF)( const integer* m, integer* p, const integer* n, scomplex* a,
                    const integer* lda, scomplex* taua, scomplex* b, const integer* ldb,
                    scomplex* taub, scomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(zggrqf,ZGGRQF)( const integer* m, integer* p, const integer* n, dcomplex* a,
                    const integer* lda, dcomplex* taua, dcomplex* b, const integer* ldb,
                    dcomplex* taub, dcomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(sgebrd,SGEBRD)( const integer* m, const integer* n, float* a, const integer* lda,
                    float* d, float* e, float* tauq, float* taup,
                    float* work, integer* lwork, integer *info );

void FC_FUNC(dgebrd,DGEBRD)( const integer* m, const integer* n, double* a, const integer* lda,
                    double* d, double* e, double* tauq, double* taup,
                    double* work, integer* lwork, integer *info );

void FC_FUNC(cgebrd,CGEBRD)( const integer* m, const integer* n, scomplex* a, const integer* lda,
                    float* d, float* e, scomplex* tauq, scomplex* taup,
                    scomplex* work, integer* lwork, integer *info );

void FC_FUNC(zgebrd,ZGEBRD)( const integer* m, const integer* n, dcomplex* a, const integer* lda,
                    double* d, double* e, dcomplex* tauq, dcomplex* taup,
                    dcomplex* work, integer* lwork, integer *info );

void FC_FUNC(sgbbrd,SGBBRD)( char* vect, const integer* m, const integer* n, const integer* ncc,
                    const integer* kl, const integer* ku, float* ab,
                    const integer* ldab, float* d, float* e, float* q,
                    const integer* ldq, float* pt, const integer* ldpt, float* c,
                    const integer* ldc, float* work, integer *info );

void FC_FUNC(dgbbrd,DGBBRD)( char* vect, const integer* m, const integer* n, const integer* ncc,
                    const integer* kl, const integer* ku, double* ab,
                    const integer* ldab, double* d, double* e, double* q,
                    const integer* ldq, double* pt, const integer* ldpt, double* c,
                    const integer* ldc, double* work, integer *info );

void FC_FUNC(cgbbrd,CGBBRD)( char* vect, const integer* m, const integer* n, const integer* ncc,
                    const integer* kl, const integer* ku, scomplex* ab,
                    const integer* ldab, float* d, float* e, scomplex* q,
                    const integer* ldq, scomplex* pt, const integer* ldpt, scomplex* c,
                    const integer* ldc, scomplex* work, float* rwork, integer *info );

void FC_FUNC(zgbbrd,ZGBBRD)( char* vect, const integer* m, const integer* n, const integer* ncc,
                    const integer* kl, const integer* ku, dcomplex* ab,
                    const integer* ldab, double* d, double* e, dcomplex* q,
                    const integer* ldq, dcomplex* pt, const integer* ldpt, dcomplex* c,
                    const integer* ldc, dcomplex* work, double* rwork, integer *info );

void FC_FUNC(sorgbr,SORGBR)( char* vect, const integer* m, const integer* n, const integer* k,
                    float* a, const integer* lda, const float* tau, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dorgbr,DORGBR)( char* vect, const integer* m, const integer* n, const integer* k,
                    double* a, const integer* lda, const double* tau, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(cungbr,CUNGBR)( char* vect, const integer* m, const integer* n, const integer* k,
                    scomplex* a, const integer* lda, const scomplex* tau, scomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(zungbr,ZUNGBR)( char* vect, const integer* m, const integer* n, const integer* k,
                    dcomplex* a, const integer* lda, const dcomplex* tau, dcomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(sormbr,SORMBR)( char* vect, char* side, const char* trans, const integer* m,
                    const integer* n, const integer* k, const float* a,
                    const integer* lda, const float* tau, float* c,
                    const integer* ldc, float* work, integer* lwork,
                    integer *info );

void FC_FUNC(dormbr,DORMBR)( char* vect, char* side, const char* trans, const integer* m,
                    const integer* n, const integer* k, const double* a,
                    const integer* lda, const double* tau, double* c,
                    const integer* ldc, double* work, integer* lwork,
                    integer *info );

void FC_FUNC(cunmbr,CUNMBR)( char* vect, char* side, const char* trans, const integer* m,
                    const integer* n, const integer* k, const scomplex* a,
                    const integer* lda, const scomplex* tau, scomplex* c,
                    const integer* ldc, scomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(zunmbr,ZUNMBR)( char* vect, char* side, const char* trans, const integer* m,
                    const integer* n, const integer* k, const dcomplex* a,
                    const integer* lda, const dcomplex* tau, dcomplex* c,
                    const integer* ldc, dcomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(sbdsqr,SBDSQR)( const char* uplo, const integer* n, const integer* ncvt,
                    const integer* nru, const integer* ncc, float* d, float* e,
                    float* vt, const integer* ldvt, float* u, const integer* ldu,
                    float* c, const integer* ldc, float* work,
                    integer *info );

void FC_FUNC(dbdsqr,DBDSQR)( const char* uplo, const integer* n, const integer* ncvt,
                    const integer* nru, const integer* ncc, double* d, double* e,
                    double* vt, const integer* ldvt, double* u, const integer* ldu,
                    double* c, const integer* ldc, double* work,
                    integer *info );

void FC_FUNC(cbdsqr,CBDSQR)( const char* uplo, const integer* n, const integer* ncvt,
                    const integer* nru, const integer* ncc, float* d, float* e,
                    scomplex* vt, const integer* ldvt, scomplex* u, const integer* ldu,
                    scomplex* c, const integer* ldc, float* rwork,
                    integer *info );

void FC_FUNC(zbdsqr,ZBDSQR)( const char* uplo, const integer* n, const integer* ncvt,
                    const integer* nru, const integer* ncc, double* d, double* e,
                    dcomplex* vt, const integer* ldvt, dcomplex* u, const integer* ldu,
                    dcomplex* c, const integer* ldc, double* rwork,
                    integer *info );

void FC_FUNC(sbdsdc,SBDSDC)( const char* uplo, char* compq, const integer* n, float* d,
                    float* e, float* u, const integer* ldu, float* vt,
                    const integer* ldvt, float* q, integer* iq, float* work,
                    integer* iwork, integer *info );

void FC_FUNC(dbdsdc,DBDSDC)( const char* uplo, char* compq, const integer* n, double* d,
                    double* e, double* u, const integer* ldu, double* vt,
                    const integer* ldvt, double* q, integer* iq, double* work,
                    integer* iwork, integer *info );

void FC_FUNC(ssytrd,SSYTRD)( const char* uplo, const integer* n, float* a, const integer* lda,
                    float* d, float* e, float* tau, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dsytrd,DSYTRD)( const char* uplo, const integer* n, double* a, const integer* lda,
                    double* d, double* e, double* tau, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(chetrd,CHETRD)( const char* uplo, const integer* n, scomplex* a, const integer* lda,
                    float* d, float* e, scomplex* tau, scomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(zhetrd,ZHETRD)( const char* uplo, const integer* n, dcomplex* a, const integer* lda,
                    double* d, double* e, dcomplex* tau, dcomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(sorgtr,SORGTR)( const char* uplo, const integer* n, float* a, const integer* lda,
                    const float* tau, float* work, integer* lwork,
                    integer *info );

void FC_FUNC(dorgtr,DORGTR)( const char* uplo, const integer* n, double* a, const integer* lda,
                    const double* tau, double* work, integer* lwork,
                    integer *info );

void FC_FUNC(cungtr,CUNGTR)( const char* uplo, const integer* n, scomplex* a, const integer* lda,
                    const scomplex* tau, scomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(zungtr,ZUNGTR)( const char* uplo, const integer* n, dcomplex* a, const integer* lda,
                    const dcomplex* tau, dcomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(sormtr,SORMTR)( char* side, const char* uplo, const char* trans, const integer* m,
                    const integer* n, const float* a, const integer* lda,
                    const float* tau, float* c, const integer* ldc, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dormtr,DORMTR)( char* side, const char* uplo, const char* trans, const integer* m,
                    const integer* n, const double* a, const integer* lda,
                    const double* tau, double* c, const integer* ldc, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(cunmtr,CUNMTR)( char* side, const char* uplo, const char* trans, const integer* m,
                    const integer* n, const scomplex* a, const integer* lda,
                    const scomplex* tau, scomplex* c, const integer* ldc, scomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(zunmtr,ZUNMTR)( char* side, const char* uplo, const char* trans, const integer* m,
                    const integer* n, const dcomplex* a, const integer* lda,
                    const dcomplex* tau, dcomplex* c, const integer* ldc, dcomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(ssptrd,SSPTRD)( const char* uplo, const integer* n, float* ap, float* d, float* e,
                    float* tau, integer *info );

void FC_FUNC(dsptrd,DSPTRD)( const char* uplo, const integer* n, double* ap, double* d, double* e,
                    double* tau, integer *info );

void FC_FUNC(chptrd,CHPTRD)( const char* uplo, const integer* n, scomplex* ap, float* d, float* e,
                    scomplex* tau, integer *info );

void FC_FUNC(zhptrd,ZHPTRD)( const char* uplo, const integer* n, dcomplex* ap, double* d, double* e,
                    dcomplex* tau, integer *info );

void FC_FUNC(sopgtr,SOPGTR)( const char* uplo, const integer* n, const float* ap,
                    const float* tau, float* q, const integer* ldq, float* work,
                    integer *info );

void FC_FUNC(dopgtr,DOPGTR)( const char* uplo, const integer* n, const double* ap,
                    const double* tau, double* q, const integer* ldq, double* work,
                    integer *info );

void FC_FUNC(cupgtr,CUPGTR)( const char* uplo, const integer* n, const scomplex* ap,
                    const scomplex* tau, scomplex* q, const integer* ldq, scomplex* work,
                    integer *info );

void FC_FUNC(zupgtr,ZUPGTR)( const char* uplo, const integer* n, const dcomplex* ap,
                    const dcomplex* tau, dcomplex* q, const integer* ldq, dcomplex* work,
                    integer *info );

void FC_FUNC(sopmtr,SOPMTR)( char* side, const char* uplo, const char* trans, const integer* m,
                    const integer* n, const float* ap, const float* tau,
                    float* c, const integer* ldc, float* work,
                    integer *info );

void FC_FUNC(dopmtr,DOPMTR)( char* side, const char* uplo, const char* trans, const integer* m,
                    const integer* n, const double* ap, const double* tau,
                    double* c, const integer* ldc, double* work,
                    integer *info );

void FC_FUNC(cupmtr,CUPMTR)( char* side, const char* uplo, const char* trans, const integer* m,
                    const integer* n, const scomplex* ap, const scomplex* tau,
                    scomplex* c, const integer* ldc, scomplex* work,
                    integer *info );

void FC_FUNC(zupmtr,ZUPMTR)( char* side, const char* uplo, const char* trans, const integer* m,
                    const integer* n, const dcomplex* ap, const dcomplex* tau,
                    dcomplex* c, const integer* ldc, dcomplex* work,
                    integer *info );

void FC_FUNC(ssbtrd,SSBTRD)( char* vect, const char* uplo, const integer* n, const integer* kd,
                    float* ab, const integer* ldab, float* d, float* e,
                    float* q, const integer* ldq, float* work,
                    integer *info );

void FC_FUNC(dsbtrd,DSBTRD)( char* vect, const char* uplo, const integer* n, const integer* kd,
                    double* ab, const integer* ldab, double* d, double* e,
                    double* q, const integer* ldq, double* work,
                    integer *info );

void FC_FUNC(chbtrd,CHBTRD)( char* vect, const char* uplo, const integer* n, const integer* kd,
                    scomplex* ab, const integer* ldab, float* d, float* e,
                    scomplex* q, const integer* ldq, scomplex* work,
                    integer *info );

void FC_FUNC(zhbtrd,ZHBTRD)( char* vect, const char* uplo, const integer* n, const integer* kd,
                    dcomplex* ab, const integer* ldab, double* d, double* e,
                    dcomplex* q, const integer* ldq, dcomplex* work,
                    integer *info );

void FC_FUNC(ssterf,SSTERF)( const integer* n, float* d, float* e, integer *info );

void FC_FUNC(dsterf,DSTERF)( const integer* n, double* d, double* e, integer *info );

void FC_FUNC(ssteqr,SSTEQR)( char* compz, const integer* n, float* d, float* e, float* z,
                    const integer* ldz, float* work, integer *info );

void FC_FUNC(dsteqr,DSTEQR)( char* compz, const integer* n, double* d, double* e, double* z,
                    const integer* ldz, double* work, integer *info );

void FC_FUNC(csteqr,CSTEQR)( char* compz, const integer* n, float* d, float* e, scomplex* z,
                    const integer* ldz, float* work, integer *info );

void FC_FUNC(zsteqr,ZSTEQR)( char* compz, const integer* n, double* d, double* e, dcomplex* z,
                    const integer* ldz, double* work, integer *info );

void FC_FUNC(sstemr,SSTEMR)( char* jobz, char* range, const integer* n, float* d,
                    float* e, float* vl, float* vu, integer* il,
                    integer* iu, const integer* m, float* w, float* z,
                    const integer* ldz, const integer* nzc, integer* isuppz,
                    logical* tryrac, float* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(dstemr,DSTEMR)( char* jobz, char* range, const integer* n, double* d,
                    double* e, double* vl, double* vu, integer* il,
                    integer* iu, const integer* m, double* w, double* z,
                    const integer* ldz, const integer* nzc, integer* isuppz,
                    logical* tryrac, double* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(cstemr,CSTEMR)( char* jobz, char* range, const integer* n, float* d,
                    float* e, float* vl, float* vu, integer* il,
                    integer* iu, const integer* m, float* w, scomplex* z,
                    const integer* ldz, const integer* nzc, integer* isuppz,
                    logical* tryrac, float* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(zstemr,ZSTEMR)( char* jobz, char* range, const integer* n, double* d,
                    double* e, double* vl, double* vu, integer* il,
                    integer* iu, const integer* m, double* w, dcomplex* z,
                    const integer* ldz, const integer* nzc, integer* isuppz,
                    logical* tryrac, double* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(sstedc,SSTEDC)( char* compz, const integer* n, float* d, float* e, float* z,
                    const integer* ldz, float* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(dstedc,DSTEDC)( char* compz, const integer* n, double* d, double* e, double* z,
                    const integer* ldz, double* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(cstedc,CSTEDC)( char* compz, const integer* n, float* d, float* e, scomplex* z,
                    const integer* ldz, scomplex* work, integer* lwork, float* rwork, integer* lrwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(zstedc,ZSTEDC)( char* compz, const integer* n, double* d, double* e, dcomplex* z,
                    const integer* ldz, dcomplex* work, integer* lwork, double* rwork, integer* lrwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(sstegr,SSTEGR)( char* jobz, char* range, const integer* n, float* d,
                    float* e, float* vl, float* vu, integer* il,
                    integer* iu, float* abstol, const integer* m, float* w,
                    float* z, const integer* ldz, integer* isuppz,
                    float* work, integer* lwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(dstegr,DSTEGR)( char* jobz, char* range, const integer* n, double* d,
                    double* e, double* vl, double* vu, integer* il,
                    integer* iu, double* abstol, const integer* m, double* w,
                    double* z, const integer* ldz, integer* isuppz,
                    double* work, integer* lwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(cstegr,CSTEGR)( char* jobz, char* range, const integer* n, float* d,
                    float* e, float* vl, float* vu, integer* il,
                    integer* iu, float* abstol, const integer* m, float* w,
                    scomplex* z, const integer* ldz, integer* isuppz,
                    float* work, integer* lwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(zstegr,ZSTEGR)( char* jobz, char* range, const integer* n, double* d,
                    double* e, double* vl, double* vu, integer* il,
                    integer* iu, double* abstol, const integer* m, double* w,
                    dcomplex* z, const integer* ldz, integer* isuppz,
                    double* work, integer* lwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(spteqr,SPTEQR)( char* compz, const integer* n, float* d, float* e, float* z,
                    const integer* ldz, float* work, integer *info );

void FC_FUNC(dpteqr,DPTEQR)( char* compz, const integer* n, double* d, double* e, double* z,
                    const integer* ldz, double* work, integer *info );

void FC_FUNC(cpteqr,CPTEQR)( char* compz, const integer* n, float* d, float* e, scomplex* z,
                    const integer* ldz, float* work, integer *info );

void FC_FUNC(zpteqr,ZPTEQR)( char* compz, const integer* n, double* d, double* e, dcomplex* z,
                    const integer* ldz, double* work, integer *info );

void FC_FUNC(sstebz,SSTEBZ)( char* range, char* order, const integer* n, float* vl,
                    float* vu, integer* il, integer* iu, float* abstol,
                    const float* d, const float* e, const integer* m,
                    const integer* nsplit, float* w, integer* iblock,
                    integer* isplit, float* work, integer* iwork,
                    integer *info );

void FC_FUNC(dstebz,DSTEBZ)( char* range, char* order, const integer* n, double* vl,
                    double* vu, integer* il, integer* iu, double* abstol,
                    const double* d, const double* e, const integer* m,
                    const integer* nsplit, double* w, integer* iblock,
                    integer* isplit, double* work, integer* iwork,
                    integer *info );

void FC_FUNC(sstein,SSTEIN)( const integer* n, const float* d, const float* e,
                    const integer* m, const float* w, const integer* iblock,
                    const integer* isplit, float* z, const integer* ldz,
                    float* work, integer* iwork, integer* ifailv,
                    integer *info );

void FC_FUNC(dstein,DSTEIN)( const integer* n, const double* d, const double* e,
                    const integer* m, const double* w, const integer* iblock,
                    const integer* isplit, double* z, const integer* ldz,
                    double* work, integer* iwork, integer* ifailv,
                    integer *info );

void FC_FUNC(cstein,CSTEIN)( const integer* n, const float* d, const float* e,
                    const integer* m, const float* w, const integer* iblock,
                    const integer* isplit, scomplex* z, const integer* ldz,
                    float* work, integer* iwork, integer* ifailv,
                    integer *info );

void FC_FUNC(zstein,ZSTEIN)( const integer* n, const double* d, const double* e,
                    const integer* m, const double* w, const integer* iblock,
                    const integer* isplit, dcomplex* z, const integer* ldz,
                    double* work, integer* iwork, integer* ifailv,
                    integer *info );

void FC_FUNC(sdisna,SDISNA)( char* job, const integer* m, const integer* n, const float* d,
                    float* sep, integer *info );

void FC_FUNC(ddisna,DDISNA)( char* job, const integer* m, const integer* n, const double* d,
                    double* sep, integer *info );

void FC_FUNC(ssygst,SSYGST)( integer* itype, const char* uplo, const integer* n, float* a,
                    const integer* lda, const float* b, const integer* ldb,
                    integer *info );

void FC_FUNC(dsygst,DSYGST)( integer* itype, const char* uplo, const integer* n, double* a,
                    const integer* lda, const double* b, const integer* ldb,
                    integer *info );

void FC_FUNC(chegst,CHEGST)( integer* itype, const char* uplo, const integer* n, scomplex* a,
                    const integer* lda, const scomplex* b, const integer* ldb,
                    integer *info );

void FC_FUNC(zhegst,ZHEGST)( integer* itype, const char* uplo, const integer* n, dcomplex* a,
                    const integer* lda, const dcomplex* b, const integer* ldb,
                    integer *info );

void FC_FUNC(sspgst,SSPGST)( integer* itype, const char* uplo, const integer* n, float* ap,
                    const float* bp, integer *info );

void FC_FUNC(dspgst,DSPGST)( integer* itype, const char* uplo, const integer* n, double* ap,
                    const double* bp, integer *info );

void FC_FUNC(chpgst,CHPGST)( integer* itype, const char* uplo, const integer* n, scomplex* ap,
                    const scomplex* bp, integer *info );

void FC_FUNC(zhpgst,ZHPGST)( integer* itype, const char* uplo, const integer* n, dcomplex* ap,
                    const dcomplex* bp, integer *info );

void FC_FUNC(ssbgst,SSBGST)( char* vect, const char* uplo, const integer* n, const integer* ka,
                    const integer* kb, float* ab, const integer* ldab,
                    const float* bb, const integer* ldbb, float* x,
                    const integer* ldx, float* work, integer *info );

void FC_FUNC(dsbgst,DSBGST)( char* vect, const char* uplo, const integer* n, const integer* ka,
                    const integer* kb, double* ab, const integer* ldab,
                    const double* bb, const integer* ldbb, double* x,
                    const integer* ldx, double* work, integer *info );

void FC_FUNC(chbgst,CHBGST)( char* vect, const char* uplo, const integer* n, const integer* ka,
                    const integer* kb, scomplex* ab, const integer* ldab,
                    const scomplex* bb, const integer* ldbb, scomplex* x,
                    const integer* ldx, scomplex* work, float* rwork, integer *info );

void FC_FUNC(zhbgst,ZHBGST)( char* vect, const char* uplo, const integer* n, const integer* ka,
                    const integer* kb, dcomplex* ab, const integer* ldab,
                    const dcomplex* bb, const integer* ldbb, dcomplex* x,
                    const integer* ldx, dcomplex* work, double* rwork, integer *info );

void FC_FUNC(spbstf,SPBSTF)( const char* uplo, const integer* n, const integer* kb, float* bb,
                    const integer* ldbb, integer *info );

void FC_FUNC(dpbstf,DPBSTF)( const char* uplo, const integer* n, const integer* kb, double* bb,
                    const integer* ldbb, integer *info );

void FC_FUNC(cpbstf,CPBSTF)( const char* uplo, const integer* n, const integer* kb, scomplex* bb,
                    const integer* ldbb, integer *info );

void FC_FUNC(zpbstf,ZPBSTF)( const char* uplo, const integer* n, const integer* kb, dcomplex* bb,
                    const integer* ldbb, integer *info );

void FC_FUNC(sgehrd,SGEHRD)( const integer* n, integer* ilo, integer* ihi, float* a,
                    const integer* lda, float* tau, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dgehrd,DGEHRD)( const integer* n, integer* ilo, integer* ihi, double* a,
                    const integer* lda, double* tau, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(cgehrd,CGEHRD)( const integer* n, integer* ilo, integer* ihi, scomplex* a,
                    const integer* lda, scomplex* tau, scomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(zgehrd,ZGEHRD)( const integer* n, integer* ilo, integer* ihi, dcomplex* a,
                    const integer* lda, dcomplex* tau, dcomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(sorghr,SORGHR)( const integer* n, integer* ilo, integer* ihi, float* a,
                    const integer* lda, const float* tau, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dorghr,DORGHR)( const integer* n, integer* ilo, integer* ihi, double* a,
                    const integer* lda, const double* tau, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(cunghr,CUNGHR)( const integer* n, integer* ilo, integer* ihi, scomplex* a,
                    const integer* lda, const scomplex* tau, scomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(zunghr,ZUNGHR)( const integer* n, integer* ilo, integer* ihi, dcomplex* a,
                    const integer* lda, const dcomplex* tau, dcomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(sormhr,SORMHR)( char* side, const char* trans, const integer* m, const integer* n,
                    integer* ilo, integer* ihi, const float* a,
                    const integer* lda, const float* tau, float* c,
                    const integer* ldc, float* work, integer* lwork,
                    integer *info );

void FC_FUNC(dormhr,DORMHR)( char* side, const char* trans, const integer* m, const integer* n,
                    integer* ilo, integer* ihi, const double* a,
                    const integer* lda, const double* tau, double* c,
                    const integer* ldc, double* work, integer* lwork,
                    integer *info );

void FC_FUNC(cunmhr,CUNMHR)( char* side, const char* trans, const integer* m, const integer* n,
                    integer* ilo, integer* ihi, const scomplex* a,
                    const integer* lda, const scomplex* tau, scomplex* c,
                    const integer* ldc, scomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(zunmhr,ZUNMHR)( char* side, const char* trans, const integer* m, const integer* n,
                    integer* ilo, integer* ihi, const dcomplex* a,
                    const integer* lda, const dcomplex* tau, dcomplex* c,
                    const integer* ldc, dcomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(sgebal,SGEBAL)( char* job, const integer* n, float* a, const integer* lda,
                    integer* ilo, integer* ihi, float* scale,
                    integer *info );

void FC_FUNC(dgebal,DGEBAL)( char* job, const integer* n, double* a, const integer* lda,
                    integer* ilo, integer* ihi, double* scale,
                    integer *info );

void FC_FUNC(cgebal,CGEBAL)( char* job, const integer* n, scomplex* a, const integer* lda,
                    integer* ilo, integer* ihi, float* scale,
                    integer *info );

void FC_FUNC(zgebal,ZGEBAL)( char* job, const integer* n, dcomplex* a, const integer* lda,
                    integer* ilo, integer* ihi, double* scale,
                    integer *info );

void FC_FUNC(sgebak,SGEBAK)( char* job, char* side, const integer* n, integer* ilo,
                    integer* ihi, const float* scale, const integer* m,
                    float* v, const integer* ldv, integer *info );

void FC_FUNC(dgebak,DGEBAK)( char* job, char* side, const integer* n, integer* ilo,
                    integer* ihi, const double* scale, const integer* m,
                    double* v, const integer* ldv, integer *info );

void FC_FUNC(cgebak,CGEBAK)( char* job, char* side, const integer* n, integer* ilo,
                    integer* ihi, const float* scale, const integer* m,
                    scomplex* v, const integer* ldv, integer *info );

void FC_FUNC(zgebak,ZGEBAK)( char* job, char* side, const integer* n, integer* ilo,
                    integer* ihi, const double* scale, const integer* m,
                    dcomplex* v, const integer* ldv, integer *info );

void FC_FUNC(shseqr,SHSEQR)( char* job, char* compz, const integer* n, integer* ilo,
                    integer* ihi, float* h, const integer* ldh, float* wr,
                    float* wi, float* z, const integer* ldz, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dhseqr,DHSEQR)( char* job, char* compz, const integer* n, integer* ilo,
                    integer* ihi, double* h, const integer* ldh, double* wr,
                    double* wi, double* z, const integer* ldz, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(chseqr,CHSEQR)( char* job, char* compz, const integer* n, integer* ilo,
                    integer* ihi, scomplex* h, const integer* ldh, scomplex* w,
                    scomplex* z, const integer* ldz, scomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(zhseqr,ZHSEQR)( char* job, char* compz, const integer* n, integer* ilo,
                    integer* ihi, dcomplex* h, const integer* ldh, dcomplex* w,
                    dcomplex* z, const integer* ldz, dcomplex* work,
                    integer* lwork, integer *info );

void FC_FUNC(shsein,SHSEIN)( char* job, char* eigsrc, char* initv,
                    logical* select, const integer* n, const float* h,
                    const integer* ldh, float* wr, const float* wi, float* vl,
                    const integer* ldvl, float* vr, const integer* ldvr,
                    const integer* mm, const integer* m, float* work,
                    integer* ifaill, integer* ifailr, integer *info );

void FC_FUNC(dhsein,DHSEIN)( char* job, char* eigsrc, char* initv,
                    logical* select, const integer* n, const double* h,
                    const integer* ldh, double* wr, const double* wi, double* vl,
                    const integer* ldvl, double* vr, const integer* ldvr,
                    const integer* mm, const integer* m, double* work,
                    integer* ifaill, integer* ifailr, integer *info );

void FC_FUNC(chsein,CHSEIN)( char* job, char* eigsrc, char* initv,
                    logical* select, const integer* n, const scomplex* h,
                    const integer* ldh, scomplex* w, scomplex* vl,
                    const integer* ldvl, scomplex* vr, const integer* ldvr,
                    const integer* mm, const integer* m, scomplex* work, float* rwork,
                    integer* ifaill, integer* ifailr, integer *info );

void FC_FUNC(zhsein,ZHSEIN)( char* job, char* eigsrc, char* initv,
                    logical* select, const integer* n, const dcomplex* h,
                    const integer* ldh, dcomplex* w, dcomplex* vl,
                    const integer* ldvl, dcomplex* vr, const integer* ldvr,
                    const integer* mm, const integer* m, dcomplex* work, double* rwork,
                    integer* ifaill, integer* ifailr, integer *info );

void FC_FUNC(strevc,STREVC)( char* side, char* howmny, const logical* select,
                    const integer* n, const float* t, const integer* ldt, float* vl,
                    const integer* ldvl, float* vr, const integer* ldvr,
                    const integer* mm, const integer* m, float* work,
                    integer *info );

void FC_FUNC(dtrevc,DTREVC)( char* side, char* howmny, const logical* select,
                    const integer* n, const double* t, const integer* ldt, double* vl,
                    const integer* ldvl, double* vr, const integer* ldvr,
                    const integer* mm, const integer* m, double* work,
                    integer *info );

void FC_FUNC(ctrevc,CTREVC)( char* side, char* howmny, const logical* select,
                    const integer* n, const scomplex* t, const integer* ldt, scomplex* vl,
                    const integer* ldvl, scomplex* vr, const integer* ldvr,
                    const integer* mm, const integer* m, scomplex* work, float* rwork,
                    integer *info );

void FC_FUNC(ztrevc,ZTREVC)( char* side, char* howmny, const logical* select,
                    const integer* n, const dcomplex* t, const integer* ldt, dcomplex* vl,
                    const integer* ldvl, dcomplex* vr, const integer* ldvr,
                    const integer* mm, const integer* m, dcomplex* work, double* rwork,
                    integer *info );

void FC_FUNC(strsna,STRSNA)( char* job, char* howmny, const logical* select,
                    const integer* n, const float* t, const integer* ldt,
                    const float* vl, const integer* ldvl, const float* vr,
                    const integer* ldvr, float* s, float* sep, const integer* mm,
                    const integer* m, float* work, const integer* ldwork,
                    integer* iwork, integer *info );

void FC_FUNC(dtrsna,DTRSNA)( char* job, char* howmny, const logical* select,
                    const integer* n, const double* t, const integer* ldt,
                    const double* vl, const integer* ldvl, const double* vr,
                    const integer* ldvr, double* s, double* sep, const integer* mm,
                    const integer* m, double* work, const integer* ldwork,
                    integer* iwork, integer *info );

void FC_FUNC(ctrsna,CTRSNA)( char* job, char* howmny, const logical* select,
                    const integer* n, const scomplex* t, const integer* ldt,
                    const scomplex* vl, const integer* ldvl, const scomplex* vr,
                    const integer* ldvr, float* s, float* sep, const integer* mm,
                    const integer* m, scomplex* work, const integer* ldwork,
                    float* rwork, integer *info );

void FC_FUNC(ztrsna,ZTRSNA)( char* job, char* howmny, const logical* select,
                    const integer* n, const dcomplex* t, const integer* ldt,
                    const dcomplex* vl, const integer* ldvl, const dcomplex* vr,
                    const integer* ldvr, double* s, double* sep, const integer* mm,
                    const integer* m, dcomplex* work, const integer* ldwork,
                    double* rwork, integer *info );

void FC_FUNC(strexc,STREXC)( char* compq, const integer* n, float* t, const integer* ldt,
                    float* q, const integer* ldq, integer* ifst,
                    integer* ilst, float* work, integer *info );

void FC_FUNC(dtrexc,DTREXC)( char* compq, const integer* n, double* t, const integer* ldt,
                    double* q, const integer* ldq, integer* ifst,
                    integer* ilst, double* work, integer *info );

void FC_FUNC(ctrexc,CTREXC)( char* compq, const integer* n, scomplex* t, const integer* ldt,
                    scomplex* q, const integer* ldq, integer* ifst,
                    integer* ilst, integer *info );

void FC_FUNC(ztrexc,ZTREXC)( char* compq, const integer* n, dcomplex* t, const integer* ldt,
                    dcomplex* q, const integer* ldq, integer* ifst,
                    integer* ilst, integer *info );

void FC_FUNC(strsen,STRSEN)( char* job, char* compq, const logical* select,
                    const integer* n, float* t, const integer* ldt, float* q,
                    const integer* ldq, float* wr, float* wi, const integer* m,
                    float* s, float* sep, float* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(dtrsen,DTRSEN)( char* job, char* compq, const logical* select,
                    const integer* n, double* t, const integer* ldt, double* q,
                    const integer* ldq, double* wr, double* wi, const integer* m,
                    double* s, double* sep, double* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(ctrsen,CTRSEN)( char* job, char* compq, const logical* select,
                    const integer* n, scomplex* t, const integer* ldt, scomplex* q,
                    const integer* ldq, scomplex* w, const integer* m,
                    float* s, float* sep, scomplex* work, integer* lwork, integer *info );

void FC_FUNC(ztrsen,ZTRSEN)( char* job, char* compq, const logical* select,
                    const integer* n, dcomplex* t, const integer* ldt, dcomplex* q,
                    const integer* ldq, dcomplex* w, const integer* m,
                    double* s, double* sep, dcomplex* work, integer* lwork, integer *info );

void FC_FUNC(strsyl,STRSYL)( char* trana, char* tranb, integer* isgn, const integer* m,
                    const integer* n, const float* a, const integer* lda,
                    const float* b, const integer* ldb, float* c,
                    const integer* ldc, float* scale, integer *info );

void FC_FUNC(dtrsyl,DTRSYL)( char* trana, char* tranb, integer* isgn, const integer* m,
                    const integer* n, const double* a, const integer* lda,
                    const double* b, const integer* ldb, double* c,
                    const integer* ldc, double* scale, integer *info );

void FC_FUNC(ctrsyl,CTRSYL)( char* trana, char* tranb, integer* isgn, const integer* m,
                    const integer* n, const scomplex* a, const integer* lda,
                    const scomplex* b, const integer* ldb, scomplex* c,
                    const integer* ldc, float* scale, integer *info );

void FC_FUNC(ztrsyl,ZTRSYL)( char* trana, char* tranb, integer* isgn, const integer* m,
                    const integer* n, const dcomplex* a, const integer* lda,
                    const dcomplex* b, const integer* ldb, dcomplex* c,
                    const integer* ldc, double* scale, integer *info );

void FC_FUNC(sgghrd,SGGHRD)( char* compq, char* compz, const integer* n, integer* ilo,
                    integer* ihi, float* a, const integer* lda, float* b,
                    const integer* ldb, float* q, const integer* ldq, float* z,
                    const integer* ldz, integer *info );

void FC_FUNC(dgghrd,DGGHRD)( char* compq, char* compz, const integer* n, integer* ilo,
                    integer* ihi, double* a, const integer* lda, double* b,
                    const integer* ldb, double* q, const integer* ldq, double* z,
                    const integer* ldz, integer *info );

void FC_FUNC(cgghrd,CGGHRD)( char* compq, char* compz, const integer* n, integer* ilo,
                    integer* ihi, scomplex* a, const integer* lda, scomplex* b,
                    const integer* ldb, scomplex* q, const integer* ldq, scomplex* z,
                    const integer* ldz, integer *info );

void FC_FUNC(zgghrd,ZGGHRD)( char* compq, char* compz, const integer* n, integer* ilo,
                    integer* ihi, dcomplex* a, const integer* lda, dcomplex* b,
                    const integer* ldb, dcomplex* q, const integer* ldq, dcomplex* z,
                    const integer* ldz, integer *info );

void FC_FUNC(sggbal,SGGBAL)( char* job, const integer* n, float* a, const integer* lda,
                    float* b, const integer* ldb, integer* ilo,
                    integer* ihi, float* lscale, float* rscale,
                    float* work, integer *info );

void FC_FUNC(dggbal,DGGBAL)( char* job, const integer* n, double* a, const integer* lda,
                    double* b, const integer* ldb, integer* ilo,
                    integer* ihi, double* lscale, double* rscale,
                    double* work, integer *info );

void FC_FUNC(cggbal,CGGBAL)( char* job, const integer* n, scomplex* a, const integer* lda,
                    scomplex* b, const integer* ldb, integer* ilo,
                    integer* ihi, float* lscale, float* rscale,
                    float* work, integer *info );

void FC_FUNC(zggbal,ZGGBAL)( char* job, const integer* n, dcomplex* a, const integer* lda,
                    dcomplex* b, const integer* ldb, integer* ilo,
                    integer* ihi, double* lscale, double* rscale,
                    double* work, integer *info );

void FC_FUNC(sggbak,SGGBAK)( char* job, char* side, const integer* n, integer* ilo,
                    integer* ihi, const float* lscale, const float* rscale,
                    const integer* m, float* v, const integer* ldv,
                    integer *info );

void FC_FUNC(dggbak,DGGBAK)( char* job, char* side, const integer* n, integer* ilo,
                    integer* ihi, const double* lscale, const double* rscale,
                    const integer* m, double* v, const integer* ldv,
                    integer *info );

void FC_FUNC(cggbak,CGGBAK)( char* job, char* side, const integer* n, integer* ilo,
                    integer* ihi, const float* lscale, const float* rscale,
                    const integer* m, scomplex* v, const integer* ldv,
                    integer *info );

void FC_FUNC(zggbak,ZGGBAK)( char* job, char* side, const integer* n, integer* ilo,
                    integer* ihi, const double* lscale, const double* rscale,
                    const integer* m, dcomplex* v, const integer* ldv,
                    integer *info );

void FC_FUNC(shgeqz,SHGEQZ)( char* job, char* compq, char* compz, const integer* n,
                    integer* ilo, integer* ihi, float* h,
                    const integer* ldh, float* t, const integer* ldt, float* alphar,
                    float* alphai, float* beta, float* q, const integer* ldq,
                    float* z, const integer* ldz, float* work, integer* lwork,
                    integer *info );

void FC_FUNC(dhgeqz,DHGEQZ)( char* job, char* compq, char* compz, const integer* n,
                    integer* ilo, integer* ihi, double* h,
                    const integer* ldh, double* t, const integer* ldt, double* alphar,
                    double* alphai, double* beta, double* q, const integer* ldq,
                    double* z, const integer* ldz, double* work, integer* lwork,
                    integer *info );

void FC_FUNC(chgeqz,CHGEQZ)( char* job, char* compq, char* compz, const integer* n,
                    integer* ilo, integer* ihi, scomplex* h,
                    const integer* ldh, scomplex* t, const integer* ldt, scomplex* alpha,
                    scomplex* beta, scomplex* q, const integer* ldq,
                    scomplex* z, const integer* ldz, scomplex* work, integer* lwork,
                    float* rwork, integer *info );

void FC_FUNC(zhgeqz,ZHGEQZ)( char* job, char* compq, char* compz, const integer* n,
                    integer* ilo, integer* ihi, dcomplex* h,
                    const integer* ldh, dcomplex* t, const integer* ldt, dcomplex* alpha,
                    dcomplex* beta, dcomplex* q, const integer* ldq,
                    dcomplex* z, const integer* ldz, dcomplex* work, integer* lwork,
                    double* rwork, integer *info );

void FC_FUNC(stgevc,STGEVC)( char* side, char* howmny, const logical* select,
                    const integer* n, const float* s, const integer* lds,
                    const float* p, const integer* ldp, float* vl,
                    const integer* ldvl, float* vr, const integer* ldvr,
                    const integer* mm, const integer* m, float* work,
                    integer *info );

void FC_FUNC(dtgevc,DTGEVC)( char* side, char* howmny, const logical* select,
                    const integer* n, const double* s, const integer* lds,
                    const double* p, const integer* ldp, double* vl,
                    const integer* ldvl, double* vr, const integer* ldvr,
                    const integer* mm, const integer* m, double* work,
                    integer *info );

void FC_FUNC(ctgevc,CTGEVC)( char* side, char* howmny, const logical* select,
                    const integer* n, const scomplex* s, const integer* lds,
                    const scomplex* p, const integer* ldp, scomplex* vl,
                    const integer* ldvl, scomplex* vr, const integer* ldvr,
                    const integer* mm, const integer* m, scomplex* work, float* rwork,
                    integer *info );

void FC_FUNC(ztgevc,ZTGEVC)( char* side, char* howmny, const logical* select,
                    const integer* n, const dcomplex* s, const integer* lds,
                    const dcomplex* p, const integer* ldp, dcomplex* vl,
                    const integer* ldvl, dcomplex* vr, const integer* ldvr,
                    const integer* mm, const integer* m, dcomplex* work, double* rwork,
                    integer *info );

void FC_FUNC(stgexc,STGEXC)( logical* wantq, logical* wantz, const integer* n,
                    float* a, const integer* lda, float* b, const integer* ldb,
                    float* q, const integer* ldq, float* z, const integer* ldz,
                    integer* ifst, integer* ilst, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dtgexc,DTGEXC)( logical* wantq, logical* wantz, const integer* n,
                    double* a, const integer* lda, double* b, const integer* ldb,
                    double* q, const integer* ldq, double* z, const integer* ldz,
                    integer* ifst, integer* ilst, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(ctgexc,CTGEXC)( logical* wantq, logical* wantz, const integer* n,
                    scomplex* a, const integer* lda, scomplex* b, const integer* ldb,
                    scomplex* q, const integer* ldq, scomplex* z, const integer* ldz,
                    integer* ifst, integer* ilst, integer *info );

void FC_FUNC(ztgexc,ZTGEXC)( logical* wantq, logical* wantz, const integer* n,
                    dcomplex* a, const integer* lda, dcomplex* b, const integer* ldb,
                    dcomplex* q, const integer* ldq, dcomplex* z, const integer* ldz,
                    integer* ifst, integer* ilst, integer *info );

void FC_FUNC(stgsen,STGSEN)( integer* ijob, logical* wantq,
                    logical* wantz, const logical* select,
                    const integer* n, float* a, const integer* lda, float* b,
                    const integer* ldb, float* alphar, float* alphai,
                    float* beta, float* q, const integer* ldq, float* z,
                    const integer* ldz, const integer* m, float* pl, float* pr,
                    float* dif, float* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(dtgsen,DTGSEN)( integer* ijob, logical* wantq,
                    logical* wantz, const logical* select,
                    const integer* n, double* a, const integer* lda, double* b,
                    const integer* ldb, double* alphar, double* alphai,
                    double* beta, double* q, const integer* ldq, double* z,
                    const integer* ldz, const integer* m, double* pl, double* pr,
                    double* dif, double* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(ctgsen,CTGSEN)( integer* ijob, logical* wantq,
                    logical* wantz, const logical* select,
                    const integer* n, scomplex* a, const integer* lda, scomplex* b,
                    const integer* ldb, scomplex* alpha,
                    scomplex* beta, scomplex* q, const integer* ldq, scomplex* z,
                    const integer* ldz, const integer* m, float* pl, float* pr,
                    float* dif, scomplex* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(ztgsen,ZTGSEN)( integer* ijob, logical* wantq,
                    logical* wantz, const logical* select,
                    const integer* n, dcomplex* a, const integer* lda, dcomplex* b,
                    const integer* ldb, dcomplex* alpha,
                    dcomplex* beta, dcomplex* q, const integer* ldq, dcomplex* z,
                    const integer* ldz, const integer* m, double* pl, double* pr,
                    double* dif, dcomplex* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(stgsyl,STGSYL)( const char* trans, integer* ijob, const integer* m, const integer* n,
                    const float* a, const integer* lda, const float* b,
                    const integer* ldb, float* c, const integer* ldc,
                    const float* d, const integer* ldd, const float* e,
                    const integer* lde, float* f, const integer* ldf, float* scale,
                    float* dif, float* work, integer* lwork,
                    integer* iwork, integer *info );

void FC_FUNC(dtgsyl,DTGSYL)( const char* trans, integer* ijob, const integer* m, const integer* n,
                    const double* a, const integer* lda, const double* b,
                    const integer* ldb, double* c, const integer* ldc,
                    const double* d, const integer* ldd, const double* e,
                    const integer* lde, double* f, const integer* ldf, double* scale,
                    double* dif, double* work, integer* lwork,
                    integer* iwork, integer *info );

void FC_FUNC(ctgsyl,CTGSYL)( const char* trans, integer* ijob, const integer* m, const integer* n,
                    const scomplex* a, const integer* lda, const scomplex* b,
                    const integer* ldb, scomplex* c, const integer* ldc,
                    const scomplex* d, const integer* ldd, const scomplex* e,
                    const integer* lde, scomplex* f, const integer* ldf, float* scale,
                    float* dif, scomplex* work, integer* lwork,
                    integer* iwork, integer *info );

void FC_FUNC(ztgsyl,ZTGSYL)( const char* trans, integer* ijob, const integer* m, const integer* n,
                    const dcomplex* a, const integer* lda, const dcomplex* b,
                    const integer* ldb, dcomplex* c, const integer* ldc,
                    const dcomplex* d, const integer* ldd, const dcomplex* e,
                    const integer* lde, dcomplex* f, const integer* ldf, double* scale,
                    double* dif, dcomplex* work, integer* lwork,
                    integer* iwork, integer *info );

void FC_FUNC(stgsna,STGSNA)( char* job, char* howmny, const logical* select,
                    const integer* n, const float* a, const integer* lda,
                    const float* b, const integer* ldb, const float* vl,
                    const integer* ldvl, const float* vr, const integer* ldvr,
                    float* s, float* dif, const integer* mm, const integer* m,
                    float* work, integer* lwork, integer* iwork,
                    integer *info );

void FC_FUNC(dtgsna,DTGSNA)( char* job, char* howmny, const logical* select,
                    const integer* n, const double* a, const integer* lda,
                    const double* b, const integer* ldb, const double* vl,
                    const integer* ldvl, const double* vr, const integer* ldvr,
                    double* s, double* dif, const integer* mm, const integer* m,
                    double* work, integer* lwork, integer* iwork,
                    integer *info );

void FC_FUNC(ctgsna,CTGSNA)( char* job, char* howmny, const logical* select,
                    const integer* n, const scomplex* a, const integer* lda,
                    const scomplex* b, const integer* ldb, const scomplex* vl,
                    const integer* ldvl, const scomplex* vr, const integer* ldvr,
                    float* s, float* dif, const integer* mm, const integer* m,
                    scomplex* work, integer* lwork, integer* iwork,
                    integer *info );

void FC_FUNC(ztgsna,ZTGSNA)( char* job, char* howmny, const logical* select,
                    const integer* n, const dcomplex* a, const integer* lda,
                    const dcomplex* b, const integer* ldb, const dcomplex* vl,
                    const integer* ldvl, const dcomplex* vr, const integer* ldvr,
                    double* s, double* dif, const integer* mm, const integer* m,
                    dcomplex* work, integer* lwork, integer* iwork,
                    integer *info );

void FC_FUNC(sggsvp,SGGSVP)( char* jobu, char* jobv, char* jobq, const integer* m,
                    integer* p, const integer* n, float* a, const integer* lda,
                    float* b, const integer* ldb, float* tola, float* tolb,
                    const integer* k, integer* l, float* u, const integer* ldu,
                    float* v, const integer* ldv, float* q, const integer* ldq,
                    integer* iwork, float* tau, float* work,
                    integer *info );

void FC_FUNC(dggsvp,DGGSVP)( char* jobu, char* jobv, char* jobq, const integer* m,
                    integer* p, const integer* n, double* a, const integer* lda,
                    double* b, const integer* ldb, double* tola, double* tolb,
                    const integer* k, integer* l, double* u, const integer* ldu,
                    double* v, const integer* ldv, double* q, const integer* ldq,
                    integer* iwork, double* tau, double* work,
                    integer *info );

void FC_FUNC(cggsvp,CGGSVP)( char* jobu, char* jobv, char* jobq, const integer* m,
                    integer* p, const integer* n, scomplex* a, const integer* lda,
                    scomplex* b, const integer* ldb, float* tola, float* tolb,
                    const integer* k, integer* l, scomplex* u, const integer* ldu,
                    scomplex* v, const integer* ldv, scomplex* q, const integer* ldq,
                    integer* iwork, float* rwork, scomplex* tau, scomplex* work,
                    integer *info );

void FC_FUNC(zggsvp,ZGGSVP)( char* jobu, char* jobv, char* jobq, const integer* m,
                    integer* p, const integer* n, dcomplex* a, const integer* lda,
                    dcomplex* b, const integer* ldb, double* tola, double* tolb,
                    const integer* k, integer* l, dcomplex* u, const integer* ldu,
                    dcomplex* v, const integer* ldv, dcomplex* q, const integer* ldq,
                    integer* iwork, double* rwork, dcomplex* tau, dcomplex* work,
                    integer *info );

void FC_FUNC(stgsja,STGSJA)( char* jobu, char* jobv, char* jobq, const integer* m,
                    integer* p, const integer* n, const integer* k, integer* l,
                    float* a, const integer* lda, float* b, const integer* ldb,
                    float* tola, float* tolb, float* alpha, float* beta,
                    float* u, const integer* ldu, float* v, const integer* ldv,
                    float* q, const integer* ldq, float* work,
                    const integer* ncycle, integer *info );

void FC_FUNC(dtgsja,DTGSJA)( char* jobu, char* jobv, char* jobq, const integer* m,
                    integer* p, const integer* n, const integer* k, integer* l,
                    double* a, const integer* lda, double* b, const integer* ldb,
                    double* tola, double* tolb, double* alpha, double* beta,
                    double* u, const integer* ldu, double* v, const integer* ldv,
                    double* q, const integer* ldq, double* work,
                    const integer* ncycle, integer *info );

void FC_FUNC(ctgsja,CTGSJA)( char* jobu, char* jobv, char* jobq, const integer* m,
                    integer* p, const integer* n, const integer* k, integer* l,
                    scomplex* a, const integer* lda, scomplex* b, const integer* ldb,
                    float* tola, float* tolb, float* alpha, float* beta,
                    scomplex* u, const integer* ldu, scomplex* v, const integer* ldv,
                    scomplex* q, const integer* ldq, scomplex* work,
                    const integer* ncycle, integer *info );

void FC_FUNC(ztgsja,ZTGSJA)( char* jobu, char* jobv, char* jobq, const integer* m,
                    integer* p, const integer* n, const integer* k, integer* l,
                    dcomplex* a, const integer* lda, dcomplex* b, const integer* ldb,
                    double* tola, double* tolb, double* alpha, double* beta,
                    dcomplex* u, const integer* ldu, dcomplex* v, const integer* ldv,
                    dcomplex* q, const integer* ldq, dcomplex* work,
                    const integer* ncycle, integer *info );

void FC_FUNC(sgels,SGELS)( const char* trans, const integer* m, const integer* n, const integer* nrhs,
                   float* a, const integer* lda, float* b, const integer* ldb,
                   float* work, integer* lwork, integer *info );

void FC_FUNC(dgels,DGELS)( const char* trans, const integer* m, const integer* n, const integer* nrhs,
                   double* a, const integer* lda, double* b, const integer* ldb,
                   double* work, integer* lwork, integer *info );

void FC_FUNC(cgels,CGELS)( const char* trans, const integer* m, const integer* n, const integer* nrhs,
                    scomplex* a, const integer* lda, scomplex* b, const integer* ldb,
                    scomplex* work, integer* lwork, integer *info );

void FC_FUNC(zgels,ZGELS)( const char* trans, const integer* m, const integer* n, const integer* nrhs,
                   dcomplex* a, const integer* lda, dcomplex* b, const integer* ldb,
                   dcomplex* work, integer* lwork, integer *info );

void FC_FUNC(sgelsy,SGELSY)( const integer* m, const integer* n, const integer* nrhs, float* a,
                    const integer* lda, float* b, const integer* ldb,
                    integer* jpvt, float* rcond, integer* rank,
                    float* work, integer* lwork, integer *info );

void FC_FUNC(dgelsy,DGELSY)( const integer* m, const integer* n, const integer* nrhs, double* a,
                    const integer* lda, double* b, const integer* ldb,
                    integer* jpvt, double* rcond, integer* rank,
                    double* work, integer* lwork, integer *info );

void FC_FUNC(cgelsy,CGELSY)( const integer* m, const integer* n, const integer* nrhs, scomplex* a,
                    const integer* lda, scomplex* b, const integer* ldb,
                    integer* jpvt, float* rcond, integer* rank,
                    scomplex* work, integer* lwork, float* rwork, integer *info );

void FC_FUNC(zgelsy,ZGELSY)( const integer* m, const integer* n, const integer* nrhs, dcomplex* a,
                    const integer* lda, dcomplex* b, const integer* ldb,
                    integer* jpvt, double* rcond, integer* rank,
                    dcomplex* work, integer* lwork, double* rwork, integer *info );

void FC_FUNC(sgelss,SGELSS)( const integer* m, const integer* n, const integer* nrhs, float* a,
                    const integer* lda, float* b, const integer* ldb, float* s,
                    float* rcond, integer* rank, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dgelss,DGELSS)( const integer* m, const integer* n, const integer* nrhs, double* a,
                    const integer* lda, double* b, const integer* ldb, double* s,
                    double* rcond, integer* rank, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(cgelss,CGELSS)( const integer* m, const integer* n, const integer* nrhs, scomplex* a,
                    const integer* lda, scomplex* b, const integer* ldb, float* s,
                    float* rcond, integer* rank, scomplex* work,
                    integer* lwork, float* rwork, integer *info );

void FC_FUNC(zgelss,ZGELSS)( const integer* m, const integer* n, const integer* nrhs, dcomplex* a,
                    const integer* lda, dcomplex* b, const integer* ldb, double* s,
                    double* rcond, integer* rank, dcomplex* work,
                    integer* lwork, double* rwork, integer *info );

void FC_FUNC(sgglse,SGGLSE)( const integer* m, const integer* n, integer* p, float* a,
                    const integer* lda, float* b, const integer* ldb, float* c,
                    float* d, float* x, float* work, integer* lwork,
                    integer *info );

void FC_FUNC(dgglse,DGGLSE)( const integer* m, const integer* n, integer* p, double* a,
                    const integer* lda, double* b, const integer* ldb, double* c,
                    double* d, double* x, double* work, integer* lwork,
                    integer *info );

void FC_FUNC(cgglse,CGGLSE)( const integer* m, const integer* n, integer* p, scomplex* a,
                    const integer* lda, scomplex* b, const integer* ldb, scomplex* c,
                    scomplex* d, scomplex* x, scomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(zgglse,ZGGLSE)( const integer* m, const integer* n, integer* p, dcomplex* a,
                    const integer* lda, dcomplex* b, const integer* ldb, dcomplex* c,
                    dcomplex* d, dcomplex* x, dcomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(sggglm,SGGGLM)( const integer* n, const integer* m, integer* p, float* a,
                    const integer* lda, float* b, const integer* ldb, float* d,
                    float* x, float* y, float* work, integer* lwork,
                    integer *info );

void FC_FUNC(dggglm,DGGGLM)( const integer* n, const integer* m, integer* p, double* a,
                    const integer* lda, double* b, const integer* ldb, double* d,
                    double* x, double* y, double* work, integer* lwork,
                    integer *info );

void FC_FUNC(cggglm,CGGGLM)( const integer* n, const integer* m, integer* p, scomplex* a,
                    const integer* lda, scomplex* b, const integer* ldb, scomplex* d,
                    scomplex* x, scomplex* y, scomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(zggglm,ZGGGLM)( const integer* n, const integer* m, integer* p, dcomplex* a,
                    const integer* lda, dcomplex* b, const integer* ldb, dcomplex* d,
                    dcomplex* x, dcomplex* y, dcomplex* work, integer* lwork,
                    integer *info );

void FC_FUNC(ssyev,SSYEV)( char* jobz, const char* uplo, const integer* n, float* a,
                   const integer* lda, float* w, float* work, integer* lwork,
                   integer *info );

void FC_FUNC(dsyev,DSYEV)( char* jobz, const char* uplo, const integer* n, double* a,
                   const integer* lda, double* w, double* work, integer* lwork,
                   integer *info );

void FC_FUNC(cheev,CHEEV)( char* jobz, const char* uplo, const integer* n, scomplex* a,
                   const integer* lda, float* w, scomplex* work, integer* lwork,
                   float* rwork, integer *info );

void FC_FUNC(zheev,ZHEEV)( char* jobz, const char* uplo, const integer* n, dcomplex* a,
                   const integer* lda, double* w, dcomplex* work, integer* lwork,
                   double* rwork, integer *info );

void FC_FUNC(ssyevd,SSYEVD)( char* jobz, const char* uplo, const integer* n, float* a,
                    const integer* lda, float* w, float* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(dsyevd,DSYEVD)( char* jobz, const char* uplo, const integer* n, double* a,
                    const integer* lda, double* w, double* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(cheevd,CHEEVD)( char* jobz, const char* uplo, const integer* n, scomplex* a,
                    const integer* lda, float* w, scomplex* work, integer* lwork,
                    float* rwork, integer* lrwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(zheevd,ZHEEVD)( char* jobz, const char* uplo, const integer* n, dcomplex* a,
                    const integer* lda, double* w, dcomplex* work, integer* lwork,
                    double* rwork, integer* lrwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(ssyevx,SSYEVX)( char* jobz, char* range, const char* uplo, const integer* n,
                    float* a, const integer* lda, float* vl, float* vu,
                    integer* il, integer* iu, float* abstol,
                    const integer* m, float* w, float* z, const integer* ldz,
                    float* work, integer* lwork, integer* iwork,
                    integer* ifail, integer *info );

void FC_FUNC(dsyevx,DSYEVX)( char* jobz, char* range, const char* uplo, const integer* n,
                    double* a, const integer* lda, double* vl, double* vu,
                    integer* il, integer* iu, double* abstol,
                    const integer* m, double* w, double* z, const integer* ldz,
                    double* work, integer* lwork, integer* iwork,
                    integer* ifail, integer *info );

void FC_FUNC(cheevx,CHEEVX)( char* jobz, char* range, const char* uplo, const integer* n,
                    scomplex* a, const integer* lda, float* vl, float* vu,
                    integer* il, integer* iu, float* abstol,
                    const integer* m, float* w, scomplex* z, const integer* ldz,
                    scomplex* work, integer* lwork, float* rwork, integer* iwork,
                    integer* ifail, integer *info );

void FC_FUNC(zheevx,ZHEEVX)( char* jobz, char* range, const char* uplo, const integer* n,
                    dcomplex* a, const integer* lda, double* vl, double* vu,
                    integer* il, integer* iu, double* abstol,
                    const integer* m, double* w, dcomplex* z, const integer* ldz,
                    dcomplex* work, integer* lwork, double* rwork, integer* iwork,
                    integer* ifail, integer *info );

void FC_FUNC(ssyevr,SSYEVR)( char* jobz, char* range, const char* uplo, const integer* n,
                    float* a, const integer* lda, float* vl, float* vu,
                    integer* il, integer* iu, float* abstol,
                    const integer* m, float* w, float* z, const integer* ldz,
                    integer* isuppz, float* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(dsyevr,DSYEVR)( char* jobz, char* range, const char* uplo, const integer* n,
                    double* a, const integer* lda, double* vl, double* vu,
                    integer* il, integer* iu, double* abstol,
                    const integer* m, double* w, double* z, const integer* ldz,
                    integer* isuppz, double* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(cheevr,CHEEVR)( char* jobz, char* range, const char* uplo, const integer* n,
                    scomplex* a, const integer* lda, float* vl, float* vu,
                    integer* il, integer* iu, float* abstol,
                    const integer* m, float* w, scomplex* z, const integer* ldz,
                    integer* isuppz, scomplex* work, integer* lwork,
                    float* rwork, integer* lrwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(zheevr,ZHEEVR)( char* jobz, char* range, const char* uplo, const integer* n,
                    dcomplex* a, const integer* lda, double* vl, double* vu,
                    integer* il, integer* iu, double* abstol,
                    const integer* m, double* w, dcomplex* z, const integer* ldz,
                    integer* isuppz, dcomplex* work, integer* lwork,
                    double* rwork, integer* lrwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(sspev,SSPEV)( char* jobz, const char* uplo, const integer* n, float* ap, float* w,
                    float* z, const integer* ldz, float* work, integer *info );

void FC_FUNC(dspev,DSPEV)( char* jobz, const char* uplo, const integer* n, double* ap, double* w,
                   double* z, const integer* ldz, double* work, integer *info );

void FC_FUNC(chpev,CHPEV)( char* jobz, const char* uplo, const integer* n, scomplex* ap, float* w,
                   scomplex* z, const integer* ldz, scomplex* work, float* rwork, integer *info );

void FC_FUNC(zhpev,ZHPEV)( char* jobz, const char* uplo, const integer* n, dcomplex* ap, double* w,
                   dcomplex* z, const integer* ldz, dcomplex* work, double* rwork, integer *info );

void FC_FUNC(sspevd,SSPEVD)( char* jobz, const char* uplo, const integer* n, float* ap,
                    float* w, float* z, const integer* ldz, float* work,
                    integer* lwork, integer* iwork, integer* liwork,
                    integer *info );

void FC_FUNC(dspevd,DSPEVD)( char* jobz, const char* uplo, const integer* n, double* ap,
                    double* w, double* z, const integer* ldz, double* work,
                    integer* lwork, integer* iwork, integer* liwork,
                    integer *info );

void FC_FUNC(chpevd,CHPEVD)( char* jobz, const char* uplo, const integer* n, scomplex* ap,
                    float* w, scomplex* z, const integer* ldz, scomplex* work,
                    integer* lwork, float* rwork, integer* lrwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(zhpevd,ZHPEVD)( char* jobz, const char* uplo, const integer* n, dcomplex* ap,
                    double* w, dcomplex* z, const integer* ldz, dcomplex* work,
                    integer* lwork, double* rwork, integer* lrwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(sspevx,SSPEVX)( char* jobz, char* range, const char* uplo, const integer* n,
                    float* ap, float* vl, float* vu, integer* il,
                    integer* iu, float* abstol, const integer* m, float* w,
                    float* z, const integer* ldz, float* work, integer* iwork,
                    integer* ifail, integer *info );

void FC_FUNC(dspevx,DSPEVX)( char* jobz, char* range, const char* uplo, const integer* n,
                    double* ap, double* vl, double* vu, integer* il,
                    integer* iu, double* abstol, const integer* m, double* w,
                    double* z, const integer* ldz, double* work, integer* iwork,
                    integer* ifail, integer *info );

void FC_FUNC(chpevx,CHPEVX)( char* jobz, char* range, const char* uplo, const integer* n,
                    scomplex* ap, float* vl, float* vu, integer* il,
                    integer* iu, float* abstol, const integer* m, float* w,
                    scomplex* z, const integer* ldz, scomplex* work, float* rwork,
                    integer* iwork, integer* ifail, integer *info );

void FC_FUNC(zhpevx,ZHPEVX)( char* jobz, char* range, const char* uplo, const integer* n,
                    dcomplex* ap, double* vl, double* vu, integer* il,
                    integer* iu, double* abstol, const integer* m, double* w,
                    dcomplex* z, const integer* ldz, dcomplex* work, double* rwork,
                    integer* iwork, integer* ifail, integer *info );

void FC_FUNC(ssbev,SSBEV)( char* jobz, const char* uplo, const integer* n, const integer* kd,
                   float* ab, const integer* ldab, float* w, float* z,
                   const integer* ldz, float* work, integer *info );

void FC_FUNC(dsbev,DSBEV)( char* jobz, const char* uplo, const integer* n, const integer* kd,
                   double* ab, const integer* ldab, double* w, double* z,
                   const integer* ldz, double* work, integer *info );

void FC_FUNC(chbev,CHBEV)( char* jobz, const char* uplo, const integer* n, const integer* kd,
                    scomplex* ab, const integer* ldab, float* w, scomplex* z,
                   const integer* ldz, scomplex* work, float* rwork, integer *info );

void FC_FUNC(zhbev,ZHBEV)( char* jobz, const char* uplo, const integer* n, const integer* kd,
                    dcomplex* ab, const integer* ldab, double* w, dcomplex* z,
                   const integer* ldz, dcomplex* work, double* rwork, integer *info );

void FC_FUNC(ssbevd,SSBEVD)( char* jobz, const char* uplo, const integer* n, const integer* kd,
                    float* ab, const integer* ldab, float* w, float* z,
                    const integer* ldz, float* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(dsbevd,DSBEVD)( char* jobz, const char* uplo, const integer* n, const integer* kd,
                    double* ab, const integer* ldab, double* w, double* z,
                    const integer* ldz, double* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(chbevd,CHBEVD)( char* jobz, const char* uplo, const integer* n, const integer* kd,
                    scomplex* ab, const integer* ldab, float* w, scomplex* z,
                    const integer* ldz, scomplex* work, integer* lwork,
                    float* rwork, integer* lrwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(zhbevd,ZHBEVD)( char* jobz, const char* uplo, const integer* n, const integer* kd,
                    dcomplex* ab, const integer* ldab, double* w, dcomplex* z,
                    const integer* ldz, dcomplex* work, integer* lwork,
                    double* rwork, integer* lrwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(ssbevx,SSBEVX)( char* jobz, char* range, const char* uplo, const integer* n,
                    const integer* kd, float* ab, const integer* ldab, float* q,
                    const integer* ldq, float* vl, float* vu, integer* il,
                    integer* iu, float* abstol, const integer* m, float* w,
                    float* z, const integer* ldz, float* work, integer* iwork,
                    integer* ifail, integer *info );

void FC_FUNC(dsbevx,DSBEVX)( char* jobz, char* range, const char* uplo, const integer* n,
                    const integer* kd, double* ab, const integer* ldab, double* q,
                    const integer* ldq, double* vl, double* vu, integer* il,
                    integer* iu, double* abstol, const integer* m, double* w,
                    double* z, const integer* ldz, double* work, integer* iwork,
                    integer* ifail, integer *info );

void FC_FUNC(chbevx,CHBEVX)( char* jobz, char* range, const char* uplo, const integer* n,
                    const integer* kd, scomplex* ab, const integer* ldab, scomplex* q,
                    const integer* ldq, float* vl, float* vu, integer* il,
                    integer* iu, float* abstol, const integer* m, float* w,
                    scomplex* z, const integer* ldz, scomplex* work, float* rwork,
                    integer* iwork, integer* ifail, integer *info );

void FC_FUNC(zhbevx,ZHBEVX)( char* jobz, char* range, const char* uplo, const integer* n,
                    const integer* kd, dcomplex* ab, const integer* ldab, dcomplex* q,
                    const integer* ldq, double* vl, double* vu, integer* il,
                    integer* iu, double* abstol, const integer* m, double* w,
                    dcomplex* z, const integer* ldz, dcomplex* work, double* rwork,
                    integer* iwork, integer* ifail, integer *info );

void FC_FUNC(sstev,SSTEV)( char* jobz, const integer* n, float* d, float* e, float* z,
                   const integer* ldz, float* work, integer *info );

void FC_FUNC(dstev,DSTEV)( char* jobz, const integer* n, double* d, double* e, double* z,
                   const integer* ldz, double* work, integer *info );

void FC_FUNC(sstevd,SSTEVD)( char* jobz, const integer* n, float* d, float* e, float* z,
                    const integer* ldz, float* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(dstevd,DSTEVD)( char* jobz, const integer* n, double* d, double* e, double* z,
                    const integer* ldz, double* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(sstevx,SSTEVX)( char* jobz, char* range, const integer* n, float* d,
                    float* e, float* vl, float* vu, integer* il,
                    integer* iu, float* abstol, const integer* m, float* w,
                    float* z, const integer* ldz, float* work, integer* iwork,
                    integer* ifail, integer *info );

void FC_FUNC(dstevx,DSTEVX)( char* jobz, char* range, const integer* n, double* d,
                    double* e, double* vl, double* vu, integer* il,
                    integer* iu, double* abstol, const integer* m, double* w,
                    double* z, const integer* ldz, double* work, integer* iwork,
                    integer* ifail, integer *info );

void FC_FUNC(sstevr,SSTEVR)( char* jobz, char* range, const integer* n, float* d,
                    float* e, float* vl, float* vu, integer* il,
                    integer* iu, float* abstol, const integer* m, float* w,
                    float* z, const integer* ldz, integer* isuppz,
                    float* work, integer* lwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(dstevr,DSTEVR)( char* jobz, char* range, const integer* n, double* d,
                    double* e, double* vl, double* vu, integer* il,
                    integer* iu, double* abstol, const integer* m, double* w,
                    double* z, const integer* ldz, integer* isuppz,
                    double* work, integer* lwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(sgeev,SGEEV)( char* jobvl, char* jobvr, const integer* n, float* a,
                   const integer* lda, float* wr, float* wi, float* vl,
                   const integer* ldvl, float* vr, const integer* ldvr, float* work,
                   integer* lwork, integer *info );

void FC_FUNC(dgeev,DGEEV)( char* jobvl, char* jobvr, const integer* n, double* a,
                   const integer* lda, double* wr, double* wi, double* vl,
                   const integer* ldvl, double* vr, const integer* ldvr, double* work,
                   integer* lwork, integer *info );

void FC_FUNC(cgeev,CGEEV)( char* jobvl, char* jobvr, const integer* n, scomplex* a,
                   const integer* lda, scomplex* w, scomplex* vl,
                   const integer* ldvl, scomplex* vr, const integer* ldvr, scomplex* work,
                   integer* lwork, float* rwork, integer *info );

void FC_FUNC(zgeev,ZGEEV)( char* jobvl, char* jobvr, const integer* n, dcomplex* a,
                   const integer* lda, dcomplex* w, dcomplex* vl,
                   const integer* ldvl, dcomplex* vr, const integer* ldvr, dcomplex* work,
                   integer* lwork, double* rwork, integer *info );

void FC_FUNC(sgeevx,SGEEVX)( char* balanc, char* jobvl, char* jobvr, char* sense,
                    const integer* n, float* a, const integer* lda, float* wr,
                    float* wi, float* vl, const integer* ldvl, float* vr,
                    const integer* ldvr, integer* ilo, integer* ihi,
                    float* scale, float* abnrm, float* rconde,
                    float* rcondv, float* work, integer* lwork,
                    integer* iwork, integer *info );

void FC_FUNC(dgeevx,DGEEVX)( char* balanc, char* jobvl, char* jobvr, char* sense,
                    const integer* n, double* a, const integer* lda, double* wr,
                    double* wi, double* vl, const integer* ldvl, double* vr,
                    const integer* ldvr, integer* ilo, integer* ihi,
                    double* scale, double* abnrm, double* rconde,
                    double* rcondv, double* work, integer* lwork,
                    integer* iwork, integer *info );

void FC_FUNC(cgeevx,CGEEVX)( char* balanc, char* jobvl, char* jobvr, char* sense,
                    const integer* n, scomplex* a, const integer* lda, scomplex* w,
                    scomplex* vl, const integer* ldvl, scomplex* vr,
                    const integer* ldvr, integer* ilo, integer* ihi,
                    float* scale, float* abnrm, float* rconde,
                    float* rcondv, scomplex* work, integer* lwork,
                    float* rwork, integer *info );

void FC_FUNC(zgeevx,ZGEEVX)( char* balanc, char* jobvl, char* jobvr, char* sense,
                    const integer* n, dcomplex* a, const integer* lda, dcomplex* w,
                    dcomplex* vl, const integer* ldvl, dcomplex* vr,
                    const integer* ldvr, integer* ilo, integer* ihi,
                    double* scale, double* abnrm, double* rconde,
                    double* rcondv, dcomplex* work, integer* lwork,
                    double* rwork, integer *info );

void FC_FUNC(sgesvd,SGESVD)( char* jobu, char* jobvt, const integer* m, const integer* n,
                    float* a, const integer* lda, float* s, float* u,
                    const integer* ldu, float* vt, const integer* ldvt, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dgesvd,DGESVD)( char* jobu, char* jobvt, const integer* m, const integer* n,
                    double* a, const integer* lda, double* s, double* u,
                    const integer* ldu, double* vt, const integer* ldvt, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(cgesvd,CGESVD)( char* jobu, char* jobvt, const integer* m, const integer* n,
                    scomplex* a, const integer* lda, float* s, scomplex* u,
                    const integer* ldu, scomplex* vt, const integer* ldvt, scomplex* work,
                    integer* lwork, float* rwork, integer *info );

void FC_FUNC(zgesvd,ZGESVD)( char* jobu, char* jobvt, const integer* m, const integer* n,
                    dcomplex* a, const integer* lda, double* s, dcomplex* u,
                    const integer* ldu, dcomplex* vt, const integer* ldvt, dcomplex* work,
                    integer* lwork, double* rwork, integer *info );

void FC_FUNC(sgesdd,SGESDD)( char* jobz, const integer* m, const integer* n, float* a,
                    const integer* lda, float* s, float* u, const integer* ldu,
                    float* vt, const integer* ldvt, float* work,
                    integer* lwork, integer* iwork, integer *info );

void FC_FUNC(dgesdd,DGESDD)( char* jobz, const integer* m, const integer* n, double* a,
                    const integer* lda, double* s, double* u, const integer* ldu,
                    double* vt, const integer* ldvt, double* work,
                    integer* lwork, integer* iwork, integer *info );

void FC_FUNC(cgesdd,CGESDD)( char* jobz, const integer* m, const integer* n, scomplex* a,
                    const integer* lda, float* s, scomplex* u, const integer* ldu,
                    scomplex* vt, const integer* ldvt, scomplex* work,
                    integer* lwork, float* rwork, integer* iwork, integer *info );

void FC_FUNC(zgesdd,ZGESDD)( char* jobz, const integer* m, const integer* n, dcomplex* a,
                    const integer* lda, double* s, dcomplex* u, const integer* ldu,
                    dcomplex* vt, const integer* ldvt, dcomplex* work,
                    integer* lwork, double* rwork, integer* iwork, integer *info );

void FC_FUNC(sgejsv,SGEJSV)( char* joba, char* jobu, char* jobv, char* jobr, char* jobt,
                    char* jobp, const integer* m, const integer* n, float* a,
                    const integer* lda, float* sva, float* u, const integer* ldu,
                    float* v, const integer* ldv, float* work, integer* lwork,
                    integer* iwork, integer *info );

void FC_FUNC(dgejsv,DGEJSV)( char* joba, char* jobu, char* jobv, char* jobr, char* jobt,
                    char* jobp, const integer* m, const integer* n, double* a,
                    const integer* lda, double* sva, double* u, const integer* ldu,
                    double* v, const integer* ldv, double* work, integer* lwork,
                    integer* iwork, integer *info );

void FC_FUNC(sgesvj,SGESVJ)( char* joba, char* jobu, char* jobv, const integer* m,
                    const integer* n, float* a, const integer* lda, float* sva,
                    const integer* mv, float* v, const integer* ldv, float* work,
                    integer* lwork, integer *info );

void FC_FUNC(dgesvj,DGESVJ)( char* joba, char* jobu, char* jobv, const integer* m,
                    const integer* n, double* a, const integer* lda, double* sva,
                    const integer* mv, double* v, const integer* ldv, double* work,
                    integer* lwork, integer *info );

void FC_FUNC(sggsvd,SGGSVD)( char* jobu, char* jobv, char* jobq, const integer* m,
                    const integer* n, integer* p, const integer* k, integer* l,
                    float* a, const integer* lda, float* b, const integer* ldb,
                    float* alpha, float* beta, float* u, const integer* ldu,
                    float* v, const integer* ldv, float* q, const integer* ldq,
                    float* work, integer* iwork, integer *info );

void FC_FUNC(dggsvd,DGGSVD)( char* jobu, char* jobv, char* jobq, const integer* m,
                    const integer* n, integer* p, const integer* k, integer* l,
                    double* a, const integer* lda, double* b, const integer* ldb,
                    double* alpha, double* beta, double* u, const integer* ldu,
                    double* v, const integer* ldv, double* q, const integer* ldq,
                    double* work, integer* iwork, integer *info );

void FC_FUNC(cggsvd,CGGSVD)( char* jobu, char* jobv, char* jobq, const integer* m,
                    const integer* n, integer* p, const integer* k, integer* l,
                    scomplex* a, const integer* lda, scomplex* b, const integer* ldb,
                    float* alpha, float* beta, scomplex* u, const integer* ldu,
                    scomplex* v, const integer* ldv, scomplex* q, const integer* ldq,
                    scomplex* work, float* rwork, integer* iwork, integer *info );

void FC_FUNC(zggsvd,ZGGSVD)( char* jobu, char* jobv, char* jobq, const integer* m,
                    const integer* n, integer* p, const integer* k, integer* l,
                    dcomplex* a, const integer* lda, dcomplex* b, const integer* ldb,
                    double* alpha, double* beta, dcomplex* u, const integer* ldu,
                    dcomplex* v, const integer* ldv, dcomplex* q, const integer* ldq,
                    dcomplex* work, double* rwork, integer* iwork, integer *info );

void FC_FUNC(ssygv,SSYGV)( integer* itype, char* jobz, const char* uplo, const integer* n,
                   float* a, const integer* lda, float* b, const integer* ldb,
                   float* w, float* work, integer* lwork,
                   integer *info );

void FC_FUNC(dsygv,DSYGV)( integer* itype, char* jobz, const char* uplo, const integer* n,
                   double* a, const integer* lda, double* b, const integer* ldb,
                   double* w, double* work, integer* lwork,
                   integer *info );

void FC_FUNC(chegv,CHEGV)( integer* itype, char* jobz, const char* uplo, const integer* n,
                   scomplex* a, const integer* lda, scomplex* b, const integer* ldb,
                   float* w, scomplex* work, integer* lwork,
                   float* rwork, integer *info );

void FC_FUNC(zhegv,ZHEGV)( integer* itype, char* jobz, const char* uplo, const integer* n,
                   dcomplex* a, const integer* lda, dcomplex* b, const integer* ldb,
                   double* w, dcomplex* work, integer* lwork,
                   double* rwork, integer *info );

void FC_FUNC(ssygvd,SSYGVD)( integer* itype, char* jobz, const char* uplo, const integer* n,
                    float* a, const integer* lda, float* b, const integer* ldb,
                    float* w, float* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(dsygvd,DSYGVD)( integer* itype, char* jobz, const char* uplo, const integer* n,
                    double* a, const integer* lda, double* b, const integer* ldb,
                    double* w, double* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(chegvd,CHEGVD)( integer* itype, char* jobz, const char* uplo, const integer* n,
                    scomplex* a, const integer* lda, scomplex* b, const integer* ldb,
                    float* w, scomplex* work, integer* lwork,
                    float* rwork, integer* lrwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(zhegvd,ZHEGVD)( integer* itype, char* jobz, const char* uplo, const integer* n,
                    dcomplex* a, const integer* lda, dcomplex* b, const integer* ldb,
                    double* w, dcomplex* work, integer* lwork,
                    double* rwork, integer* lrwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(ssygvx,SSYGVX)( integer* itype, char* jobz, char* range, const char* uplo,
                    const integer* n, float* a, const integer* lda, float* b,
                    const integer* ldb, float* vl, float* vu, integer* il,
                    integer* iu, float* abstol, const integer* m, float* w,
                    float* z, const integer* ldz, float* work, integer* lwork,
                    integer* iwork, integer* ifail, integer *info );

void FC_FUNC(dsygvx,DSYGVX)( integer* itype, char* jobz, char* range, const char* uplo,
                    const integer* n, double* a, const integer* lda, double* b,
                    const integer* ldb, double* vl, double* vu, integer* il,
                    integer* iu, double* abstol, const integer* m, double* w,
                    double* z, const integer* ldz, double* work, integer* lwork,
                    integer* iwork, integer* ifail, integer *info );

void FC_FUNC(chegvx,CHEGVX)( integer* itype, char* jobz, char* range, const char* uplo,
                    const integer* n, scomplex* a, const integer* lda, scomplex* b,
                    const integer* ldb, float* vl, float* vu, integer* il,
                    integer* iu, float* abstol, const integer* m, float* w,
                    scomplex* z, const integer* ldz, scomplex* work, integer* lwork,
                    float* rwork, integer* iwork, integer* ifail, integer *info );

void FC_FUNC(zhegvx,ZHEGVX)( integer* itype, char* jobz, char* range, const char* uplo,
                    const integer* n, dcomplex* a, const integer* lda, dcomplex* b,
                    const integer* ldb, double* vl, double* vu, integer* il,
                    integer* iu, double* abstol, const integer* m, double* w,
                    dcomplex* z, const integer* ldz, dcomplex* work, integer* lwork,
                    double* rwork, integer* iwork, integer* ifail, integer *info );

void FC_FUNC(sspgv,SSPGV)( integer* itype, char* jobz, const char* uplo, const integer* n,
                   float* ap, float* bp, float* w, float* z,
                   const integer* ldz, float* work, integer *info );

void FC_FUNC(dspgv,DSPGV)( integer* itype, char* jobz, const char* uplo, const integer* n,
                   double* ap, double* bp, double* w, double* z,
                   const integer* ldz, double* work, integer *info );

void FC_FUNC(chpgv,CHPGV)( integer* itype, char* jobz, const char* uplo, const integer* n,
                   scomplex* ap, scomplex* bp, float* w, scomplex* z,
                   const integer* ldz, scomplex* work, float* rwork, integer *info );

void FC_FUNC(zhpgv,ZHPGV)( integer* itype, char* jobz, const char* uplo, const integer* n,
                   dcomplex* ap, dcomplex* bp, double* w, dcomplex* z,
                   const integer* ldz, dcomplex* work, double* rwork, integer *info );

void FC_FUNC(sspgvd,SSPGVD)( integer* itype, char* jobz, const char* uplo, const integer* n,
                    float* ap, float* bp, float* w, float* z,
                    const integer* ldz, float* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(dspgvd,DSPGVD)( integer* itype, char* jobz, const char* uplo, const integer* n,
                    double* ap, double* bp, double* w, double* z,
                    const integer* ldz, double* work, integer* lwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(chpgvd,CHPGVD)( integer* itype, char* jobz, const char* uplo, const integer* n,
                    scomplex* ap, scomplex* bp, float* w, scomplex* z,
                    const integer* ldz, scomplex* work, integer* lwork,
                    float* rwork, integer* lrwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(zhpgvd,ZHPGVD)( integer* itype, char* jobz, const char* uplo, const integer* n,
                    dcomplex* ap, dcomplex* bp, double* w, dcomplex* z,
                    const integer* ldz, dcomplex* work, integer* lwork,
                    double* rwork, integer* lrwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(sspgvx,SSPGVX)( integer* itype, char* jobz, char* range, const char* uplo,
                    const integer* n, float* ap, float* bp, float* vl,
                    float* vu, integer* il, integer* iu, float* abstol,
                    const integer* m, float* w, float* z, const integer* ldz,
                    float* work, integer* iwork, integer* ifail,
                    integer *info );

void FC_FUNC(dspgvx,DSPGVX)( integer* itype, char* jobz, char* range, const char* uplo,
                    const integer* n, double* ap, double* bp, double* vl,
                    double* vu, integer* il, integer* iu, double* abstol,
                    const integer* m, double* w, double* z, const integer* ldz,
                    double* work, integer* iwork, integer* ifail,
                    integer *info );

void FC_FUNC(chpgvx,CHPGVX)( integer* itype, char* jobz, char* range, const char* uplo,
                    const integer* n, scomplex* ap, scomplex* bp, float* vl,
                    float* vu, integer* il, integer* iu, float* abstol,
                    const integer* m, float* w, scomplex* z, const integer* ldz,
                    scomplex* work, float* rwork, integer* iwork, integer* ifail,
                    integer *info );

void FC_FUNC(zhpgvx,ZHPGVX)( integer* itype, char* jobz, char* range, const char* uplo,
                    const integer* n, dcomplex* ap, dcomplex* bp, double* vl,
                    double* vu, integer* il, integer* iu, double* abstol,
                    const integer* m, double* w, dcomplex* z, const integer* ldz,
                    dcomplex* work, double* rwork, integer* iwork, integer* ifail,
                    integer *info );

void FC_FUNC(ssbgv,SSBGV)( char* jobz, const char* uplo, const integer* n, const integer* ka,
                   const integer* kb, float* ab, const integer* ldab, float* bb,
                   const integer* ldbb, float* w, float* z, const integer* ldz,
                   float* work, integer *info );

void FC_FUNC(dsbgv,DSBGV)( char* jobz, const char* uplo, const integer* n, const integer* ka,
                   const integer* kb, double* ab, const integer* ldab, double* bb,
                   const integer* ldbb, double* w, double* z, const integer* ldz,
                   double* work, integer *info );

void FC_FUNC(chbgv,CHBGV)( char* jobz, const char* uplo, const integer* n, const integer* ka,
                   const integer* kb, scomplex* ab, const integer* ldab, scomplex* bb,
                   const integer* ldbb, float* w, scomplex* z, const integer* ldz,
                   scomplex* work, float* rwork, integer *info );

void FC_FUNC(zhbgv,ZHBGV)( char* jobz, const char* uplo, const integer* n, const integer* ka,
                   const integer* kb, dcomplex* ab, const integer* ldab, dcomplex* bb,
                   const integer* ldbb, double* w, dcomplex* z, const integer* ldz,
                   dcomplex* work, double* rwork, integer *info );

void FC_FUNC(ssbgvd,SSBGVD)( char* jobz, const char* uplo, const integer* n, const integer* ka,
                    const integer* kb, float* ab, const integer* ldab, float* bb,
                    const integer* ldbb, float* w, float* z, const integer* ldz,
                    float* work, integer* lwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(dsbgvd,DSBGVD)( char* jobz, const char* uplo, const integer* n, const integer* ka,
                    const integer* kb, double* ab, const integer* ldab, double* bb,
                    const integer* ldbb, double* w, double* z, const integer* ldz,
                    double* work, integer* lwork, integer* iwork,
                    integer* liwork, integer *info );

void FC_FUNC(chbgvd,CHBGVD)( char* jobz, const char* uplo, const integer* n, const integer* ka,
                    const integer* kb, scomplex* ab, const integer* ldab, scomplex* bb,
                    const integer* ldbb, float* w, scomplex* z, const integer* ldz,
                    scomplex* work, integer* lwork, float* rwork, integer* lrwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(zhbgvd,ZHBGVD)( char* jobz, const char* uplo, const integer* n, const integer* ka,
                    const integer* kb, dcomplex* ab, const integer* ldab, dcomplex* bb,
                    const integer* ldbb, double* w, dcomplex* z, const integer* ldz,
                    dcomplex* work, integer* lwork, double* rwork, integer* lrwork,
                    integer* iwork, integer* liwork, integer *info );

void FC_FUNC(ssbgvx,SSBGVX)( char* jobz, char* range, const char* uplo, const integer* n,
                    const integer* ka, const integer* kb, float* ab,
                    const integer* ldab, float* bb, const integer* ldbb, float* q,
                    const integer* ldq, float* vl, float* vu, integer* il,
                    integer* iu, float* abstol, const integer* m, float* w,
                    float* z, const integer* ldz, float* work, integer* iwork,
                    integer* ifail, integer *info );

void FC_FUNC(dsbgvx,DSBGVX)( char* jobz, char* range, const char* uplo, const integer* n,
                    const integer* ka, const integer* kb, double* ab,
                    const integer* ldab, double* bb, const integer* ldbb, double* q,
                    const integer* ldq, double* vl, double* vu, integer* il,
                    integer* iu, double* abstol, const integer* m, double* w,
                    double* z, const integer* ldz, double* work, integer* iwork,
                    integer* ifail, integer *info );

void FC_FUNC(chbgvx,CHBGVX)( char* jobz, char* range, const char* uplo, const integer* n,
                    const integer* ka, const integer* kb, scomplex* ab,
                    const integer* ldab, scomplex* bb, const integer* ldbb, scomplex* q,
                    const integer* ldq, float* vl, float* vu, integer* il,
                    integer* iu, float* abstol, const integer* m, float* w,
                    scomplex* z, const integer* ldz, scomplex* work, float* rwork,
                    integer* iwork, integer* ifail, integer *info );

void FC_FUNC(zhbgvx,ZHBGVX)( char* jobz, char* range, const char* uplo, const integer* n,
                    const integer* ka, const integer* kb, dcomplex* ab,
                    const integer* ldab, dcomplex* bb, const integer* ldbb, dcomplex* q,
                    const integer* ldq, double* vl, double* vu, integer* il,
                    integer* iu, double* abstol, const integer* m, double* w,
                    dcomplex* z, const integer* ldz, dcomplex* work, double* rwork,
                    integer* iwork, integer* ifail, integer *info );

void FC_FUNC(sggev,SGGEV)( char* jobvl, char* jobvr, const integer* n, float* a,
                   const integer* lda, float* b, const integer* ldb, float* alphar,
                   float* alphai, float* beta, float* vl, const integer* ldvl,
                   float* vr, const integer* ldvr, float* work,
                   integer* lwork, integer *info );

void FC_FUNC(dggev,DGGEV)( char* jobvl, char* jobvr, const integer* n, double* a,
                   const integer* lda, double* b, const integer* ldb, double* alphar,
                   double* alphai, double* beta, double* vl, const integer* ldvl,
                   double* vr, const integer* ldvr, double* work,
                   integer* lwork, integer *info );

void FC_FUNC(cggev,CGGEV)( char* jobvl, char* jobvr, const integer* n, scomplex* a,
                   const integer* lda, scomplex* b, const integer* ldb, scomplex* alpha,
                   scomplex* beta, scomplex* vl, const integer* ldvl,
                   scomplex* vr, const integer* ldvr, scomplex* work,
                   integer* lwork, float* rwork, integer *info );

void FC_FUNC(zggev,ZGGEV)( char* jobvl, char* jobvr, const integer* n, dcomplex* a,
                   const integer* lda, dcomplex* b, const integer* ldb, dcomplex* alpha,
                   dcomplex* beta, dcomplex* vl, const integer* ldvl,
                   dcomplex* vr, const integer* ldvr, dcomplex* work,
                   integer* lwork, double* rwork, integer *info );

void FC_FUNC(sggevx,SGGEVX)( char* balanc, char* jobvl, char* jobvr, char* sense,
                    const integer* n, float* a, const integer* lda, float* b,
                    const integer* ldb, float* alphar, float* alphai,
                    float* beta, float* vl, const integer* ldvl, float* vr,
                    const integer* ldvr, integer* ilo, integer* ihi,
                    float* lscale, float* rscale, float* abnrm,
                    float* bbnrm, float* rconde, float* rcondv, float* work,
                    integer* lwork, integer* iwork, logical* bwork,
                    integer *info );

void FC_FUNC(dggevx,DGGEVX)( char* balanc, char* jobvl, char* jobvr, char* sense,
                    const integer* n, double* a, const integer* lda, double* b,
                    const integer* ldb, double* alphar, double* alphai,
                    double* beta, double* vl, const integer* ldvl, double* vr,
                    const integer* ldvr, integer* ilo, integer* ihi,
                    double* lscale, double* rscale, double* abnrm,
                    double* bbnrm, double* rconde, double* rcondv, double* work,
                    integer* lwork, integer* iwork, logical* bwork,
                    integer *info );

void FC_FUNC(cggevx,CGGEVX)( char* balanc, char* jobvl, char* jobvr, char* sense,
                    const integer* n, scomplex* a, const integer* lda, scomplex* b,
                    const integer* ldb, scomplex* alpha,
                    scomplex* beta, scomplex* vl, const integer* ldvl, scomplex* vr,
                    const integer* ldvr, integer* ilo, integer* ihi,
                    float* lscale, float* rscale, float* abnrm,
                    float* bbnrm, float* rconde, float* rcondv, scomplex* work,
                    integer* lwork, float* rwork, integer* iwork, logical* bwork,
                    integer *info );

void FC_FUNC(zggevx,ZGGEVX)( char* balanc, char* jobvl, char* jobvr, char* sense,
                    const integer* n, dcomplex* a, const integer* lda, dcomplex* b,
                    const integer* ldb, dcomplex* alpha,
                    dcomplex* beta, dcomplex* vl, const integer* ldvl, dcomplex* vr,
                    const integer* ldvr, integer* ilo, integer* ihi,
                    double* lscale, double* rscale, double* abnrm,
                    double* bbnrm, double* rconde, double* rcondv, dcomplex* work,
                    integer* lwork, double* rwork, integer* iwork, logical* bwork,
                    integer *info );

void FC_FUNC(ssfrk,SSFRK)( const char* transr, const char* uplo, const char* trans, const integer* n,
                   const integer* k, float* alpha, const float* a,
                   const integer* lda, float* beta, float* c );

void FC_FUNC(dsfrk,DSFRK)( const char* transr, const char* uplo, const char* trans, const integer* n,
                   const integer* k, double* alpha, const double* a,
                   const integer* lda, double* beta, double* c );

void FC_FUNC(chfrk,CHFRK)( const char* transr, const char* uplo, const char* trans, const integer* n,
                   const integer* k, float* alpha, const scomplex* a,
                   const integer* lda, float* beta, scomplex* c );

void FC_FUNC(zhfrk,ZHFRK)( const char* transr, const char* uplo, const char* trans, const integer* n,
                   const integer* k, double* alpha, const dcomplex* a,
                   const integer* lda, double* beta, dcomplex* c );

void FC_FUNC(stfsm,STFSM)( const char* transr, char* side, const char* uplo, const char* trans,
                   char* diag, const integer* m, const integer* n, float* alpha,
                   const float* a, float* b, const integer* ldb );

void FC_FUNC(dtfsm,DTFSM)( const char* transr, char* side, const char* uplo, const char* trans,
                   char* diag, const integer* m, const integer* n, double* alpha,
                   const double* a, double* b, const integer* ldb );

void FC_FUNC(ctfsm,CTFSM)( const char* transr, char* side, const char* uplo, const char* trans,
                   char* diag, const integer* m, const integer* n, scomplex* alpha,
                   const scomplex* a, scomplex* b, const integer* ldb );

void FC_FUNC(ztfsm,ZTFSM)( const char* transr, char* side, const char* uplo, const char* trans,
                   char* diag, const integer* m, const integer* n, dcomplex* alpha,
                   const dcomplex* a, dcomplex* b, const integer* ldb );

void FC_FUNC(stfttp,STFTTP)( const char* transr, const char* uplo, const integer* n, const float* arf,
                    float* ap, integer *info );

void FC_FUNC(dtfttp,DTFTTP)( const char* transr, const char* uplo, const integer* n, const double* arf,
                    double* ap, integer *info );

void FC_FUNC(ctfttp,CTFTTP)( const char* transr, const char* uplo, const integer* n, const scomplex* arf,
                    scomplex* ap, integer *info );

void FC_FUNC(ztfttp,ZTFTTP)( const char* transr, const char* uplo, const integer* n, const dcomplex* arf,
                    dcomplex* ap, integer *info );

void FC_FUNC(stfttr,STFTTR)( const char* transr, const char* uplo, const integer* n, const float* arf,
                    float* a, const integer* lda, integer *info );

void FC_FUNC(dtfttr,DTFTTR)( const char* transr, const char* uplo, const integer* n, const double* arf,
                    double* a, const integer* lda, integer *info );

void FC_FUNC(ctfttr,CTFTTR)( const char* transr, const char* uplo, const integer* n, const scomplex* arf,
                    scomplex* a, const integer* lda, integer *info );

void FC_FUNC(ztfttr,ZTFTTR)( const char* transr, const char* uplo, const integer* n, const dcomplex* arf,
                    dcomplex* a, const integer* lda, integer *info );

void FC_FUNC(stpttf,STPTTF)( const char* transr, const char* uplo, const integer* n, const float* ap,
                    float* arf, integer *info );

void FC_FUNC(dtpttf,DTPTTF)( const char* transr, const char* uplo, const integer* n, const double* ap,
                    double* arf, integer *info );

void FC_FUNC(ctpttf,CTPTTF)( const char* transr, const char* uplo, const integer* n, const scomplex* ap,
                    scomplex* arf, integer *info );

void FC_FUNC(ztpttf,ZTPTTF)( const char* transr, const char* uplo, const integer* n, const dcomplex* ap,
                    dcomplex* arf, integer *info );

void FC_FUNC(stpttr,STPTTR)( const char* uplo, const integer* n, const float* ap, float* a,
                    const integer* lda, integer *info );

void FC_FUNC(dtpttr,DTPTTR)( const char* uplo, const integer* n, const double* ap, double* a,
                    const integer* lda, integer *info );

void FC_FUNC(ctpttr,CTPTTR)( const char* uplo, const integer* n, const scomplex* ap, scomplex* a,
                    const integer* lda, integer *info );

void FC_FUNC(ztpttr,ZTPTTR)( const char* uplo, const integer* n, const dcomplex* ap, dcomplex* a,
                    const integer* lda, integer *info );

void FC_FUNC(strttf,STRTTF)( const char* transr, const char* uplo, const integer* n, const float* a,
                    const integer* lda, float* arf, integer *info );

void FC_FUNC(dtrttf,DTRTTF)( const char* transr, const char* uplo, const integer* n, const double* a,
                    const integer* lda, double* arf, integer *info );

void FC_FUNC(ctrttf,CTRTTF)( const char* transr, const char* uplo, const integer* n, const scomplex* a,
                    const integer* lda, scomplex* arf, integer *info );

void FC_FUNC(ztrttf,ZTRTTF)( const char* transr, const char* uplo, const integer* n, const dcomplex* a,
                    const integer* lda, dcomplex* arf, integer *info );

void FC_FUNC(strttp,STRTTP)( const char* uplo, const integer* n, const float* a, const integer* lda,
                    float* ap, integer *info );

void FC_FUNC(dtrttp,DTRTTP)( const char* uplo, const integer* n, const double* a, const integer* lda,
                    double* ap, integer *info );

void FC_FUNC(ctrttp,CTRTTP)( const char* uplo, const integer* n, const scomplex* a, const integer* lda,
                    scomplex* ap, integer *info );

void FC_FUNC(ztrttp,ZTRTTP)( const char* uplo, const integer* n, const dcomplex* a, const integer* lda,
                    dcomplex* ap, integer *info );

void FC_FUNC(sgeqrfp,SGEQRFP)( const integer* m, const integer* n, float* a, const integer* lda,
                     float* tau, float* work, integer* lwork,
                     integer *info );

void FC_FUNC(dgeqrfp,DGEQRFP)( const integer* m, const integer* n, double* a, const integer* lda,
                     double* tau, double* work, integer* lwork,
                     integer *info );

void FC_FUNC(cgeqrfp,CGEQRFP)( const integer* m, const integer* n, scomplex* a, const integer* lda,
                     scomplex* tau, scomplex* work, integer* lwork,
                     integer *info );

void FC_FUNC(zgeqrfp,ZGEQRFP)( const integer* m, const integer* n, dcomplex* a, const integer* lda,
                     dcomplex* tau, dcomplex* work, integer* lwork,
                     integer *info );

#ifdef __cplusplus
}
#endif

#endif
