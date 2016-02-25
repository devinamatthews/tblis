#ifndef _TENSOR_UTIL_BLAS_H_
#define _TENSOR_UTIL_BLAS_H_

#include "fortran.hpp"

#ifdef __cplusplus
extern "C"
{
#endif

/******************************************************************************
 *
 * Level 1 BLAS, FORTRAN prototypes
 *
 *****************************************************************************/
void    FC_FUNC(srotg,SROTG)  (float* a, float* b, float* c, float* s);
void    FC_FUNC(srotmg,SROTMG)(float* d1, float* d2, float* a, const float* b, float* param);
void    FC_FUNC(srot,SROT)    (const integer* n,                           float* x, const integer* incx,       float* y, const integer* incy, const float* c, const float* s);
void    FC_FUNC(srotm,SROTM)  (const integer* n,                           float* x, const integer* incx,       float* y, const integer* incy, float* param);
void    FC_FUNC(sswap,SSWAP)  (const integer* n,                           float* x, const integer* incx,       float* y, const integer* incy);
void    FC_FUNC(sscal,SSCAL)  (const integer* n, const float* alpha,       float* x, const integer* incx);
void    FC_FUNC(scopy,SCOPY)  (const integer* n,                     const float* x, const integer* incx,       float* y, const integer* incy);
void    FC_FUNC(saxpy,SAXPY)  (const integer* n, const float* alpha, const float* x, const integer* incx,       float* y, const integer* incy);
float   FC_FUNC(sdot,SDOT)    (const integer* n,                     const float* x, const integer* incx, const float* y, const integer* incy);
float   FC_FUNC(snrm2,SNRM2)  (const integer* n,                     const float* x, const integer* incx);
float   FC_FUNC(sasum,SASUM)  (const integer* n,                     const float* x, const integer* incx);
integer FC_FUNC(isamax,ISAMAX)(const integer* n,                     const float* x, const integer* incx);

void    FC_FUNC(drotg,DROTG)  (double* a, double* b, double* c, double* s);
void    FC_FUNC(drotmg,DROTMG)(double* d1, double* d2, double* a, const double* b, double* param);
void    FC_FUNC(drot,DROT)    (const integer* n,                            double* x, const integer* incx,       double* y, const integer* incy, const double* c, const double* s);
void    FC_FUNC(drotm,DROTM)  (const integer* n,                            double* x, const integer* incx,       double* y, const integer* incy, double* param);
void    FC_FUNC(dswap,DSWAP)  (const integer* n,                            double* x, const integer* incx,       double* y, const integer* incy);
void    FC_FUNC(dscal,DSCAL)  (const integer* n, const double* alpha,       double* x, const integer* incx);
void    FC_FUNC(dcopy,DCOPY)  (const integer* n,                      const double* x, const integer* incx,       double* y, const integer* incy);
void    FC_FUNC(daxpy,DAXPY)  (const integer* n, const double* alpha, const double* x, const integer* incx,       double* y, const integer* incy);
double  FC_FUNC(ddot,DDOT)    (const integer* n,                      const double* x, const integer* incx, const double* y, const integer* incy);
double  FC_FUNC(dnrm2,DNRM2)  (const integer* n,                      const double* x, const integer* incx);
double  FC_FUNC(dasum,DASUM)  (const integer* n,                      const double* x, const integer* incx);
integer FC_FUNC(idamax,IDAMAX)(const integer* n,                      const double* x, const integer* incx);

void     FC_FUNC(crotg,CROTG)  (scomplex* a, scomplex* b, float* c, scomplex* s);
void     FC_FUNC(csrot,CSROT)  (const integer* n,                              scomplex* x, const integer* incx,       scomplex* y, const integer* incy, const float* c, const float* s);
void     FC_FUNC(cswap,CSWAP)  (const integer* n,                              scomplex* x, const integer* incx,       scomplex* y, const integer* incy);
void     FC_FUNC(cscal,CSCAL)  (const integer* n, const scomplex* alpha,       scomplex* x, const integer* incx);
void     FC_FUNC(csscal,CSSCAL)(const integer* n, const    float* alpha,       scomplex* x, const integer* incx);
void     FC_FUNC(ccopy,CCOPY)  (const integer* n,                        const scomplex* x, const integer* incx,       scomplex* y, const integer* incy);
void     FC_FUNC(caxpy,CAXPY)  (const integer* n, const scomplex* alpha, const scomplex* x, const integer* incx,       scomplex* y, const integer* incy);
scomplex FC_FUNC(cdotu,CDOTU)  (const integer* n,                        const scomplex* x, const integer* incx, const scomplex* y, const integer* incy);
scomplex FC_FUNC(cdotc,CDOTC)  (const integer* n,                        const scomplex* x, const integer* incx, const scomplex* y, const integer* incy);
float    FC_FUNC(scnrm2,SCNRM2)(const integer* n,                        const scomplex* x, const integer* incx);
float    FC_FUNC(scasum,SCASUM)(const integer* n,                        const scomplex* x, const integer* incx);
integer  FC_FUNC(icamax,ICAMAX)(const integer* n,                        const scomplex* x, const integer* incx);

void     FC_FUNC(zrotg,ZROTG)  (dcomplex* a, dcomplex* b, double* c, dcomplex* s);
void     FC_FUNC(zdrot,ZDROT)  (const integer* n,                              dcomplex* x, const integer* incx,       dcomplex* y, const integer* incy, const double* c, const double* s);
void     FC_FUNC(zswap,ZSWAP)  (const integer* n,                              dcomplex* x, const integer* incx,       dcomplex* y, const integer* incy);
void     FC_FUNC(zscal,ZSCAL)  (const integer* n, const dcomplex* alpha,       dcomplex* x, const integer* incx);
void     FC_FUNC(zdscal,ZDSCAL)(const integer* n, const   double* alpha,       dcomplex* x, const integer* incx);
void     FC_FUNC(zcopy,ZCOPY)  (const integer* n,                        const dcomplex* x, const integer* incx,       dcomplex* y, const integer* incy);
void     FC_FUNC(zaxpy,ZAXPY)  (const integer* n, const dcomplex* alpha, const dcomplex* x, const integer* incx,       dcomplex* y, const integer* incy);
dcomplex FC_FUNC(zdotu,ZDOTU)  (const integer* n,                        const dcomplex* x, const integer* incx, const dcomplex* y, const integer* incy);
dcomplex FC_FUNC(zdotc,ZDOTC)  (const integer* n,                        const dcomplex* x, const integer* incx, const dcomplex* y, const integer* incy);
double   FC_FUNC(dznrm2,DZNRM2)(const integer* n,                        const dcomplex* x, const integer* incx);
double   FC_FUNC(dzasum,DZASUM)(const integer* n,                        const dcomplex* x, const integer* incx);
integer  FC_FUNC(izamax,IZAMAX)(const integer* n,                        const dcomplex* x, const integer* incx);

/******************************************************************************
 *
 * Level 2 BLAS, FORTRAN prototypes
 *
 *****************************************************************************/
void FC_FUNC(sgemv,SGEMV)(                  const char* trans,                   const integer* m, const integer* n,                                       const float* alpha, const float* a, const integer* lda,  const float* x, const integer* incx, const float* beta, float* y, const integer* incy);
void FC_FUNC(sgbmv,SGBMV)(                  const char* trans,                   const integer* m, const integer* n, const integer* kl, const integer* ku, const float* alpha, const float* a, const integer* lda,  const float* x, const integer* incx, const float* beta, float* y, const integer* incy);
void FC_FUNC(ssymv,SSYMV)(const char* uplo,                                                        const integer* n,                                       const float* alpha, const float* a, const integer* lda,  const float* x, const integer* incx, const float* beta, float* y, const integer* incy);
void FC_FUNC(ssbmv,SSBMV)(const char* uplo,                                                        const integer* n, const integer* k,                     const float* alpha, const float* a, const integer* lda,  const float* x, const integer* incx, const float* beta, float* y, const integer* incy);
void FC_FUNC(sspmv,SSPMV)(const char* uplo,                                                        const integer* n,                                       const float* alpha, const float* ap,                     const float* x, const integer* incx, const float* beta, float* y, const integer* incy);
void FC_FUNC(strmv,STRMV)(const char* uplo, const char* trans, const char* diag,                   const integer* n,                                                           const float* a, const integer* lda,        float* x, const integer* incx);
void FC_FUNC(stbmv,STBMV)(const char* uplo, const char* trans, const char* diag,                   const integer* n, const integer* k,                                         const float* a, const integer* lda,        float* x, const integer* incx);
void FC_FUNC(stpmv,STPMV)(const char* uplo, const char* trans, const char* diag,                   const integer* n,                                                           const float* ap,                           float* x, const integer* incx);
void FC_FUNC(strsv,STRSV)(const char* uplo, const char* trans, const char* diag,                   const integer* n,                                                           const float* a, const integer* lda,        float* x, const integer* incx);
void FC_FUNC(stbsv,STBSV)(const char* uplo, const char* trans, const char* diag,                   const integer* n, const integer* k,                                         const float* a, const integer* lda,        float* x, const integer* incx);
void FC_FUNC(stpsv,STPSV)(const char* uplo, const char* trans, const char* diag,                   const integer* n,                                                           const float* ap,                           float* x, const integer* incx);
void FC_FUNC(sger,SGER)  (                                                       const integer* m, const integer* n,                                       const float* alpha, const float* x, const integer* incx, const float* y, const integer* incy, float* a, const integer* lda);
void FC_FUNC(ssyr,SSYR)  (const char* uplo,                                                        const integer* n,                                       const float* alpha, const float* x, const integer* incx,                                      float* a, const integer* lda);
void FC_FUNC(sspr,SSPR)  (const char* uplo,                                                        const integer* n,                                       const float* alpha, const float* x, const integer* incx,                                      float* ap);
void FC_FUNC(ssyr2,SSYR2)(const char* uplo,                                                        const integer* n,                                       const float* alpha, const float* x, const integer* incx, const float* y, const integer* incy, float* a, const integer* lda);
void FC_FUNC(sspr2,SSPR2)(const char* uplo,                                                        const integer* n,                                       const float* alpha, const float* x, const integer* incx, const float* y, const integer* incy, float* ap);

void FC_FUNC(dgemv,DGEMV)(                  const char* trans,                   const integer* m, const integer* n,                                       const double* alpha, const double* a, const integer* lda, const double* x, const integer* incx, const double* beta, double* y, const integer* incy);
void FC_FUNC(dgbmv,DGBMV)(                  const char* trans,                   const integer* m, const integer* n, const integer* kl, const integer* ku, const double* alpha, const double* a, const integer* lda, const double* x, const integer* incx, const double* beta, double* y, const integer* incy);
void FC_FUNC(dsymv,DSYMV)(const char* uplo,                                                        const integer* n,                                       const double* alpha, const double* a, const integer* lda, const double* x, const integer* incx, const double* beta, double* y, const integer* incy);
void FC_FUNC(dsbmv,DSBMV)(const char* uplo,                                                        const integer* n, const integer* k,                     const double* alpha, const double* a, const integer* lda, const double* x, const integer* incx, const double* beta, double* y, const integer* incy);
void FC_FUNC(dspmv,DSPMV)(const char* uplo,                                                        const integer* n,                                       const double* alpha, const double* ap,                    const double* x, const integer* incx, const double* beta, double* y, const integer* incy);
void FC_FUNC(dtrmv,DTRMV)(const char* uplo, const char* trans, const char* diag,                   const integer* n,                                                            const double* a, const integer* lda,       double* x, const integer* incx);
void FC_FUNC(dtbmv,DTBMV)(const char* uplo, const char* trans, const char* diag,                   const integer* n, const integer* k,                                          const double* a, const integer* lda,       double* x, const integer* incx);
void FC_FUNC(dtpmv,DTPMV)(const char* uplo, const char* trans, const char* diag,                   const integer* n,                                                            const double* ap,                          double* x, const integer* incx);
void FC_FUNC(dtrsv,DTRSV)(const char* uplo, const char* trans, const char* diag,                   const integer* n,                                                            const double* a, const integer* lda,       double* x, const integer* incx);
void FC_FUNC(dtbsv,DTBSV)(const char* uplo, const char* trans, const char* diag,                   const integer* n, const integer* k,                                          const double* a, const integer* lda,       double* x, const integer* incx);
void FC_FUNC(dtpsv,DTPSV)(const char* uplo, const char* trans, const char* diag,                   const integer* n,                                                            const double* ap,                          double* x, const integer* incx);
void FC_FUNC(dger,DGER)  (                                                       const integer* m, const integer* n,                                       const double* alpha, const double* x, const integer* incx, const double* y, const integer* incy, double* a, const integer* lda);
void FC_FUNC(dsyr,DSYR)  (const char* uplo,                                                        const integer* n,                                       const double* alpha, const double* x, const integer* incx,                                       double* a, const integer* lda);
void FC_FUNC(dspr,DSPR)  (const char* uplo,                                                        const integer* n,                                       const double* alpha, const double* x, const integer* incx,                                       double* ap);
void FC_FUNC(dsyr2,DSYR2)(const char* uplo,                                                        const integer* n,                                       const double* alpha, const double* x, const integer* incx, const double* y, const integer* incy, double* a, const integer* lda);
void FC_FUNC(dspr2,DSPR2)(const char* uplo,                                                        const integer* n,                                       const double* alpha, const double* x, const integer* incx, const double* y, const integer* incy, double* ap);

void FC_FUNC(cgemv,CGEMV)(                  const char* trans,                   const integer* m, const integer* n,                                       const scomplex* alpha, const scomplex* a, const integer* lda,  const scomplex* x, const integer* incx, const scomplex* beta, scomplex* y, const integer* incy);
void FC_FUNC(cgbmv,CGBMV)(                  const char* trans,                   const integer* m, const integer* n, const integer* kl, const integer* ku, const scomplex* alpha, const scomplex* a, const integer* lda,  const scomplex* x, const integer* incx, const scomplex* beta, scomplex* y, const integer* incy);
void FC_FUNC(chemv,CHEMV)(const char* uplo,                                                        const integer* n,                                       const scomplex* alpha, const scomplex* a, const integer* lda,  const scomplex* x, const integer* incx, const scomplex* beta, scomplex* y, const integer* incy);
void FC_FUNC(chbmv,CHBMV)(const char* uplo,                                                        const integer* n, const integer* k,                     const scomplex* alpha, const scomplex* a, const integer* lda,  const scomplex* x, const integer* incx, const scomplex* beta, scomplex* y, const integer* incy);
void FC_FUNC(chpmv,CHPMV)(const char* uplo,                                                        const integer* n,                                       const scomplex* alpha, const scomplex* ap,                     const scomplex* x, const integer* incx, const scomplex* beta, scomplex* y, const integer* incy);
void FC_FUNC(ctrmv,CTRMV)(const char* uplo, const char* trans, const char* diag,                   const integer* n,                                                              const scomplex* a, const integer* lda,        scomplex* x, const integer* incx);
void FC_FUNC(ctbmv,CTBMV)(const char* uplo, const char* trans, const char* diag,                   const integer* n, const integer* k,                                            const scomplex* a, const integer* lda,        scomplex* x, const integer* incx);
void FC_FUNC(ctpmv,CTPMV)(const char* uplo, const char* trans, const char* diag,                   const integer* n,                                                              const scomplex* ap,                           scomplex* x, const integer* incx);
void FC_FUNC(ctrsv,CTRSV)(const char* uplo, const char* trans, const char* diag,                   const integer* n,                                                              const scomplex* a, const integer* lda,        scomplex* x, const integer* incx);
void FC_FUNC(ctbsv,CTBSV)(const char* uplo, const char* trans, const char* diag,                   const integer* n, const integer* k,                                            const scomplex* a, const integer* lda,        scomplex* x, const integer* incx);
void FC_FUNC(ctpsv,CTPSV)(const char* uplo, const char* trans, const char* diag,                   const integer* n,                                                              const scomplex* ap,                           scomplex* x, const integer* incx);
void FC_FUNC(cgeru,CGERU)(                                                       const integer* m, const integer* n,                                       const scomplex* alpha, const scomplex* x, const integer* incx, const scomplex* y, const integer* incy, scomplex* a, const integer* lda);
void FC_FUNC(cgerc,CGERC)(                                                       const integer* m, const integer* n,                                       const scomplex* alpha, const scomplex* x, const integer* incx, const scomplex* y, const integer* incy, scomplex* a, const integer* lda);
void FC_FUNC(cher,CHER)  (const char* uplo,                                                        const integer* n,                                       const    float* alpha, const scomplex* x, const integer* incx,                                         scomplex* a, const integer* lda);
void FC_FUNC(chpr,CHPR)  (const char* uplo,                                                        const integer* n,                                       const    float* alpha, const scomplex* x, const integer* incx,                                         scomplex* ap);
void FC_FUNC(cher2,CHER2)(const char* uplo,                                                        const integer* n,                                       const scomplex* alpha, const scomplex* x, const integer* incx, const scomplex* y, const integer* incy, scomplex* a, const integer* lda);
void FC_FUNC(chpr2,CHPR2)(const char* uplo,                                                        const integer* n,                                       const scomplex* alpha, const scomplex* x, const integer* incx, const scomplex* y, const integer* incy, scomplex* ap);

void FC_FUNC(zgemv,ZGEMV)(                  const char* trans,                   const integer* m, const integer* n,                                       const dcomplex* alpha, const dcomplex* a, const integer* lda,  const dcomplex* x, const integer* incx, const dcomplex* beta, dcomplex* y, const integer* incy);
void FC_FUNC(zgbmv,ZGBMV)(                  const char* trans,                   const integer* m, const integer* n, const integer* kl, const integer* ku, const dcomplex* alpha, const dcomplex* a, const integer* lda,  const dcomplex* x, const integer* incx, const dcomplex* beta, dcomplex* y, const integer* incy);
void FC_FUNC(zhemv,ZHEMV)(const char* uplo,                                                        const integer* n,                                       const dcomplex* alpha, const dcomplex* a, const integer* lda,  const dcomplex* x, const integer* incx, const dcomplex* beta, dcomplex* y, const integer* incy);
void FC_FUNC(zhbmv,ZHBMV)(const char* uplo,                                                        const integer* n, const integer* k,                     const dcomplex* alpha, const dcomplex* a, const integer* lda,  const dcomplex* x, const integer* incx, const dcomplex* beta, dcomplex* y, const integer* incy);
void FC_FUNC(zhpmv,ZHPMV)(const char* uplo,                                                        const integer* n,                                       const dcomplex* alpha, const dcomplex* ap,                     const dcomplex* x, const integer* incx, const dcomplex* beta, dcomplex* y, const integer* incy);
void FC_FUNC(ztrmv,ZTRMV)(const char* uplo, const char* trans, const char* diag,                   const integer* n,                                                              const dcomplex* a, const integer* lda,        dcomplex* x, const integer* incx);
void FC_FUNC(ztbmv,ZTBMV)(const char* uplo, const char* trans, const char* diag,                   const integer* n, const integer* k,                                            const dcomplex* a, const integer* lda,        dcomplex* x, const integer* incx);
void FC_FUNC(ztpmv,ZTPMV)(const char* uplo, const char* trans, const char* diag,                   const integer* n,                                                              const dcomplex* ap,                           dcomplex* x, const integer* incx);
void FC_FUNC(ztrsv,ZTRSV)(const char* uplo, const char* trans, const char* diag,                   const integer* n,                                                              const dcomplex* a, const integer* lda,        dcomplex* x, const integer* incx);
void FC_FUNC(ztbsv,ZTBSV)(const char* uplo, const char* trans, const char* diag,                   const integer* n, const integer* k,                                            const dcomplex* a, const integer* lda,        dcomplex* x, const integer* incx);
void FC_FUNC(ztpsv,ZTPSV)(const char* uplo, const char* trans, const char* diag,                   const integer* n,                                                              const dcomplex* ap,                           dcomplex* x, const integer* incx);
void FC_FUNC(zgerc,ZGERC)(                                                       const integer* m, const integer* n,                                       const dcomplex* alpha, const dcomplex* x, const integer* incx, const dcomplex* y, const integer* incy, dcomplex* a, const integer* lda);
void FC_FUNC(zgeru,ZGERU)(                                                       const integer* m, const integer* n,                                       const dcomplex* alpha, const dcomplex* x, const integer* incx, const dcomplex* y, const integer* incy, dcomplex* a, const integer* lda);
void FC_FUNC(zher,ZHER)  (const char* uplo,                                                        const integer* n,                                       const   double* alpha, const dcomplex* x, const integer* incx,                                         dcomplex* a, const integer* lda);
void FC_FUNC(zhpr,ZHPR)  (const char* uplo,                                                        const integer* n,                                       const   double* alpha, const dcomplex* x, const integer* incx,                                         dcomplex* ap);
void FC_FUNC(zher2,ZHER2)(const char* uplo,                                                        const integer* n,                                       const dcomplex* alpha, const dcomplex* x, const integer* incx, const dcomplex* y, const integer* incy, dcomplex* a, const integer* lda);
void FC_FUNC(zhpr2,ZHPR2)(const char* uplo,                                                        const integer* n,                                       const dcomplex* alpha, const dcomplex* x, const integer* incx, const dcomplex* y, const integer* incy, dcomplex* ap);

/******************************************************************************
 *
 * Level 3 BLAS, FORTRAN prototypes
 *
 *****************************************************************************/
void FC_FUNC(sgemm,SGEMM)  (                                    const char* transa, const char* transb,                   const integer* m, const integer* n, const integer* k, const float* alpha, const float* a, const integer* lda, const float* b, const integer* ldb, const float* beta, float* c, const integer* ldc);
void FC_FUNC(ssymm,SSYMM)  (const char* side, const char* uplo,                                                           const integer* m, const integer* n,                   const float* alpha, const float* a, const integer* lda, const float* b, const integer* ldb, const float* beta, float* c, const integer* ldc);
void FC_FUNC(ssyrk,SSYRK)  (                  const char* uplo, const char* trans,                                                          const integer* n, const integer* k, const float* alpha, const float* a, const integer* lda,                                     const float* beta, float* c, const integer* ldc);
void FC_FUNC(ssyr2k,SSYR2K)(                  const char* uplo, const char* trans,                                                          const integer* n, const integer* k, const float* alpha, const float* a, const integer* lda, const float* b, const integer* ldb, const float* beta, float* c, const integer* ldc);
void FC_FUNC(strmm,STRMM)  (const char* side, const char* uplo, const char* transa,                     const char* diag, const integer* m, const integer* n,                   const float* alpha, const float* a, const integer* lda,       float* b, const integer* ldb);
void FC_FUNC(strsm,STRSM)  (const char* side, const char* uplo, const char* transa,                     const char* diag, const integer* m, const integer* n,                   const float* alpha, const float* a, const integer* lda,       float* b, const integer* ldb);

void FC_FUNC(dgemm,DGEMM)  (                                    const char* transa, const char* transb,                   const integer* m, const integer* n, const integer* k, const double* alpha, const double* a, const integer* lda, const double* b, const integer* ldb, const double* beta, double* c, const integer* ldc);
void FC_FUNC(dsymm,DSYMM)  (const char* side, const char* uplo,                                                           const integer* m, const integer* n,                   const double* alpha, const double* a, const integer* lda, const double* b, const integer* ldb, const double* beta, double* c, const integer* ldc);
void FC_FUNC(dsyrk,DSYRK)  (                  const char* uplo, const char* trans,                                                          const integer* n, const integer* k, const double* alpha, const double* a, const integer* lda,                                      const double* beta, double* c, const integer* ldc);
void FC_FUNC(dsyr2k,DSYR2K)(                  const char* uplo, const char* trans,                                                          const integer* n, const integer* k, const double* alpha, const double* a, const integer* lda, const double* b, const integer* ldb, const double* beta, double* c, const integer* ldc);
void FC_FUNC(dtrmm,DTRMM)  (const char* side, const char* uplo, const char* transa,                     const char* diag, const integer* m, const integer* n,                   const double* alpha, const double* a, const integer* lda,       double* b, const integer* ldb);
void FC_FUNC(dtrsm,DTRSM)  (const char* side, const char* uplo, const char* transa,                     const char* diag, const integer* m, const integer* n,                   const double* alpha, const double* a, const integer* lda,       double* b, const integer* ldb);

void FC_FUNC(cgemm,CGEMM)  (                                    const char* transa, const char* transb,                   const integer* m, const integer* n, const integer* k, const scomplex* alpha, const scomplex* a, const integer* lda, const scomplex* b, const integer* ldb, const scomplex* beta, scomplex* c, const integer* ldc);
void FC_FUNC(chemm,CHEMM)  (const char* side, const char* uplo,                                                           const integer* m, const integer* n,                   const scomplex* alpha, const scomplex* a, const integer* lda, const scomplex* b, const integer* ldb, const scomplex* beta, scomplex* c, const integer* ldc);
void FC_FUNC(csyrk,CSYRK)  (                  const char* uplo, const char* trans,                                                          const integer* n, const integer* k, const scomplex* alpha, const scomplex* a, const integer* lda,                                        const scomplex* beta, scomplex* c, const integer* ldc);
void FC_FUNC(csyr2k,CSYR2K)(                  const char* uplo, const char* trans,                                                          const integer* n, const integer* k, const scomplex* alpha, const scomplex* a, const integer* lda, const scomplex* b, const integer* ldb, const scomplex* beta, scomplex* c, const integer* ldc);
void FC_FUNC(cherk,CHERK)  (                  const char* uplo, const char* trans,                                                          const integer* n, const integer* k, const    float* alpha, const scomplex* a, const integer* lda,                                        const    float* beta, scomplex* c, const integer* ldc);
void FC_FUNC(cher2k,CHER2K)(                  const char* uplo, const char* trans,                                                          const integer* n, const integer* k, const scomplex* alpha, const scomplex* a, const integer* lda, const scomplex* b, const integer* ldb, const    float* beta, scomplex* c, const integer* ldc);
void FC_FUNC(ctrmm,CTRMM)  (const char* side, const char* uplo, const char* transa,                     const char* diag, const integer* m, const integer* n,                   const scomplex* alpha, const scomplex* a, const integer* lda,       scomplex* b, const integer* ldb);
void FC_FUNC(ctrsm,CTRSM)  (const char* side, const char* uplo, const char* transa,                     const char* diag, const integer* m, const integer* n,                   const scomplex* alpha, const scomplex* a, const integer* lda,       scomplex* b, const integer* ldb);

void FC_FUNC(zgemm,ZGEMM)  (                                    const char* transa, const char* transb,                   const integer* m, const integer* n, const integer* k, const dcomplex* alpha, const dcomplex* a, const integer* lda, const dcomplex* b, const integer* ldb, const dcomplex* beta, dcomplex* c, const integer* ldc);
void FC_FUNC(zhemm,ZHEMM)  (const char* side, const char* uplo,                                                           const integer* m, const integer* n,                   const dcomplex* alpha, const dcomplex* a, const integer* lda, const dcomplex* b, const integer* ldb, const dcomplex* beta, dcomplex* c, const integer* ldc);
void FC_FUNC(zsyrk,ZSYRK)  (                  const char* uplo, const char* trans,                                                          const integer* n, const integer* k, const dcomplex* alpha, const dcomplex* a, const integer* lda,                                        const dcomplex* beta, dcomplex* c, const integer* ldc);
void FC_FUNC(zsyr2k,ZSYR2K)(                  const char* uplo, const char* trans,                                                          const integer* n, const integer* k, const dcomplex* alpha, const dcomplex* a, const integer* lda, const dcomplex* b, const integer* ldb, const dcomplex* beta, dcomplex* c, const integer* ldc);
void FC_FUNC(zherk,ZHERK)  (                  const char* uplo, const char* trans,                                                          const integer* n, const integer* k, const   double* alpha, const dcomplex* a, const integer* lda,                                        const   double* beta, dcomplex* c, const integer* ldc);
void FC_FUNC(zher2k,ZHER2K)(                  const char* uplo, const char* trans,                                                          const integer* n, const integer* k, const dcomplex* alpha, const dcomplex* a, const integer* lda, const dcomplex* b, const integer* ldb, const   double* beta, dcomplex* c, const integer* ldc);
void FC_FUNC(ztrmm,ZTRMM)  (const char* side, const char* uplo, const char* transa,                     const char* diag, const integer* m, const integer* n,                   const dcomplex* alpha, const dcomplex* a, const integer* lda,       dcomplex* b, const integer* ldb);
void FC_FUNC(ztrsm,ZTRSM)  (const char* side, const char* uplo, const char* transa,                     const char* diag, const integer* m, const integer* n,                   const dcomplex* alpha, const dcomplex* a, const integer* lda,       dcomplex* b, const integer* ldb);

/******************************************************************************
 *
 * Level 1 BLAS, C wrappers
 *
 *****************************************************************************/
static inline void   c_srotg (float* a, float* b, float* c, float* s)
{
    FC_FUNC(srotg,SROTG)(a, b, c, s);
}

static inline void   c_srotmg(float* d1, float* d2, float* a, float b, float* param)
{
    FC_FUNC(srotmg,SROTMG)(d1, d2, a, &b, param);
}

static inline void   c_srot  (integer n, float* x, integer incx,
                                               float* y, integer incy, float c, float s)
{
    FC_FUNC(srot,SROT)(&n, x, &incx, y, &incy, &c, &s);
}

static inline void   c_srotm (integer n, float* x, integer incx,
                                               float* y, integer incy, float* param)
{
    FC_FUNC(srotm,SROTM)(&n, x, &incx, y, &incy, param);
}

static inline void   c_sswap (integer n, float* x, integer incx,
                                               float* y, integer incy)
{
    FC_FUNC(sswap,SSWAP)(&n, x, &incx, y, &incy);
}

static inline void   c_sscal (integer n, float alpha, float* x, integer incx)
{
    FC_FUNC(sscal,SSCAL)(&n, &alpha, x, &incx);
}

static inline void   c_scopy (integer n, const float* x, integer incx,
                                                     float* y, integer incy)
{
    FC_FUNC(scopy,SCOPY)(&n, (float*)x, &incx, y, &incy);
}

static inline void   c_saxpy (integer n, float alpha, const float* x, integer incx,
                                                                        float* y, integer incy)
{
    FC_FUNC(saxpy,SAXPY)(&n, &alpha, (float*)x, &incx, y, &incy);
}

static inline float c_sdot  (integer n, const float* x, integer incx,
                                              const float* y, integer incy)
{
    return FC_FUNC(sdot,SDOT)(&n, (float*)x, &incx, (float*)y, &incy);
}

static inline float c_snrm2 (integer n, const float* x, integer incx)
{
    return FC_FUNC(snrm2,SNRM2)(&n, (float*)x, &incx);
}

static inline float c_sasum (integer n, const float* x, integer incx)
{
    return FC_FUNC(sasum,SASUM)(&n, (float*)x, &incx);
}

static inline integer c_isamax(integer n, const float* x, integer incx)
{
    return FC_FUNC(isamax,ISAMAX)(&n, (float*)x, &incx)-1;
}

static inline void   c_drotg (double* a, double* b, double* c, double* s)
{
    FC_FUNC(drotg,DROTG)(a, b, c, s);
}

static inline void   c_drotmg(double* d1, double* d2, double* a, double b, double* param)
{
    FC_FUNC(drotmg,DROTMG)(d1, d2, a, &b, param);
}

static inline void   c_drot  (integer n, double* x, integer incx,
                                               double* y, integer incy, double c, double s)
{
    FC_FUNC(drot,DROT)(&n, x, &incx, y, &incy, &c, &s);
}

static inline void   c_drotm (integer n, double* x, integer incx,
                                               double* y, integer incy, double* param)
{
    FC_FUNC(drotm,DROTM)(&n, x, &incx, y, &incy, param);
}

static inline void   c_dswap (integer n, double* x, integer incx,
                                               double* y, integer incy)
{
    FC_FUNC(dswap,DSWAP)(&n, x, &incx, y, &incy);
}

static inline void   c_dscal (integer n, double alpha, double* x, integer incx)
{
    FC_FUNC(dscal,DSCAL)(&n, &alpha, x, &incx);
}

static inline void   c_dcopy (integer n, const double* x, integer incx,
                                                     double* y, integer incy)
{
    FC_FUNC(dcopy,DCOPY)(&n, (double*)x, &incx, y, &incy);
}

static inline void   c_daxpy (integer n, double alpha, const double* x, integer incx,
                                                                         double* y, integer incy)
{
    FC_FUNC(daxpy,DAXPY)(&n, &alpha, (double*)x, &incx, y, &incy);
}

static inline double c_ddot  (integer n, const double* x, integer incx,
                                               const double* y, integer incy)
{
    return FC_FUNC(ddot,DDOT)(&n, (double*)x, &incx, (double*)y, &incy);
}

static inline double c_dnrm2 (integer n, const double* x, integer incx)
{
    return FC_FUNC(dnrm2,DNRM2)(&n, (double*)x, &incx);
}

static inline double c_dasum (integer n, const double* x, integer incx)
{
    return FC_FUNC(dasum,DASUM)(&n, (double*)x, &incx);
}

static inline integer c_idamax(integer n, const double* x, integer incx)
{
    return FC_FUNC(idamax,IDAMAX)(&n, (double*)x, &incx)-1;
}

static inline void   c_crotg (scomplex* a, scomplex* b, float* c, scomplex* s)
{
    FC_FUNC(crotg,CROTG)(a, b, c, s);
}

static inline void   c_csrot  (integer n, scomplex* x, integer incx,
                                                scomplex* y, integer incy, float c, float s)
{
    FC_FUNC(csrot,CSROT)(&n, x, &incx, y, &incy, &c, &s);
}

static inline void   c_cswap (integer n, scomplex* x, integer incx,
                                               scomplex* y, integer incy)
{
    FC_FUNC(cswap,CSWAP)(&n, x, &incx, y, &incy);
}

static inline void   c_cscal (integer n, scomplex alpha, scomplex* x, integer incx)
{
    FC_FUNC(cscal,CSCAL)(&n, &alpha, x, &incx);
}

static inline void   c_csscal (integer n, float alpha, scomplex* x, integer incx)
{
    FC_FUNC(csscal,CSSCAL)(&n, &alpha, x, &incx);
}

static inline void   c_ccopy (integer n, const scomplex* x, integer incx,
                                                     scomplex* y, integer incy)
{
    FC_FUNC(ccopy,CCOPY)(&n, (scomplex*)x, &incx, y, &incy);
}

static inline void   c_caxpy (integer n, scomplex alpha, const scomplex* x, integer incx,
                                                                           scomplex* y, integer incy)
{
    FC_FUNC(caxpy,CAXPY)(&n, &alpha, (scomplex*)x, &incx, y, &incy);
}

static inline scomplex c_cdotu (integer n, const scomplex* x, integer incx,
                                                 const scomplex* y, integer incy)
{
    return FC_FUNC(cdotu,CDOTU)(&n, (scomplex*)x, &incx, (scomplex*)y, &incy);
}

static inline scomplex c_cdotc (integer n, const scomplex* x, integer incx,
                                                 const scomplex* y, integer incy)
{
    return FC_FUNC(cdotc,CDOTC)(&n, (scomplex*)x, &incx, (scomplex*)y, &incy);
}

static inline float c_scnrm2 (integer n, const scomplex* x, integer incx)
{
    return FC_FUNC(scnrm2,SCNRM2)(&n, (scomplex*)x, &incx);
}

static inline float c_scasum (integer n, const scomplex* x, integer incx)
{
    return FC_FUNC(scasum,SCASUM)(&n, (scomplex*)x, &incx);
}

static inline integer c_icamax(integer n, const scomplex* x, integer incx)
{
    return FC_FUNC(icamax,ICAMAX)(&n, (scomplex*)x, &incx)-1;
}

static inline void   c_zrotg (dcomplex* a, dcomplex* b, double* c, dcomplex* s)
{
    FC_FUNC(zrotg,ZROTG)(a, b, c, s);
}

static inline void   c_zdrot  (integer n, dcomplex* x, integer incx,
                                                dcomplex* y, integer incy, double c, double s)
{
    FC_FUNC(zdrot,ZDROT)(&n, x, &incx, y, &incy, &c, &s);
}

static inline void   c_zswap (integer n, dcomplex* x, integer incx,
                                               dcomplex* y, integer incy)
{
    FC_FUNC(zswap,ZSWAP)(&n, x, &incx, y, &incy);
}

static inline void   c_zscal (integer n, dcomplex alpha, dcomplex* x, integer incx)
{
    FC_FUNC(zscal,ZSCAL)(&n, &alpha, x, &incx);
}

static inline void   c_zdscal (integer n, double alpha, dcomplex* x, integer incx)
{
    FC_FUNC(zdscal,ZDSCAL)(&n, &alpha, x, &incx);
}

static inline void   c_zcopy (integer n, const dcomplex* x, integer incx,
                                                     dcomplex* y, integer incy)
{
    FC_FUNC(zcopy,ZCOPY)(&n, (dcomplex*)x, &incx, y, &incy);
}

static inline void   c_zaxpy (integer n, dcomplex alpha, const dcomplex* x, integer incx,
                                                                           dcomplex* y, integer incy)
{
    FC_FUNC(zaxpy,ZAXPY)(&n, &alpha, (dcomplex*)x, &incx, y, &incy);
}

static inline dcomplex c_zdotu (integer n, const dcomplex* x, integer incx,
                                                 const dcomplex* y, integer incy)
{
    return FC_FUNC(zdotu,ZDOTU)(&n, (dcomplex*)x, &incx, (dcomplex*)y, &incy);
}

static inline dcomplex c_zdotc (integer n, const dcomplex* x, integer incx,
                                                 const dcomplex* y, integer incy)
{
    return FC_FUNC(zdotc,ZDOTC)(&n, (dcomplex*)x, &incx, (dcomplex*)y, &incy);
}

static inline double c_dznrm2 (integer n, const dcomplex* x, integer incx)
{
    return FC_FUNC(dznrm2,DZNRM2)(&n, (dcomplex*)x, &incx);
}

static inline double c_dzasum (integer n, const dcomplex* x, integer incx)
{
    return FC_FUNC(dzasum,DZASUM)(&n, (dcomplex*)x, &incx);
}

static inline integer c_izamax(integer n, const dcomplex* x, integer incx)
{
    return FC_FUNC(izamax,IZAMAX)(&n, (dcomplex*)x, &incx)-1;
}

/******************************************************************************
 *
 * Level 2 BLAS, C wrappers
 *
 *****************************************************************************/
static inline void c_sgemv(char trans, integer m, integer n,
                           float alpha, const float* a, integer lda,
                                              const float* x, integer incx,
                           float  beta,       float* y, integer incy)
{
    FC_FUNC(sgemv,SGEMV)(&trans, &m, &n, &alpha, (float*)a, &lda, (float*)x, &incx, &beta, y, &incy);
}

static inline void c_sgbmv(char trans,
                           integer m, integer n, integer kl, integer ku,
                           float alpha, const float* a, integer lda,
                                              const float* x, integer incx,
                           float  beta,       float* y, integer incy)
{
    FC_FUNC(sgbmv,SGBMV)(&trans, &m, &n, &kl, &ku, &alpha, (float*)a, &lda, (float*)x, &incx, &beta, y, &incy);
}

static inline void c_ssymv(char uplo, integer n,
                           float alpha, const float* a, integer lda,
                                              const float* x, integer incx,
                           float  beta,       float* y, integer incy)
{
    FC_FUNC(ssymv,SSYMV)(&uplo, &n, &alpha, (float*)a, &lda, (float*)x, &incx, &beta, y, &incy);
}

static inline void c_ssbmv(char uplo, integer n, integer k,
                           float alpha, const float* a, integer lda,
                                              const float* x, integer incx,
                           float  beta,       float* y, integer incy)
{
    FC_FUNC(ssbmv,SSBMV)(&uplo, &n, &k, &alpha, (float*)a, &lda, (float*)x, &incx, &beta, y, &incy);
}

static inline void c_sspmv(char uplo, integer n,
                           float alpha, const float* ap,
                                              const float*  x, integer incx,
                           float  beta,       float*  y, integer incy)
{
    FC_FUNC(sspmv,SSPMV)(&uplo, &n, &alpha, (float*)ap, (float*)x, &incx, &beta, y, &incy);
}

static inline void c_strmv(char uplo, char trans, char diag, integer n,
                           const float* a, integer lda,
                                 float* x, integer incx)
{
    FC_FUNC(strmv,STRMV)(&uplo, &trans, &diag, &n, (float*)a, &lda, x, &incx);
}

static inline void c_stbmv(char uplo, char trans, char diag,
                           integer n, integer k,
                           const float* a, integer lda,
                                 float* x, integer incx)
{
    FC_FUNC(stbmv,STBMV)(&uplo, &trans, &diag, &n, &k, (float*)a, &lda, x, &incx);
}

static inline void c_stpmv(char uplo, char trans, char diag, integer n,
                           const float* ap, float* x, integer incx)
{
    FC_FUNC(stpmv,STPMV)(&uplo, &trans, &diag, &n, (float*)ap, x, &incx);
}

static inline void c_strsv(char uplo, char trans, char diag, integer n,
                           const float* a, integer lda,
                                 float* x, integer incx)
{
    FC_FUNC(strsv,STRSV)(&uplo, &trans, &diag, &n, (float*)a, &lda, x, &incx);
}

static inline void c_stbsv(char uplo, char trans, char diag,
                           integer n, integer k,
                           const float* a, integer lda,
                                 float* x, integer incx)
{
    FC_FUNC(stbsv,STBSV)(&uplo, &trans, &diag, &n, &k, (float*)a, &lda, x, &incx);
}

static inline void c_stpsv(char uplo, char trans, char diag, integer n,
                           const float* ap, float* x, integer incx)
{
    FC_FUNC(stpsv,STPSV)(&uplo, &trans, &diag, &n, (float*)ap, x, &incx);
}

static inline void c_sger (integer m, integer n,
                           float alpha, const float* x, integer incx,
                                              const float* y, integer incy,
                                                    float* a, integer lda)
{
    FC_FUNC(sger,SGER)(&m, &n, &alpha, (float*)x, &incx, (float*)y, &incy, a, &lda);
}

static inline void c_ssyr (char uplo, integer n,
                           float alpha, const float* x, integer incx,
                                                    float* a, integer lda)
{
    FC_FUNC(ssyr,SSYR)(&uplo, &n, &alpha, (float*)x, &incx, a, &lda);
}

static inline void c_sspr (char uplo, integer n,
                           float alpha, const float* x, integer incx,
                                                    float* ap)
{
    FC_FUNC(sspr,SSPR)(&uplo, &n, &alpha, (float*)x, &incx, ap);
}

static inline void c_ssyr2(char uplo, integer n,
                           float alpha, const float* x, integer incx,
                                              const float* y, integer incy,
                                                    float* a, integer lda)
{
    FC_FUNC(ssyr2,SSYR2)(&uplo, &n, &alpha, (float*)x, &incx, (float*)y, &incy, a, &lda);
}

static inline void c_sspr2(char uplo, integer n,
                           float alpha, const float* x, integer incx,
                                              const float* y, integer incy,
                                                    float* ap)
{
    FC_FUNC(sspr2,SSPR2)(&uplo, &n, &alpha, (float*)x, &incx, (float*)y, &incy, ap);
}

static inline void c_dgemv(char trans, integer m, integer n,
                           double alpha, const double* a, integer lda,
                                               const double* x, integer incx,
                           double  beta,       double* y, integer incy)
{
    FC_FUNC(dgemv,DGEMV)(&trans, &m, &n, &alpha, (double*)a, &lda, (double*)x, &incx, &beta, y, &incy);
}

static inline void c_dgbmv(char trans,
                           integer m, integer n, integer kl, integer ku,
                           double alpha, const double* a, integer lda,
                                               const double* x, integer incx,
                           double  beta,       double* y, integer incy)
{
    FC_FUNC(dgbmv,DGBMV)(&trans, &m, &n, &kl, &ku, &alpha, (double*)a, &lda, (double*)x, &incx, &beta, y, &incy);
}

static inline void c_dsymv(char uplo, integer n,
                           double alpha, const double* a, integer lda,
                                               const double* x, integer incx,
                           double  beta,       double* y, integer incy)
{
    FC_FUNC(dsymv,DSYMV)(&uplo, &n, &alpha, (double*)a, &lda, (double*)x, &incx, &beta, y, &incy);
}

static inline void c_dsbmv(char uplo, integer n, integer k,
                           double alpha, const double* a, integer lda,
                                               const double* x, integer incx,
                           double  beta,       double* y, integer incy)
{
    FC_FUNC(dsbmv,DSBMV)(&uplo, &n, &k, &alpha, (double*)a, &lda, (double*)x, &incx, &beta, y, &incy);
}

static inline void c_dspmv(char uplo, integer n,
                           double alpha, const double* ap,
                                               const double*  x, integer incx,
                           double  beta,       double*  y, integer incy)
{
    FC_FUNC(dspmv,DSPMV)(&uplo, &n, &alpha, (double*)ap, (double*)x, &incx, &beta, y, &incy);
}

static inline void c_dtrmv(char uplo, char trans, char diag, integer n,
                           const double* a, integer lda,
                                 double* x, integer incx)
{
    FC_FUNC(dtrmv,DTRMV)(&uplo, &trans, &diag, &n, (double*)a, &lda, x, &incx);
}

static inline void c_dtbmv(char uplo, char trans, char diag,
                           integer n, integer k,
                           const double* a, integer lda,
                                 double* x, integer incx)
{
    FC_FUNC(dtbmv,DTBMV)(&uplo, &trans, &diag, &n, &k, (double*)a, &lda, x, &incx);
}

static inline void c_dtpmv(char uplo, char trans, char diag, integer n,
                           const double* ap, double* x, integer incx)
{
    FC_FUNC(dtpmv,DTPMV)(&uplo, &trans, &diag, &n, (double*)ap, x, &incx);
}

static inline void c_dtrsv(char uplo, char trans, char diag, integer n,
                           const double* a, integer lda,
                                 double* x, integer incx)
{
    FC_FUNC(dtrsv,DTRSV)(&uplo, &trans, &diag, &n, (double*)a, &lda, x, &incx);
}

static inline void c_dtbsv(char uplo, char trans, char diag,
                           integer n, integer k,
                           const double* a, integer lda,
                                 double* x, integer incx)
{
    FC_FUNC(dtbsv,DTBSV)(&uplo, &trans, &diag, &n, &k, (double*)a, &lda, x, &incx);
}

static inline void c_dtpsv(char uplo, char trans, char diag, integer n,
                           const double* ap, double* x, integer incx)
{
    FC_FUNC(dtpsv,DTPSV)(&uplo, &trans, &diag, &n, (double*)ap, x, &incx);
}

static inline void c_dger (integer m, integer n,
                           double alpha, const double* x, integer incx,
                                               const double* y, integer incy,
                                                     double* a, integer lda)
{
    FC_FUNC(dger,DGER)(&m, &n, &alpha, (double*)x, &incx, (double*)y, &incy, a, &lda);
}

static inline void c_dsyr (char uplo, integer n,
                           double alpha, const double* x, integer incx,
                                                     double* a, integer lda)
{
    FC_FUNC(dsyr,DSYR)(&uplo, &n, &alpha, (double*)x, &incx, a, &lda);
}

static inline void c_dspr (char uplo, integer n,
                           double alpha, const double* x, integer incx,
                                                     double* ap)
{
    FC_FUNC(dspr,DSPR)(&uplo, &n, &alpha, (double*)x, &incx, ap);
}

static inline void c_dsyr2(char uplo, integer n,
                           double alpha, const double* x, integer incx,
                                               const double* y, integer incy,
                                                     double* a, integer lda)
{
    FC_FUNC(dsyr2,DSYR2)(&uplo, &n, &alpha, (double*)x, &incx, (double*)y, &incy, a, &lda);
}

static inline void c_dspr2(char uplo, integer n,
                           double alpha, const double* x, integer incx,
                                               const double* y, integer incy,
                                                     double* ap)
{
    FC_FUNC(dspr2,DSPR2)(&uplo, &n, &alpha, (double*)x, &incx, (double*)y, &incy, ap);
}

static inline void c_cgemv(char trans, integer m, integer n,
                           scomplex alpha, const scomplex* a, integer lda,
                                                 const scomplex* x, integer incx,
                           scomplex  beta,       scomplex* y, integer incy)
{
    FC_FUNC(cgemv,CGEMV)(&trans, &m, &n, &alpha, (scomplex*)a, &lda, (scomplex*)x, &incx, &beta, y, &incy);
}

static inline void c_cgbmv(char trans,
                           integer m, integer n, integer kl, integer ku,
                           scomplex alpha, const scomplex* a, integer lda,
                                                 const scomplex* x, integer incx,
                           scomplex  beta,       scomplex* y, integer incy)
{
    FC_FUNC(cgbmv,CGBMV)(&trans, &m, &n, &kl, &ku, &alpha, (scomplex*)a, &lda, (scomplex*)x, &incx, &beta, y, &incy);
}

static inline void c_chemv(char uplo, integer n,
                           scomplex alpha, const scomplex* a, integer lda,
                                                 const scomplex* x, integer incx,
                           scomplex  beta,       scomplex* y, integer incy)
{
    FC_FUNC(chemv,CHEMV)(&uplo, &n, &alpha, (scomplex*)a, &lda, (scomplex*)x, &incx, &beta, y, &incy);
}

static inline void c_chbmv(char uplo, integer n, integer k,
                           scomplex alpha, const scomplex* a, integer lda,
                                                 const scomplex* x, integer incx,
                           scomplex  beta,       scomplex* y, integer incy)
{
    FC_FUNC(chbmv,CHBMV)(&uplo, &n, &k, &alpha, (scomplex*)a, &lda, (scomplex*)x, &incx, &beta, y, &incy);
}

static inline void c_chpmv(char uplo, integer n,
                           scomplex alpha, const scomplex* ap,
                                                 const scomplex*  x, integer incx,
                           scomplex  beta,       scomplex*  y, integer incy)
{
    FC_FUNC(chpmv,CHPMV)(&uplo, &n, &alpha, (scomplex*)ap, (scomplex*)x, &incx, &beta, y, &incy);
}

static inline void c_ctrmv(char uplo, char trans, char diag, integer n,
                           const scomplex* a, integer lda,
                                 scomplex* x, integer incx)
{
    FC_FUNC(ctrmv,CTRMV)(&uplo, &trans, &diag, &n, (scomplex*)a, &lda, x, &incx);
}

static inline void c_ctbmv(char uplo, char trans, char diag,
                           integer n, integer k,
                           const scomplex* a, integer lda,
                                 scomplex* x, integer incx)
{
    FC_FUNC(ctbmv,CTBMV)(&uplo, &trans, &diag, &n, &k, (scomplex*)a, &lda, x, &incx);
}

static inline void c_ctpmv(char uplo, char trans, char diag, integer n,
                           const scomplex* ap, scomplex* x, integer incx)
{
    FC_FUNC(ctpmv,CTPMV)(&uplo, &trans, &diag, &n, (scomplex*)ap, x, &incx);
}

static inline void c_ctrsv(char uplo, char trans, char diag, integer n,
                           const scomplex* a, integer lda,
                                 scomplex* x, integer incx)
{
    FC_FUNC(ctrsv,CTRSV)(&uplo, &trans, &diag, &n, (scomplex*)a, &lda, x, &incx);
}

static inline void c_ctbsv(char uplo, char trans, char diag,
                           integer n, integer k,
                           const scomplex* a, integer lda,
                                 scomplex* x, integer incx)
{
    FC_FUNC(ctbsv,CTBSV)(&uplo, &trans, &diag, &n, &k, (scomplex*)a, &lda, x, &incx);
}

static inline void c_ctpsv(char uplo, char trans, char diag, integer n,
                           const scomplex* ap, scomplex* x, integer incx)
{
    FC_FUNC(ctpsv,CTPSV)(&uplo, &trans, &diag, &n, (scomplex*)ap, x, &incx);
}

static inline void c_cgeru(integer m, integer n,
                           scomplex alpha, const scomplex* x, integer incx,
                                                 const scomplex* y, integer incy,
                                                       scomplex* a, integer lda)
{
    FC_FUNC(cgeru,CGERU)(&m, &n, &alpha, (scomplex*)x, &incx, (scomplex*)y, &incy, a, &lda);
}

static inline void c_cgerc(integer m, integer n,
                           scomplex alpha, const scomplex* x, integer incx,
                                                 const scomplex* y, integer incy,
                                                       scomplex* a, integer lda)
{
    FC_FUNC(cgerc,CGERC)(&m, &n, &alpha, (scomplex*)x, &incx, (scomplex*)y, &incy, a, &lda);
}

static inline void c_cher (char uplo, integer n,
                           float alpha, const scomplex* x, integer incx,
                                                       scomplex* a, integer lda)
{
    FC_FUNC(cher,CHER)(&uplo, &n, &alpha, (scomplex*)x, &incx, a, &lda);
}

static inline void c_chpr (char uplo, integer n,
                           float alpha, const scomplex* x, integer incx,
                                                       scomplex* ap)
{
    FC_FUNC(chpr,CHPR)(&uplo, &n, &alpha, (scomplex*)x, &incx, ap);
}

static inline void c_cher2(char uplo, integer n,
                           scomplex alpha, const scomplex* x, integer incx,
                                                 const scomplex* y, integer incy,
                                                       scomplex* a, integer lda)
{
    FC_FUNC(cher2,CHER2)(&uplo, &n, &alpha, (scomplex*)x, &incx, (scomplex*)y, &incy, a, &lda);
}

static inline void c_chpr2(char uplo, integer n,
                           scomplex alpha, const scomplex* x, integer incx,
                                                 const scomplex* y, integer incy,
                                                       scomplex* ap)
{
    FC_FUNC(chpr2,CHPR2)(&uplo, &n, &alpha, (scomplex*)x, &incx, (scomplex*)y, &incy, ap);
}

static inline void c_zgemv(char trans, integer m, integer n,
                           dcomplex alpha, const dcomplex* a, integer lda,
                                                 const dcomplex* x, integer incx,
                           dcomplex  beta,       dcomplex* y, integer incy)
{
    FC_FUNC(zgemv,ZGEMV)(&trans, &m, &n, &alpha, (dcomplex*)a, &lda, (dcomplex*)x, &incx, &beta, y, &incy);
}

static inline void c_zgbmv(char trans,
                           integer m, integer n, integer kl, integer ku,
                           dcomplex alpha, const dcomplex* a, integer lda,
                                                 const dcomplex* x, integer incx,
                           dcomplex  beta,       dcomplex* y, integer incy)
{
    FC_FUNC(zgbmv,ZGBMV)(&trans, &m, &n, &kl, &ku, &alpha, (dcomplex*)a, &lda, (dcomplex*)x, &incx, &beta, y, &incy);
}

static inline void c_zhemv(char uplo, integer n,
                           dcomplex alpha, const dcomplex* a, integer lda,
                                                 const dcomplex* x, integer incx,
                           dcomplex  beta,       dcomplex* y, integer incy)
{
    FC_FUNC(zhemv,ZHEMV)(&uplo, &n, &alpha, (dcomplex*)a, &lda, (dcomplex*)x, &incx, &beta, y, &incy);
}

static inline void c_zhbmv(char uplo, integer n, integer k,
                           dcomplex alpha, const dcomplex* a, integer lda,
                                                 const dcomplex* x, integer incx,
                           dcomplex  beta,       dcomplex* y, integer incy)
{
    FC_FUNC(zhbmv,ZHBMV)(&uplo, &n, &k, &alpha, (dcomplex*)a, &lda, (dcomplex*)x, &incx, &beta, y, &incy);
}

static inline void c_zhpmv(char uplo, integer n,
                           dcomplex alpha, const dcomplex* ap,
                                                 const dcomplex*  x, integer incx,
                           dcomplex  beta,       dcomplex*  y, integer incy)
{
    FC_FUNC(zhpmv,ZHPMV)(&uplo, &n, &alpha, (dcomplex*)ap, (dcomplex*)x, &incx, &beta, y, &incy);
}

static inline void c_ztrmv(char uplo, char trans, char diag, integer n,
                           const dcomplex* a, integer lda,
                                 dcomplex* x, integer incx)
{
    FC_FUNC(ztrmv,ZTRMV)(&uplo, &trans, &diag, &n, (dcomplex*)a, &lda, x, &incx);
}

static inline void c_ztbmv(char uplo, char trans, char diag,
                           integer n, integer k,
                           const dcomplex* a, integer lda,
                                 dcomplex* x, integer incx)
{
    FC_FUNC(ztbmv,ZTBMV)(&uplo, &trans, &diag, &n, &k, (dcomplex*)a, &lda, x, &incx);
}

static inline void c_ztpmv(char uplo, char trans, char diag, integer n,
                           const dcomplex* ap, dcomplex* x, integer incx)
{
    FC_FUNC(ztpmv,ZTPMV)(&uplo, &trans, &diag, &n, (dcomplex*)ap, x, &incx);
}

static inline void c_ztrsv(char uplo, char trans, char diag, integer n,
                           const dcomplex* a, integer lda,
                                 dcomplex* x, integer incx)
{
    FC_FUNC(ztrsv,ZTRSV)(&uplo, &trans, &diag, &n, (dcomplex*)a, &lda, x, &incx);
}

static inline void c_ztbsv(char uplo, char trans, char diag,
                           integer n, integer k,
                           const dcomplex* a, integer lda,
                                 dcomplex* x, integer incx)
{
    FC_FUNC(ztbsv,ZTBSV)(&uplo, &trans, &diag, &n, &k, (dcomplex*)a, &lda, x, &incx);
}

static inline void c_ztpsv(char uplo, char trans, char diag, integer n,
                           const dcomplex* ap, dcomplex* x, integer incx)
{
    FC_FUNC(ztpsv,ZTPSV)(&uplo, &trans, &diag, &n, (dcomplex*)ap, x, &incx);
}

static inline void c_zgeru(integer m, integer n,
                           dcomplex alpha, const dcomplex* x, integer incx,
                                                 const dcomplex* y, integer incy,
                                                       dcomplex* a, integer lda)
{
    FC_FUNC(zgeru,ZGERU)(&m, &n, &alpha, (dcomplex*)x, &incx, (dcomplex*)y, &incy, a, &lda);
}

static inline void c_zgerc(integer m, integer n,
                           dcomplex alpha, const dcomplex* x, integer incx,
                                                 const dcomplex* y, integer incy,
                                                       dcomplex* a, integer lda)
{
    FC_FUNC(zgerc,ZGERC)(&m, &n, &alpha, (dcomplex*)x, &incx, (dcomplex*)y, &incy, a, &lda);
}

static inline void c_zher (char uplo, integer n,
                           double alpha, const dcomplex* x, integer incx,
                                                       dcomplex* a, integer lda)
{
    FC_FUNC(zher,ZHER)(&uplo, &n, &alpha, (dcomplex*)x, &incx, a, &lda);
}

static inline void c_zhpr (char uplo, integer n,
                           double alpha, const dcomplex* x, integer incx,
                                                       dcomplex* ap)
{
    FC_FUNC(zhpr,ZHPR)(&uplo, &n, &alpha, (dcomplex*)x, &incx, ap);
}

static inline void c_zher2(char uplo, integer n,
                           dcomplex alpha, const dcomplex* x, integer incx,
                                                 const dcomplex* y, integer incy,
                                                       dcomplex* a, integer lda)
{
    FC_FUNC(zher2,ZHER2)(&uplo, &n, &alpha, (dcomplex*)x, &incx, (dcomplex*)y, &incy, a, &lda);
}

static inline void c_zhpr2(char uplo, integer n,
                           dcomplex alpha, const dcomplex* x, integer incx,
                                                 const dcomplex* y, integer incy,
                                                       dcomplex* ap)
{
    FC_FUNC(zhpr2,ZHPR2)(&uplo, &n, &alpha, (dcomplex*)x, &incx, (dcomplex*)y, &incy, ap);
}

/******************************************************************************
 *
 * Level 3 BLAS, C wrappers
 *
 *****************************************************************************/
static inline void c_sgemm (char transa, char transb,
                            integer m, integer n, integer k,
                            float alpha, const float* a, integer lda,
                                               const float* b, integer ldb,
                            float  beta,       float* c, integer ldc)
{
    FC_FUNC(sgemm,SGEMM)(&transa, &transb, &m, &n, &k, &alpha, (float*)a, &lda, (float*)b, &ldb, &beta, c, &ldc);
}

static inline void c_ssymm (char side, char uplo,
                            integer m, integer n,
                            float alpha, const float* a, integer lda,
                                               const float* b, integer ldb,
                            float  beta,       float* c, integer ldc)
{
    FC_FUNC(ssymm,SSYMM)(&side, &uplo, &m, &n, &alpha, (float*)a, &lda, (float*)b, &ldb, &beta, c, &ldc);
}

static inline void c_ssyrk (char uplo, char trans,
                            integer n, integer k,
                            float alpha, const float* a, integer lda,
                            float  beta,       float* c, integer ldc)
{
    FC_FUNC(ssyrk,SSYRK)(&uplo, &trans, &n, &k, &alpha, (float*)a, &lda, &beta, c, &ldc);
}

static inline void c_ssyr2k(char uplo, char trans,
                            integer n, integer k,
                            float alpha, const float* a, integer lda,
                                               const float* b, integer ldb,
                            float  beta,       float* c, integer ldc)
{
    FC_FUNC(ssyr2k,SSYR2K)(&uplo, &trans, &n, &k, &alpha, (float*)a, &lda, (float*)b, &ldb, &beta, c, &ldc);
}

static inline void c_strmm (char side, char uplo, char transa, char diag,
                            integer m, integer n,
                            float alpha, const float* a, integer lda,
                                                     float* b, integer ldb)
{
    FC_FUNC(strmm,STRMM)(&side, &uplo, &transa, &diag, &m, &n, &alpha, (float*)a, &lda, b, &ldb);
}

static inline void c_strsm (char side, char uplo, char transa, char diag,
                            integer m, integer n,
                            float alpha, const float* a, integer lda,
                                                     float* b, integer ldb)
{
    FC_FUNC(strsm,STRSM)(&side, &uplo, &transa, &diag, &m, &n, &alpha, (float*)a, &lda, b, &ldb);
}

static inline void c_dgemm (char transa, char transb,
                            integer m, integer n, integer k,
                            double alpha, const double* a, integer lda,
                                                const double* b, integer ldb,
                            double  beta,       double* c, integer ldc)
{
    FC_FUNC(dgemm,DGEMM)(&transa, &transb, &m, &n, &k, &alpha, (double*)a, &lda, (double*)b, &ldb, &beta, c, &ldc);
}

static inline void c_dsymm (char side, char uplo,
                            integer m, integer n,
                            double alpha, const double* a, integer lda,
                                                const double* b, integer ldb,
                            double  beta,       double* c, integer ldc)
{
    FC_FUNC(dsymm,DSYMM)(&side, &uplo, &m, &n, &alpha, (double*)a, &lda, (double*)b, &ldb, &beta, c, &ldc);
}

static inline void c_dsyrk (char uplo, char trans,
                            integer n, integer k,
                            double alpha, const double* a, integer lda,
                            double  beta,       double* c, integer ldc)
{
    FC_FUNC(dsyrk,DSYRK)(&uplo, &trans, &n, &k, &alpha, (double*)a, &lda, &beta, c, &ldc);
}

static inline void c_dsyr2k(char uplo, char trans,
                            integer n, integer k,
                            double alpha, const double* a, integer lda,
                                                const double* b, integer ldb,
                            double  beta,       double* c, integer ldc)
{
    FC_FUNC(dsyr2k,DSYR2K)(&uplo, &trans, &n, &k, &alpha, (double*)a, &lda, (double*)b, &ldb, &beta, c, &ldc);
}

static inline void c_dtrmm (char side, char uplo, char transa, char diag,
                            integer m, integer n,
                            double alpha, const double* a, integer lda,
                                                      double* b, integer ldb)
{
    FC_FUNC(dtrmm,DTRMM)(&side, &uplo, &transa, &diag, &m, &n, &alpha, (double*)a, &lda, b, &ldb);
}

static inline void c_dtrsm (char side, char uplo, char transa, char diag,
                            integer m, integer n,
                            double alpha, const double* a, integer lda,
                                                      double* b, integer ldb)
{
    FC_FUNC(dtrsm,DTRSM)(&side, &uplo, &transa, &diag, &m, &n, &alpha, (double*)a, &lda, b, &ldb);
}

static inline void c_cgemm (char transa, char transb,
                            integer m, integer n, integer k,
                            scomplex alpha, const scomplex* a, integer lda,
                                                  const scomplex* b, integer ldb,
                            scomplex  beta,       scomplex* c, integer ldc)
{
    FC_FUNC(cgemm,CGEMM)(&transa, &transb, &m, &n, &k, &alpha, (scomplex*)a, &lda, (scomplex*)b, &ldb, &beta, c, &ldc);
}

static inline void c_chemm (char side, char uplo,
                            integer m, integer n,
                            scomplex alpha, const scomplex* a, integer lda,
                                                  const scomplex* b, integer ldb,
                            scomplex  beta,       scomplex* c, integer ldc)
{
    FC_FUNC(chemm,CHEMM)(&side, &uplo, &m, &n, &alpha, (scomplex*)a, &lda, (scomplex*)b, &ldb, &beta, c, &ldc);
}

static inline void c_csyrk (char uplo, char trans,
                            integer n, integer k,
                            scomplex alpha, const scomplex* a, integer lda,
                            scomplex  beta,       scomplex* c, integer ldc)
{
    FC_FUNC(csyrk,CSYRK)(&uplo, &trans, &n, &k, &alpha, (scomplex*)a, &lda, &beta, c, &ldc);
}

static inline void c_csyr2k(char uplo, char trans,
                            integer n, integer k,
                            scomplex alpha, const scomplex* a, integer lda,
                                                  const scomplex* b, integer ldb,
                            scomplex  beta,       scomplex* c, integer ldc)
{
    FC_FUNC(csyr2k,CSYR2K)(&uplo, &trans, &n, &k, &alpha, (scomplex*)a, &lda, (scomplex*)b, &ldb, &beta, c, &ldc);
}

static inline void c_cherk (char uplo, char trans,
                            integer n, integer k,
                            float alpha, const scomplex* a, integer lda,
                            float  beta,       scomplex* c, integer ldc)
{
    FC_FUNC(cherk,CHERK)(&uplo, &trans, &n, &k, &alpha, (scomplex*)a, &lda, &beta, c, &ldc);
}

static inline void c_cher2k(char uplo, char trans,
                            integer n, integer k,
                            scomplex alpha, const scomplex* a, integer lda,
                                                  const scomplex* b, integer ldb,
                            float  beta,       scomplex* c, integer ldc)
{
    FC_FUNC(cher2k,CHER2K)(&uplo, &trans, &n, &k, &alpha, (scomplex*)a, &lda, (scomplex*)b, &ldb, &beta, c, &ldc);
}

static inline void c_ctrmm (char side, char uplo, char transa, char diag,
                            integer m, integer n,
                            scomplex alpha, const scomplex* a, integer lda,
                                                        scomplex* b, integer ldb)
{
    FC_FUNC(ctrmm,CTRMM)(&side, &uplo, &transa, &diag, &m, &n, &alpha, (scomplex*)a, &lda, b, &ldb);
}

static inline void c_ctrsm (char side, char uplo, char transa, char diag,
                            integer m, integer n,
                            scomplex alpha, const scomplex* a, integer lda,
                                                        scomplex* b, integer ldb)
{
    FC_FUNC(ctrsm,CTRSM)(&side, &uplo, &transa, &diag, &m, &n, &alpha, (scomplex*)a, &lda, b, &ldb);
}

static inline void c_zgemm (char transa, char transb,
                            integer m, integer n, integer k,
                            dcomplex alpha, const dcomplex* a, integer lda,
                                                  const dcomplex* b, integer ldb,
                            dcomplex  beta,       dcomplex* c, integer ldc)
{
    FC_FUNC(zgemm,ZGEMM)(&transa, &transb, &m, &n, &k, &alpha, (dcomplex*)a, &lda, (dcomplex*)b, &ldb, &beta, c, &ldc);
}

static inline void c_zhemm (char side, char uplo,
                            integer m, integer n,
                            dcomplex alpha, const dcomplex* a, integer lda,
                                                  const dcomplex* b, integer ldb,
                            dcomplex  beta,       dcomplex* c, integer ldc)
{
    FC_FUNC(zhemm,ZHEMM)(&side, &uplo, &m, &n, &alpha, (dcomplex*)a, &lda, (dcomplex*)b, &ldb, &beta, c, &ldc);
}

static inline void c_zsyrk (char uplo, char trans,
                            integer n, integer k,
                            dcomplex alpha, const dcomplex* a, integer lda,
                            dcomplex  beta,       dcomplex* c, integer ldc)
{
    FC_FUNC(zsyrk,ZSYRK)(&uplo, &trans, &n, &k, &alpha, (dcomplex*)a, &lda, &beta, c, &ldc);
}

static inline void c_zsyr2k(char uplo, char trans,
                            integer n, integer k,
                            dcomplex alpha, const dcomplex* a, integer lda,
                                                  const dcomplex* b, integer ldb,
                            dcomplex  beta,       dcomplex* c, integer ldc)
{
    FC_FUNC(zsyr2k,ZSYR2K)(&uplo, &trans, &n, &k, &alpha, (dcomplex*)a, &lda, (dcomplex*)b, &ldb, &beta, c, &ldc);
}

static inline void c_zherk (char uplo, char trans,
                            integer n, integer k,
                            double alpha, const dcomplex* a, integer lda,
                            double  beta,       dcomplex* c, integer ldc)
{
    FC_FUNC(zherk,ZHERK)(&uplo, &trans, &n, &k, &alpha, (dcomplex*)a, &lda, &beta, c, &ldc);
}

static inline void c_zher2k(char uplo, char trans,
                            integer n, integer k,
                            dcomplex alpha, const dcomplex* a, integer lda,
                                            const dcomplex* b, integer ldb,
                            double    beta,       dcomplex* c, integer ldc)
{
    FC_FUNC(zher2k,ZHER2K)(&uplo, &trans, &n, &k, &alpha, (dcomplex*)a, &lda, (dcomplex*)b, &ldb, &beta, c, &ldc);
}

static inline void c_ztrmm (char side, char uplo, char transa, char diag,
                            integer m, integer n,
                            dcomplex alpha, const dcomplex* a, integer lda,
                                                        dcomplex* b, integer ldb)
{
    FC_FUNC(ztrmm,ZTRMM)(&side, &uplo, &transa, &diag, &m, &n, &alpha, (dcomplex*)a, &lda, b, &ldb);
}

static inline void c_ztrsm (char side, char uplo, char transa, char diag,
                            integer m, integer n,
                            dcomplex alpha, const dcomplex* a, integer lda,
                                                        dcomplex* b, integer ldb)
{
    FC_FUNC(ztrsm,ZTRSM)(&side, &uplo, &transa, &diag, &m, &n, &alpha, (dcomplex*)a, &lda, b, &ldb);
}

/*
 * #define more familiar names for the C versions
 */
#define srotg  c_srotg
#define srotmg c_srotmg
#define srot   c_srot
#define srotm  c_srotm
#define sswap  c_sswap
#define sscal  c_sscal
#define scopy  c_scopy
#define saxpy  c_saxpy
#define sdot   c_sdot
#define snrm2  c_snrm2
#define sasum  c_sasum
#define isamax c_isamax
#define sgemv  c_sgemv
#define sgbmv  c_sgbmv
#define ssymv  c_ssymv
#define ssbmv  c_ssbmv
#define sspmv  c_sspmv
#define strmv  c_strmv
#define stbmv  c_stbmv
#define stpmv  c_stpmv
#define strsv  c_strsv
#define stbsv  c_stbsv
#define stpsv  c_stpsv
#define sger   c_sger
#define ssyr   c_ssyr
#define sspr   c_sspr
#define ssyr2  c_ssyr2
#define sspr2  c_sspr2
#define sgemm  c_sgemm
#define ssymm  c_ssymm
#define ssyrk  c_ssyrk
#define ssyr2k c_ssyr2k
#define strmm  c_strmm
#define strsm  c_strsm

#define drotg  c_drotg
#define drotmg c_drotmg
#define drot   c_drot
#define drotm  c_drotm
#define dswap  c_dswap
#define dscal  c_dscal
#define dcopy  c_dcopy
#define daxpy  c_daxpy
#define ddot   c_ddot
#define dnrm2  c_dnrm2
#define dasum  c_dasum
#define idamax c_idamax
#define dgemv  c_dgemv
#define dgbmv  c_dgbmv
#define dsymv  c_dsymv
#define dsbmv  c_dsbmv
#define dspmv  c_dspmv
#define dtrmv  c_dtrmv
#define dtbmv  c_dtbmv
#define dtpmv  c_dtpmv
#define dtrsv  c_dtrsv
#define dtbsv  c_dtbsv
#define dtpsv  c_dtpsv
#define dger   c_dger
#define dsyr   c_dsyr
#define dspr   c_dspr
#define dsyr2  c_dsyr2
#define dspr2  c_dspr2
#define dgemm  c_dgemm
#define dsymm  c_dsymm
#define dsyrk  c_dsyrk
#define dsyr2k c_dsyr2k
#define dtrmm  c_dtrmm
#define dtrsm  c_dtrsm

#define crotg  c_crotg
#define csrot  c_csrot
#define cswap  c_cswap
#define cscal  c_cscal
#define csscal c_csscal
#define ccopy  c_ccopy
#define caxpy  c_caxpy
#define cdotu  c_cdotu
#define cdotc  c_cdotc
#define scnrm2 c_scnrm2
#define scasum c_scasum
#define icamax c_icamax
#define cgemv  c_cgemv
#define cgbmv  c_cgbmv
#define chemv  c_chemv
#define chbmv  c_chbmv
#define chpmv  c_chpmv
#define ctrmv  c_ctrmv
#define ctbmv  c_ctbmv
#define ctpmv  c_ctpmv
#define ctrsv  c_ctrsv
#define ctbsv  c_ctbsv
#define ctpsv  c_ctpsv
#define cgeru  c_cgeru
#define cgerc  c_cgerc
#define cher   c_cher
#define chpr   c_chpr
#define cher2  c_cher2
#define chpr2  c_chpr2
#define cgemm  c_cgemm
#define csymm  c_csymm
#define chemm  c_chemm
#define csyrk  c_csyrk
#define csyr2k c_csyr2k
#define cherk  c_cherk
#define cher2k c_cher2k
#define ctrmm  c_ctrmm
#define ctrsm  c_ctrsm

#define zrotg  c_zrotg
#define zdrot  c_zdrot
#define zswap  c_zswap
#define zscal  c_zscal
#define zdscal c_zdscal
#define zcopy  c_zcopy
#define zaxpy  c_zaxpy
#define zdotu  c_zdotu
#define zdotc  c_zdotc
#define dznrm2 c_dznrm2
#define dzasum c_dzasum
#define izamax c_izamax
#define zgemv  c_zgemv
#define zgbmv  c_zgbmv
#define zhemv  c_zhemv
#define zhbmv  c_zhbmv
#define zhpmv  c_zhpmv
#define ztrmv  c_ztrmv
#define ztbmv  c_ztbmv
#define ztpmv  c_ztpmv
#define ztrsv  c_ztrsv
#define ztbsv  c_ztbsv
#define ztpsv  c_ztpsv
#define zgeru  c_zgeru
#define zgerc  c_zgerc
#define zher   c_zher
#define zhpr   c_zhpr
#define zher2  c_zher2
#define zhpr2  c_zhpr2
#define zgemm  c_zgemm
#define zsymm  c_zsymm
#define zhemm  c_zhemm
#define zsyrk  c_zsyrk
#define zsyr2k c_zsyr2k
#define zherk  c_zherk
#define zher2k c_zher2k
#define ztrmm  c_ztrmm
#define ztrsm  c_ztrsm

#ifdef __cplusplus
}

namespace blas
{

/******************************************************************************
 *
 * Level 1 BLAS, C++ overloads
 *
 *****************************************************************************/
inline void rotg(float* a, float* b, float* c, float* s)
{
    srotg(a, b, c, s);
}

inline void rotg(double* a, double* b, double* c, double* s)
{
    drotg(a, b, c, s);
}

inline void rotg(scomplex* a, scomplex* b, float* c, scomplex* s)
{
    crotg(a, b, c, s);
}

inline void rotg(dcomplex* a, dcomplex* b, double* c, dcomplex* s)
{
    zrotg(a, b, c, s);
}

inline void rotmg(float* d1, float* d2, float* a, float b, float* param)
{
    srotmg(d1, d2, a, b, param);
}

inline void rotmg(double* d1, double* d2, double* a, double b, double* param)
{
    drotmg(d1, d2, a, b, param);
}

inline void rot(integer n, float* x, integer incx,
                                 float* y, integer incy, float c, float s)
{
    srot(n, x, incx, y, incy, c, s);
}

inline void rot(integer n, double* x, integer incx,
                                 double* y, integer incy, double c, double s)
{
    drot(n, x, incx, y, incy, c, s);
}

inline void rot(integer n, scomplex* x, integer incx,
                                 scomplex* y, integer incy, float c, float s)
{
    csrot(n, x, incx, y, incy, c, s);
}

inline void rot(integer n, dcomplex* x, integer incx,
                                 dcomplex* y, integer incy, double c, double s)
{
    zdrot(n, x, incx, y, incy, c, s);
}

inline void rotm(integer n, float* x, integer incx,
                                  float* y, integer incy, float* param)
{
    srotm(n, x, incx, y, incy, param);
}

inline void rotm(integer n, double* x, integer incx,
                                  double* y, integer incy, double* param)
{
    drotm(n, x, incx, y, incy, param);
}

inline void swap(integer n, float* x, integer incx,
                                  float* y, integer incy)
{
    sswap(n, x, incx, y, incy);
}

inline void swap(integer n, double* x, integer incx,
                                  double* y, integer incy)
{
    dswap(n, x, incx, y, incy);
}

inline void swap(integer n, scomplex* x, integer incx,
                                  scomplex* y, integer incy)
{
    cswap(n, x, incx, y, incy);
}

inline void swap(integer n, dcomplex* x, integer incx,
                                  dcomplex* y, integer incy)
{
    zswap(n, x, incx, y, incy);
}

inline void scal(integer n, float alpha, float* x, integer incx)
{
    sscal(n, alpha, x, incx);
}

inline void scal(integer n, double alpha, double* x, integer incx)
{
    dscal(n, alpha, x, incx);
}

inline void scal(integer n, scomplex alpha, scomplex* x, integer incx)
{
    cscal(n, alpha, x, incx);
}

inline void scal(integer n, dcomplex alpha, dcomplex* x, integer incx)
{
    zscal(n, alpha, x, incx);
}

inline void scal(integer n, float alpha, scomplex* x, integer incx)
{
    csscal(n, alpha, x, incx);
}

inline void scal(integer n, float alpha, dcomplex* x, integer incx)
{
    zdscal(n, alpha, x, incx);
}

inline void copy(integer n, const float* x, integer incx,
                                        float* y, integer incy)
{
    scopy(n, x, incx, y, incy);
}

inline void copy(integer n, const double* x, integer incx,
                                        double* y, integer incy)
{
    dcopy(n, x, incx, y, incy);
}

inline void copy(integer n, const scomplex* x, integer incx,
                                        scomplex* y, integer incy)
{
    ccopy(n, x, incx, y, incy);
}

inline void copy(integer n, const dcomplex* x, integer incx,
                                        dcomplex* y, integer incy)
{
    zcopy(n, x, incx, y, incy);
}

inline void axpy(integer n, float alpha, const float* x, integer incx,
                                                           float* y, integer incy)
{
    saxpy(n, alpha, x, incx, y, incy);
}

inline void axpy(integer n, double alpha, const double* x, integer incx,
                                                            double* y, integer incy)
{
    daxpy(n, alpha, x, incx, y, incy);
}

inline void axpy(integer n, scomplex alpha, const scomplex* x, integer incx,
                                                              scomplex* y, integer incy)
{
    caxpy(n, alpha, x, incx, y, incy);
}

inline void axpy(integer n, dcomplex alpha, const dcomplex* x, integer incx,
                                                              dcomplex* y, integer incy)
{
    zaxpy(n, alpha, x, incx, y, incy);
}

inline float dotc(integer n, const float* x, integer incx,
                                   const float* y, integer incy)
{
    return sdot(n, x, incx, y, incy);
}

inline double dotc(integer n, const double* x, integer incx,
                                    const double* y, integer incy)
{
    return ddot(n, x, incx, y, incy);
}

inline scomplex dotc(integer n, const scomplex* x, integer incx,
                                      const scomplex* y, integer incy)
{
    return cdotc(n, x, incx, y, incy);
}

inline dcomplex dotc(integer n, const dcomplex* x, integer incx,
                                      const dcomplex* y, integer incy)
{
    return zdotc(n, x, incx, y, incy);
}

inline float dotu(integer n, const float* x, integer incx,
                                   const float* y, integer incy)
{
    return sdot(n, x, incx, y, incy);
}

inline double dotu(integer n, const double* x, integer incx,
                                    const double* y, integer incy)
{
    return ddot(n, x, incx, y, incy);
}

inline scomplex dotu(integer n, const scomplex* x, integer incx,
                                      const scomplex* y, integer incy)
{
    return cdotu(n, x, incx, y, incy);
}

inline dcomplex dotu(integer n, const dcomplex* x, integer incx,
                                      const dcomplex* y, integer incy)
{
    return zdotu(n, x, incx, y, incy);
}

#define dot dotc

inline float nrm2(integer n, const float* x, integer incx)
{
    return snrm2(n, x, incx);
}

inline double nrm2(integer n, const double* x, integer incx)
{
    return dnrm2(n, x, incx);
}

inline float nrm2(integer n, const scomplex* x, integer incx)
{
    return scnrm2(n, x, incx);
}

inline double nrm2(integer n, const dcomplex* x, integer incx)
{
    return dznrm2(n, x, incx);
}

inline float asum(integer n, const float* x, integer incx)
{
    return sasum(n, x, incx);
}

inline double asum(integer n, const double* x, integer incx)
{
    return dasum(n, x, incx);
}

inline float asum(integer n, const scomplex* x, integer incx)
{
    return scasum(n, x, incx);
}

inline double asum(integer n, const dcomplex* x, integer incx)
{
    return dzasum(n, x, incx);
}

inline integer iamax(integer n, const float* x, integer incx)
{
    return c_isamax(n, x, incx);
}

inline integer iamax(integer n, const double* x, integer incx)
{
    return c_idamax(n, x, incx);
}

inline integer iamax(integer n, const scomplex* x, integer incx)
{
    return c_icamax(n, x, incx);
}

inline integer iamax(integer n, const dcomplex* x, integer incx)
{
    return c_izamax(n, x, incx);
}

inline float amax(integer n, const float* x, integer incx)
{
    return x[c_isamax(n, x, incx)];
}

inline double amax(integer n, const double* x, integer incx)
{
    return x[c_idamax(n, x, incx)];
}

inline scomplex amax(integer n, const scomplex* x, integer incx)
{
    return x[c_icamax(n, x, incx)];
}

inline dcomplex amax(integer n, const dcomplex* x, integer incx)
{
    return x[c_izamax(n, x, incx)];
}

/******************************************************************************
 *
 * Level 2 BLAS, C++ overloads
 *
 *****************************************************************************/
inline void gemv(char trans, integer m, integer n,
                 float alpha, const float* a, integer lda,
                                    const float* x, integer incx,
                 float  beta,       float* y, integer incy)
{
    sgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

inline void gemv(char trans, integer m, integer n,
                 double alpha, const double* a, integer lda,
                                     const double* x, integer incx,
                 double  beta,       double* y, integer incy)
{
    dgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

inline void gemv(char trans, integer m, integer n,
                 scomplex alpha, const scomplex* a, integer lda,
                                       const scomplex* x, integer incx,
                 scomplex  beta,       scomplex* y, integer incy)
{
    cgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

inline void gemv(char trans, integer m, integer n,
                 dcomplex alpha, const dcomplex* a, integer lda,
                                       const dcomplex* x, integer incx,
                 dcomplex  beta,       dcomplex* y, integer incy)
{
    zgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

inline void gbmv(char trans,
                 integer m, integer n, integer kl, integer ku,
                 float alpha, const float* a, integer lda,
                                    const float* x, integer incx,
                 float  beta,       float* y, integer incy)
{
    sgbmv(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

inline void gbmv(char trans,
                 integer m, integer n, integer kl, integer ku,
                 double alpha, const double* a, integer lda,
                                     const double* x, integer incx,
                 double  beta,       double* y, integer incy)
{
    dgbmv(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

inline void gbmv(char trans,
                 integer m, integer n, integer kl, integer ku,
                 scomplex alpha, const scomplex* a, integer lda,
                                       const scomplex* x, integer incx,
                 scomplex  beta,       scomplex* y, integer incy)
{
    cgbmv(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

inline void gbmv(char trans,
                 integer m, integer n, integer kl, integer ku,
                 dcomplex alpha, const dcomplex* a, integer lda,
                                       const dcomplex* x, integer incx,
                 dcomplex  beta,       dcomplex* y, integer incy)
{
    zgbmv(trans, m, n, kl, ku, alpha, a, lda, x, incx, beta, y, incy);
}

inline void hemv(char uplo, integer n,
                 float alpha, const float* a, integer lda,
                                    const float* x, integer incx,
                 float  beta,       float* y, integer incy)
{
    ssymv(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

inline void hemv(char uplo, integer n,
                 double alpha, const double* a, integer lda,
                                     const double* x, integer incx,
                 double  beta,       double* y, integer incy)
{
    dsymv(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

inline void hemv(char uplo, integer n,
                 scomplex alpha, const scomplex* a, integer lda,
                                       const scomplex* x, integer incx,
                 scomplex  beta,       scomplex* y, integer incy)
{
    chemv(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

inline void hemv(char uplo, integer n,
                 dcomplex alpha, const dcomplex* a, integer lda,
                                       const dcomplex* x, integer incx,
                 dcomplex  beta,       dcomplex* y, integer incy)
{
    zhemv(uplo, n, alpha, a, lda, x, incx, beta, y, incy);
}

inline void hbmv(char uplo, integer n, integer k,
                 float alpha, const float* a, integer lda,
                                    const float* x, integer incx,
                 float  beta,       float* y, integer incy)
{
    ssbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

inline void hbmv(char uplo, integer n, integer k,
                 double alpha, const double* a, integer lda,
                                     const double* x, integer incx,
                 double  beta,       double* y, integer incy)
{
    dsbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

inline void hbmv(char uplo, integer n, integer k,
                 scomplex alpha, const scomplex* a, integer lda,
                                       const scomplex* x, integer incx,
                 scomplex  beta,       scomplex* y, integer incy)
{
    chbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

inline void hbmv(char uplo, integer n, integer k,
                 dcomplex alpha, const dcomplex* a, integer lda,
                                       const dcomplex* x, integer incx,
                 dcomplex  beta,       dcomplex* y, integer incy)
{
    zhbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy);
}

inline void hpmv(char uplo, integer n,
                 float alpha, const float* ap,
                                    const float*  x, integer incx,
                 float  beta,       float*  y, integer incy)
{
    sspmv(uplo, n, alpha, ap, x, incx, beta, y, incy);
}

inline void hpmv(char uplo, integer n,
                 double alpha, const double* ap,
                                     const double*  x, integer incx,
                 double  beta,       double*  y, integer incy)
{
    dspmv(uplo, n, alpha, ap, x, incx, beta, y, incy);
}

inline void hpmv(char uplo, integer n,
                 scomplex alpha, const scomplex* ap,
                                       const scomplex*  x, integer incx,
                 scomplex  beta,       scomplex*  y, integer incy)
{
    chpmv(uplo, n, alpha, ap, x, incx, beta, y, incy);
}

inline void hpmv(char uplo, integer n,
                 dcomplex alpha, const dcomplex* ap,
                                       const dcomplex*  x, integer incx,
                 dcomplex  beta,       dcomplex*  y, integer incy)
{
    zhpmv(uplo, n, alpha, ap, x, incx, beta, y, incy);
}

inline void trmv(char uplo, char trans, char diag, integer n,
                 const float* a, integer lda,
                       float* x, integer incx)
{
    strmv(uplo, trans, diag, n, a, lda, x, incx);
}

inline void trmv(char uplo, char trans, char diag, integer n,
                 const double* a, integer lda,
                       double* x, integer incx)
{
    dtrmv(uplo, trans, diag, n, a, lda, x, incx);
}

inline void trmv(char uplo, char trans, char diag, integer n,
                 const scomplex* a, integer lda,
                       scomplex* x, integer incx)
{
    ctrmv(uplo, trans, diag, n, a, lda, x, incx);
}

inline void trmv(char uplo, char trans, char diag, integer n,
                 const dcomplex* a, integer lda,
                       dcomplex* x, integer incx)
{
    ztrmv(uplo, trans, diag, n, a, lda, x, incx);
}

inline void tbmv(char uplo, char trans, char diag,
                 integer n, integer k,
                 const float* a, integer lda,
                       float* x, integer incx)
{
    stbmv(uplo, trans, diag, n, k, a, lda, x, incx);
}

inline void tbmv(char uplo, char trans, char diag,
                 integer n, integer k,
                 const double* a, integer lda,
                       double* x, integer incx)
{
    dtbmv(uplo, trans, diag, n, k, a, lda, x, incx);
}

inline void tbmv(char uplo, char trans, char diag,
                 integer n, integer k,
                 const scomplex* a, integer lda,
                       scomplex* x, integer incx)
{
    ctbmv(uplo, trans, diag, n, k, a, lda, x, incx);
}

inline void tbmv(char uplo, char trans, char diag,
                 integer n, integer k,
                 const dcomplex* a, integer lda,
                       dcomplex* x, integer incx)
{
    ztbmv(uplo, trans, diag, n, k, a, lda, x, incx);
}

inline void tpmv(char uplo, char trans, char diag, integer n,
                 const float* ap, float* x, integer incx)
{
    stpmv(uplo, trans, diag, n, ap, x, incx);
}

inline void tpmv(char uplo, char trans, char diag, integer n,
                 const double* ap, double* x, integer incx)
{
    dtpmv(uplo, trans, diag, n, ap, x, incx);
}

inline void tpmv(char uplo, char trans, char diag, integer n,
                 const scomplex* ap, scomplex* x, integer incx)
{
    ctpmv(uplo, trans, diag, n, ap, x, incx);
}

inline void tpmv(char uplo, char trans, char diag, integer n,
                 const dcomplex* ap, dcomplex* x, integer incx)
{
    ztpmv(uplo, trans, diag, n, ap, x, incx);
}

inline void trsv(char uplo, char trans, char diag, integer n,
                 const float* a, integer lda,
                       float* x, integer incx)
{
    strsv(uplo, trans, diag, n, a, lda, x, incx);
}

inline void trsv(char uplo, char trans, char diag, integer n,
                 const double* a, integer lda,
                       double* x, integer incx)
{
    dtrsv(uplo, trans, diag, n, a, lda, x, incx);
}

inline void trsv(char uplo, char trans, char diag, integer n,
                 const scomplex* a, integer lda,
                       scomplex* x, integer incx)
{
    ctrsv(uplo, trans, diag, n, a, lda, x, incx);
}

inline void trsv(char uplo, char trans, char diag, integer n,
                 const dcomplex* a, integer lda,
                       dcomplex* x, integer incx)
{
    ztrsv(uplo, trans, diag, n, a, lda, x, incx);
}

inline void tbsv(char uplo, char trans, char diag,
                 integer n, integer k,
                 const float* a, integer lda,
                       float* x, integer incx)
{
    stbsv(uplo, trans, diag, n, k, a, lda, x, incx);
}

inline void tbsv(char uplo, char trans, char diag,
                 integer n, integer k,
                 const double* a, integer lda,
                       double* x, integer incx)
{
    dtbsv(uplo, trans, diag, n, k, a, lda, x, incx);
}

inline void tbsv(char uplo, char trans, char diag,
                 integer n, integer k,
                 const scomplex* a, integer lda,
                       scomplex* x, integer incx)
{
    ctbsv(uplo, trans, diag, n, k, a, lda, x, incx);
}

inline void tbsv(char uplo, char trans, char diag,
                 integer n, integer k,
                 const dcomplex* a, integer lda,
                       dcomplex* x, integer incx)
{
    ztbsv(uplo, trans, diag, n, k, a, lda, x, incx);
}

inline void tpsv(char uplo, char trans, char diag, integer n,
                 const float* ap, float* x, integer incx)
{
    stpsv(uplo, trans, diag, n, ap, x, incx);
}

inline void tpsv(char uplo, char trans, char diag, integer n,
                 const double* ap, double* x, integer incx)
{
    dtpsv(uplo, trans, diag, n, ap, x, incx);
}

inline void tpsv(char uplo, char trans, char diag, integer n,
                 const scomplex* ap, scomplex* x, integer incx)
{
    ctpsv(uplo, trans, diag, n, ap, x, incx);
}

inline void tpsv(char uplo, char trans, char diag, integer n,
                const dcomplex* ap, dcomplex* x, integer incx)
{
    ztpsv(uplo, trans, diag, n, ap, x, incx);
}

inline void gerc(integer m, integer n,
                 float alpha, const float* x, integer incx,
                                    const float* y, integer incy,
                                          float* a, integer lda)
{
    sger(m, n, alpha, x, incx, y, incy, a, lda);
}

inline void gerc(integer m, integer n,
                 double alpha, const double* x, integer incx,
                                     const double* y, integer incy,
                                           double* a, integer lda)
{
    dger(m, n, alpha, x, incx, y, incy, a, lda);
}

inline void gerc(integer m, integer n,
                 scomplex alpha, const scomplex* x, integer incx,
                                       const scomplex* y, integer incy,
                                             scomplex* a, integer lda)
{
    cgerc(m, n, alpha, x, incx, y, incy, a, lda);
}

inline void gerc(integer m, integer n,
                 dcomplex alpha, const dcomplex* x, integer incx,
                                       const dcomplex* y, integer incy,
                                             dcomplex* a, integer lda)
{
    zgerc(m, n, alpha, x, incx, y, incy, a, lda);
}

inline void geru(integer m, integer n,
                 float alpha, const float* x, integer incx,
                                    const float* y, integer incy,
                                          float* a, integer lda)
{
    sger(m, n, alpha, x, incx, y, incy, a, lda);
}

inline void geru(integer m, integer n,
                 double alpha, const double* x, integer incx,
                                     const double* y, integer incy,
                                           double* a, integer lda)
{
    dger(m, n, alpha, x, incx, y, incy, a, lda);
}

inline void geru(integer m, integer n,
                 scomplex alpha, const scomplex* x, integer incx,
                                       const scomplex* y, integer incy,
                                             scomplex* a, integer lda)
{
    cgeru(m, n, alpha, x, incx, y, incy, a, lda);
}

inline void geru(integer m, integer n,
                 dcomplex alpha, const dcomplex* x, integer incx,
                                       const dcomplex* y, integer incy,
                                             dcomplex* a, integer lda)
{
    zgeru(m, n, alpha, x, incx, y, incy, a, lda);
}

#define ger gerc

inline void her(char uplo, integer n,
                float alpha, const float* x, integer incx,
                                         float* a, integer lda)
{
    ssyr(uplo, n, alpha, x, incx, a, lda);
}

inline void her(char uplo, integer n,
                double alpha, const double* x, integer incx,
                                          double* a, integer lda)
{
    dsyr(uplo, n, alpha, x, incx, a, lda);
}

inline void her(char uplo, integer n,
                float alpha, const scomplex* x, integer incx,
                                            scomplex* a, integer lda)
{
    cher(uplo, n, alpha, x, incx, a, lda);
}

inline void her(char uplo, integer n,
                double alpha, const dcomplex* x, integer incx,
                                            dcomplex* a, integer lda)
{
    zher(uplo, n, alpha, x, incx, a, lda);
}

inline void hpr(char uplo, integer n,
                float alpha, const float* x, integer incx,
                                         float* ap)
{
    sspr(uplo, n, alpha, x, incx, ap);
}

inline void hpr(char uplo, integer n,
                double alpha, const double* x, integer incx,
                                          double* ap)
{
    dspr(uplo, n, alpha, x, incx, ap);
}

inline void hpr(char uplo, integer n,
                float alpha, const scomplex* x, integer incx,
                                            scomplex* ap)
{
    chpr(uplo, n, alpha, x, incx, ap);
}

inline void hpr(char uplo, integer n,
                double alpha, const dcomplex* x, integer incx,
                                            dcomplex* ap)
{
    zhpr(uplo, n, alpha, x, incx, ap);
}

inline void her2(char uplo, integer n,
                 float alpha, const float* x, integer incx,
                                    const float* y, integer incy,
                                          float* a, integer lda)
{
    ssyr2(uplo, n, alpha, x, incx, y, incy, a, lda);
}

inline void her2(char uplo, integer n,
                 double alpha, const double* x, integer incx,
                                     const double* y, integer incy,
                                           double* a, integer lda)
{
    dsyr2(uplo, n, alpha, x, incx, y, incy, a, lda);
}
inline void her2(char uplo, integer n,
                 scomplex alpha, const scomplex* x, integer incx,
                                       const scomplex* y, integer incy,
                                             scomplex* a, integer lda)
{
    cher2(uplo, n, alpha, x, incx, y, incy, a, lda);
}

inline void her2(char uplo, integer n,
                 dcomplex alpha, const dcomplex* x, integer incx,
                                       const dcomplex* y, integer incy,
                                             dcomplex* a, integer lda)
{
    zher2(uplo, n, alpha, x, incx, y, incy, a, lda);
}

inline void hpr2(char uplo, integer n,
                 float alpha, const float* x, integer incx,
                                    const float* y, integer incy,
                                          float* ap)
{
    sspr2(uplo, n, alpha, x, incx, y, incy, ap);
}

inline void hpr2(char uplo, integer n,
                 double alpha, const double* x, integer incx,
                                     const double* y, integer incy,
                                           double* ap)
{
    dspr2(uplo, n, alpha, x, incx, y, incy, ap);
}

inline void hpr2(char uplo, integer n,
                 scomplex alpha, const scomplex* x, integer incx,
                                       const scomplex* y, integer incy,
                                             scomplex* ap)
{
    chpr2(uplo, n, alpha, x, incx, y, incy, ap);
}

inline void hpr2(char uplo, integer n,
                 dcomplex alpha, const dcomplex* x, integer incx,
                                       const dcomplex* y, integer incy,
                                             dcomplex* ap)
{
    zhpr2(uplo, n, alpha, x, incx, y, incy, ap);
}

/******************************************************************************
 *
 * Level 3 BLAS, C++ overloads
 *
 *****************************************************************************/
inline void gemm(char transa, char transb,
                 integer m, integer n, integer k,
                 float alpha, const float* a, integer lda,
                                    const float* b, integer ldb,
                 float  beta,       float* c, integer ldc)
{
    sgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void gemm(char transa, char transb,
                 integer m, integer n, integer k,
                 double alpha, const double* a, integer lda,
                                     const double* b, integer ldb,
                 double  beta,       double* c, integer ldc)
{
    dgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void gemm(char transa, char transb,
                 integer m, integer n, integer k,
                 scomplex alpha, const scomplex* a, integer lda,
                                       const scomplex* b, integer ldb,
                 scomplex  beta,       scomplex* c, integer ldc)
{
    cgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void gemm(char transa, char transb,
                 integer m, integer n, integer k,
                 dcomplex alpha, const dcomplex* a, integer lda,
                                       const dcomplex* b, integer ldb,
                 dcomplex  beta,       dcomplex* c, integer ldc)
{
    zgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void hemm(char side, char uplo,
                 integer m, integer n,
                 float alpha, const float* a, integer lda,
                                    const float* b, integer ldb,
                 float  beta,       float* c, integer ldc)
{
    ssymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void hemm(char side, char uplo,
                 integer m, integer n,
                 double alpha, const double* a, integer lda,
                                     const double* b, integer ldb,
                 double  beta,       double* c, integer ldc)
{
    dsymm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void hemm(char side, char uplo,
                 integer m, integer n,
                 scomplex alpha, const scomplex* a, integer lda,
                                       const scomplex* b, integer ldb,
                 scomplex  beta,       scomplex* c, integer ldc)
{
    chemm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void hemm(char side, char uplo,
                 integer m, integer n,
                 dcomplex alpha, const dcomplex* a, integer lda,
                                       const dcomplex* b, integer ldb,
                 dcomplex  beta,       dcomplex* c, integer ldc)
{
    zhemm(side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void syrk(char uplo, char trans,
                 integer n, integer k,
                 float alpha, const float* a, integer lda,
                 float  beta,       float* c, integer ldc)
{
    ssyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

inline void syrk(char uplo, char trans,
                 integer n, integer k,
                 double alpha, const double* a, integer lda,
                 double  beta,       double* c, integer ldc)
{
    dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

inline void syrk(char uplo, char trans,
                 integer n, integer k,
                 scomplex alpha, const scomplex* a, integer lda,
                 scomplex  beta,       scomplex* c, integer ldc)
{
    csyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

inline void syrk(char uplo, char trans,
                 integer n, integer k,
                 dcomplex alpha, const dcomplex* a, integer lda,
                 dcomplex  beta,       dcomplex* c, integer ldc)
{
    zsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

inline void herk(char uplo, char trans,
                 integer n, integer k,
                 float alpha, const float* a, integer lda,
                 float  beta,       float* c, integer ldc)
{
    ssyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

inline void herk(char uplo, char trans,
                 integer n, integer k,
                 double alpha, const double* a, integer lda,
                 double  beta,       double* c, integer ldc)
{
    dsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

inline void herk(char uplo, char trans,
                 integer n, integer k,
                 float alpha, const scomplex* a, integer lda,
                 float  beta,       scomplex* c, integer ldc)
{
    cherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

inline void herk(char uplo, char trans,
                 integer n, integer k,
                 double alpha, const dcomplex* a, integer lda,
                 double  beta,       dcomplex* c, integer ldc)
{
    zherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc);
}

inline void syr2k(char uplo, char trans,
                  integer n, integer k,
                  float alpha, const float* a, integer lda,
                                     const float* b, integer ldb,
                  float  beta,       float* c, integer ldc)
{
    ssyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void syr2k(char uplo, char trans,
                  integer n, integer k,
                  double alpha, const double* a, integer lda,
                                      const double* b, integer ldb,
                  double  beta,       double* c, integer ldc)
{
    dsyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void syr2k(char uplo, char trans,
                  integer n, integer k,
                  scomplex alpha, const scomplex* a, integer lda,
                                        const scomplex* b, integer ldb,
                  scomplex  beta,       scomplex* c, integer ldc)
{
    csyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void syr2k(char uplo, char trans,
                  integer n, integer k,
                  dcomplex alpha, const dcomplex* a, integer lda,
                                        const dcomplex* b, integer ldb,
                  dcomplex  beta,       dcomplex* c, integer ldc)
{
    zsyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void her2k(char uplo, char trans,
                  integer n, integer k,
                  float alpha, const float* a, integer lda,
                                     const float* b, integer ldb,
                  float  beta,       float* c, integer ldc)
{
    ssyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void her2k(char uplo, char trans,
                  integer n, integer k,
                  double alpha, const double* a, integer lda,
                                      const double* b, integer ldb,
                  double  beta,       double* c, integer ldc)
{
    dsyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void her2k(char uplo, char trans,
                  integer n, integer k,
                  scomplex alpha, const scomplex* a, integer lda,
                                        const scomplex* b, integer ldb,
                  const    float  beta,       scomplex* c, integer ldc)
{
    cher2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void her2k(char uplo, char trans,
                  integer n, integer k,
                  dcomplex alpha, const dcomplex* a, integer lda,
                                        const dcomplex* b, integer ldb,
                  const   double  beta,       dcomplex* c, integer ldc)
{
    zher2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

inline void trmm(char side, char uplo, char transa, char diag,
                 integer m, integer n,
                 float alpha, const float* a, integer lda,
                                          float* b, integer ldb)
{
    strmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

inline void trmm(char side, char uplo, char transa, char diag,
                 integer m, integer n,
                 double alpha, const double* a, integer lda,
                                           double* b, integer ldb)
{
    dtrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

inline void trmm(char side, char uplo, char transa, char diag,
                 integer m, integer n,
                 scomplex alpha, const scomplex* a, integer lda,
                                             scomplex* b, integer ldb)
{
    ctrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

inline void trmm(char side, char uplo, char transa, char diag,
                 integer m, integer n,
                 dcomplex alpha, const dcomplex* a, integer lda,
                                             dcomplex* b, integer ldb)
{
    ztrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

inline void trsm(char side, char uplo, char transa, char diag,
                 integer m, integer n,
                 float alpha, const float* a, integer lda,
                                          float* b, integer ldb)
{
    strsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

inline void trsm(char side, char uplo, char transa, char diag,
                 integer m, integer n,
                 double alpha, const double* a, integer lda,
                                           double* b, integer ldb)
{
    dtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

inline void trsm(char side, char uplo, char transa, char diag,
                 integer m, integer n,
                 scomplex alpha, const scomplex* a, integer lda,
                                             scomplex* b, integer ldb)
{
    ctrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

inline void trsm(char side, char uplo, char transa, char diag,
                 integer m, integer n,
                 dcomplex alpha, const dcomplex* a, integer lda,
                                             dcomplex* b, integer ldb)
{
    ztrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb);
}

}
#endif

#endif
