#ifndef _LAWRAP_FORTRAN_H_
#define _LAWRAP_FORTRAN_H_

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#ifndef LAWRAP_COMPLEX_DEFINED

#ifdef __cplusplus

#include <complex>
#define creal real
#define crealf real
#define cimag imag
#define cimagf imag

typedef std::complex<float> scomplex;
typedef std::complex<double> dcomplex;

#elif __STDC_VERSION__ >= 199901L

#include <complex.h>

typedef float complex scomplex;
typedef double complex dcomplex;

#else

#pragma pack(push)
#pragma pack(4)
typedef struct scomplex_
{
    float real;
    float imag;
} scomplex;
#pragma pack(8)
typedef struct dcomplex_
{
    double real;
    double imag;
} dcomplex;
#pragma pack(pop)

//TODO: add creal etc.

#endif

#endif

/*
 * integer == integer*4 is usually a safe bet
 */
#ifndef FORTRAN_INTEGER_SIZE
#define FORTRAN_INTEGER_SIZE 4
#endif

#if FORTRAN_INTEGER_SIZE == 1
typedef int8_t integer;
#elif FORTRAN_INTEGER_SIZE == 2
typedef int16_t integer;
#elif FORTRAN_INTEGER_SIZE == 4
typedef int32_t integer;
#elif FORTRAN_INTEGER_SIZE == 8
typedef int64_t integer;
#else
#error "A valid FORTRAN_INTEGER_SIZE must be specified"
#endif

typedef integer logical;

/*
 * lowercase symbols with underscore are most common
 */
#ifndef FC_FUNC
#define FC_FUNC(lower,UPPER) lower ## _
#endif

#endif
