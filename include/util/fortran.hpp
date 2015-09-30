#ifndef _TENSOR_UTIL_FORTRAN_H_
#define _TENSOR_UTIL_FORTRAN_H_

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

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
