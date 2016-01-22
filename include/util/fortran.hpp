#ifndef _TENSOR_UTIL_FORTRAN_H_
#define _TENSOR_UTIL_FORTRAN_H_

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>

#include <blis.h>

typedef integer logical;

/*
 * lowercase symbols with underscore are most common
 */
#ifndef FC_FUNC
#define FC_FUNC(lower,UPPER) lower ## _
#endif

#endif
