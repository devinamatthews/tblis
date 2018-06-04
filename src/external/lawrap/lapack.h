#ifndef _LAWRAP_LAPACK_H_
#define _LAWRAP_LAPACK_H_

#include <stdlib.h>

#include "fortran.h"
#include "blas.h"

#ifndef LAWRAP_MALLOC
#ifdef __cplusplus
#define LAWRAP_MALLOC(type,num) new type[num]
#else
#define LAWRAP_MALLOC(type,num) (type*)malloc(sizeof(type)*(num))
#endif
#endif

#ifndef LAWRAP_FREE
#ifdef __cplusplus
#define LAWRAP_FREE(x) delete[] x
#else
#define LAWRAP_FREE(x) free(x)
#endif
#endif

#ifdef __cplusplus
namespace LAWrap
{
#endif

enum {AXBX=1, ABX=2, BAX=3};

#ifdef __cplusplus
}
#endif

#ifndef LAWRAP_F77_INTERFACE_DEFINED
#include "internal/lapack_f77.h"
#endif
#include "internal/lapack_c89.h"
#ifdef __cplusplus
#include "internal/lapack_c++.hpp"
#endif

#endif
