#ifndef _TBLIS_UTIL_CONFIGS_H_
#define _TBLIS_UTIL_CONFIGS_H_

#include "basic_types.h"

#ifdef __cplusplus
namespace tblis {
extern "C" {
#endif

const tblis_config* tblis_get_config(const char* name);

#ifdef __cplusplus
}}
#endif

#endif
