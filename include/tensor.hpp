#ifndef _TENSOR_HPP_
#define _TENSOR_HPP_

#include "config.h"

#include "util/blis.hpp"

#define scomplex blis::sComplex
#define dcomplex blis::dComplex
#include "tensor.h"
#undef scomplex
#undef dcomplex

#include "core/tensor_class.hpp"
#include "core/tensor_iface.hpp"
#include "core/tensor_templates.hpp"

#endif
