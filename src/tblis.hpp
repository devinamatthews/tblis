#ifndef _TBLIS_HPP_
#define _TBLIS_HPP_

#include "config.h"

#define BLIS_DISABLE_BLAS2BLIS
#include "blis++/blis++.hpp"

#include "tensor.h"

namespace tblis
{

using namespace blis;

void tblis_init();

void tblis_finalize();

}

#include "blis-like/tblis_constants.hpp"
#include "blis-like/tblis_scatter_matrix.hpp"
#include "blis-like/tblis_scatter_vector.hpp"

#include "blis-like/1m/tblis_normfm.hpp"

#include "blis-like/3/tblis_partm.hpp"
#include "blis-like/3/tblis_packm.hpp"
#include "blis-like/3/tblis_gemm_ukr.hpp"
#include "blis-like/3/tblis_gemm.hpp"

#include "core/tensor_iterator.hpp"
#include "core/tensor_class.hpp"
#include "core/tensor_iface.hpp"
#include "core/tensor_templates.hpp"
#include "core/tensor_slicer.hpp"
#include "core/tensor_partitioner.hpp"

#endif
