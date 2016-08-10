#ifndef _TBLIS_MARRAY_HPP_
#define _TBLIS_MARRAY_HPP_

#include <string>

#include "tblis_basic_types.hpp"

#define MARRAY_DEFAULT_LAYOUT COLUMN_MAJOR
#include "external/marray/include/varray.hpp"
#include "external/marray/include/marray.hpp"

namespace tblis
{

template <typename T>
using const_tensor_view = MArray::const_varray_view<T>;

template <typename T>
using tensor_view = MArray::varray_view<T>;

template <typename T, typename Allocator=aligned_allocator<T,64>>
using tensor = MArray::varray<T, Allocator>;

using MArray::const_marray_view;
using MArray::marray_view;

template <typename T, unsigned ndim, typename Allocator=aligned_allocator<T>>
using marray = MArray::marray<T, ndim, Allocator>;

using MArray::const_matrix_view;
using MArray::matrix_view;

template <typename T, typename Allocator=aligned_allocator<T>>
using matrix = MArray::matrix<T, Allocator>;

using MArray::const_row_view;
using MArray::row_view;

template <typename T, typename Allocator=aligned_allocator<T>>
using row = MArray::row<T, Allocator>;

using MArray::Layout;
using MArray::COLUMN_MAJOR;
using MArray::ROW_MAJOR;
using MArray::DEFAULT;

using MArray::uninitialized_t;
using MArray::uninitialized;

using MArray::make_array;
using MArray::make_vector;

using MArray::range_t;
using MArray::range;

}

#endif
