#ifndef _TBLIS_UTIL_HPP_
#define _TBLIS_UTIL_HPP_

#define MARRAY_DEFAULT_LAYOUT COLUMN_MAJOR
#include "external/marray/include/varray.hpp"
#include "external/stl_ext/include/algorithm.hpp"
#include "external/stl_ext/include/vector.hpp"
#include "external/stl_ext/include/complex.hpp"

#define TBLIS_STRINGIZE_(...) #__VA_ARGS__
#define TBLIS_STRINGIZE(...) TBLIS_STRINGIZE_(__VA_ARGS__)
#define TBLIS_CONCAT_(x,y) x##y
#define TBLIS_CONCAT(x,y) TBLIS_CONCAT_(x,y)
#define TBLIS_FIRST_ARG(arg,...) arg

#ifdef TBLIS_DEBUG
inline void tblis_abort_with_message(const char* cond, const char* fmt, ...)
{
    if (strlen(fmt) == 0)
    {
        fprintf(stderr, cond);
    }
    else
    {
        va_list args;
        va_start(args, fmt);
        vfprintf(stderr, fmt, args);
        va_end(args);
    }
    fprintf(stderr, "\n");
    abort();
}

#define TBLIS_ASSERT(x,...) \
if (x) {} \
else \
{ \
    tblis_abort_with_message(TBLIS_STRINGIZE(x), "" __VA_ARGS__) ; \
}
#else
#define TBLIS_ASSERT(...)
#endif

template<typename T> std::ostream& operator<<(std::ostream& os, const std::vector<T>& v)
{
    os << "[";
    if (!v.empty()) os << v[0];
    for (int i = 1;i < v.size();i++) os << ", " << v[i];
    os << "]";
    return os;
}

template<typename T, size_t N> std::ostream& operator<<(std::ostream& os, const std::array<T,N>& v)
{
    os << "[";
    if (N) os << v[0];
    for (int i = 1;i < N;i++) os << ", " << v[i];
    os << "]";
    return os;
}

template<typename T, typename U>
std::ostream& operator<<(std::ostream& os, const std::pair<T,U>& p)
{
    os << "{" << p.first << ", " << p.second << "}";
    return os;
}

template<typename T> bool operator!(const std::vector<T>& x)
{
    return x.empty();
}

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

inline long envtol(const std::string& env, long fallback=0)
{
    char* str = getenv(env.c_str());
    if (str) return strtol(str, nullptr, 10);
    return fallback;
}

template <typename T, typename U>
constexpr stl_ext::common_type_t<T,U> remainder(T N, U B)
{
    return (B-1)-(N+B-1)%B;
}

template <typename T, typename U>
constexpr stl_ext::common_type_t<T,U> round_up(T N, U B)
{
    return N + remainder(N, B);
}

template <typename T, typename U>
constexpr stl_ext::common_type_t<T,U> ceil_div(T N, U D)
{
    return (N >= 0 ? (N+D-1)/D : (N-D+1)/D);
}

template <typename T, typename U>
constexpr stl_ext::common_type_t<T,U> floor_div(T N, U D)
{
    return N/D;
}

template <typename T, typename U>
U* convert_and_align(T* x)
{
    intptr_t off = ((intptr_t)x)%alignof(U);
    return (U*)((char*)x + (off == 0 ? 0 : alignof(U)-off));
}

template <typename T, typename U>
constexpr size_t size_as_type(size_t n)
{
    return ceil_div(n*sizeof(T) + alignof(T), sizeof(U));
}

namespace matrix_constants
{
    enum {MAT_A, MAT_B, MAT_C};
    enum {DIM_M, DIM_N, DIM_K};
    enum {NONE, JC_NT, IC_NT, JR_NT, IR_NT};
}

template <typename... Args> struct has_member_helper;
template <typename... Args>
using has_member = stl_ext::conditional_t<false,
    has_member_helper<Args...>,void>;

namespace util
{

inline double tic()
{
    #ifdef __MACH__
    static double conv = -1.0;
    if (conv < 0)
    {
        mach_timebase_info_data_t timebase;
        mach_timebase_info(&timebase);
        conv = (double)timebase.numer / (double)timebase.denom;
    }
    uint64_t nsec = mach_absolute_time();
    return conv*(double)nsec/1e9;
    #else
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec+(double)ts.tv_nsec/1e9;
    #endif
}

}
}

#endif
