#ifndef MARRAY_CONFIG_HPP
#define MARRAY_CONFIG_HPP

#include <cassert>

#ifndef MARRAY_DEBUG
#ifndef NDEBUG
#define MARRAY_DEBUG 1
#else
#define MARRAY_DEBUG 0
#endif
#endif

#ifndef MARRAY_ASSERT
#if MARRAY_DEBUG
#define MARRAY_ASSERT(e) assert(e)
#else
#define MARRAY_ASSERT(e) ((void)0)
#endif
#endif

#define MARRAY_LIKELY(x) __builtin_expect((x), 1)
#define MARRAY_UNLIKELY(x) __builtin_expect((x), 0)

// Define an inner namespace depending on debug settings...
#if MARRAY_DEBUG
#define MARRAY_INNER_NAMESPACE debug
#else
#define MARRAY_INNER_NAMESPACE release
#endif

#define MARRAY_BEGIN_NAMESPACE       \
    namespace MArray                 \
    {                                \
    namespace MARRAY_INNER_NAMESPACE \
    {
#define MARRAY_END_NAMESPACE \
    }                        \
    }

MARRAY_BEGIN_NAMESPACE
// ...declare the inner namespace...
MARRAY_END_NAMESPACE

namespace MArray
{
// ...then import it into the top-level MArray namespace
using namespace MARRAY_INNER_NAMESPACE;
} // namespace MArray

#endif // MARRAY_CONFIG_HPP
