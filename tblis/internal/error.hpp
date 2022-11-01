#ifndef TBLIS_BASE_ERROR_HPP
#define TBLIS_BASE_ERROR_HPP

#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <type_traits>

#include <tblis/base/macros.h>

inline void __attribute__((format(printf, 1, 2),noreturn))
tblis_abort_with_message(const char* fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}

inline void tblis_check_assert(const char* cond_str, bool cond)
{
    if (TBLIS_UNLIKELY(cond))
        tblis_abort_with_message("%s", cond_str);
}

template <typename... Args>
inline void __attribute__((format(printf, 3, 0)))
tblis_check_assert(const char*, bool cond, const char* fmt, Args&&... args)
{
    if (TBLIS_UNLIKELY(cond))
        tblis_abort_with_message(fmt, std::forward<Args>(args)...);
}

#ifdef TBLIS_DEBUG
#define TBLIS_ASSERT(...) \
    tblis_check_assert(TBLIS_STRINGIZE(TBLIS_FIRST_ARG(__VA_ARGS__,0)), __VA_ARGS__)
#else
#define TBLIS_ASSERT(...) ((void)(__VA_ARGS__),(void)0)
#endif

#endif //TBLIS_BASE_ERROR_HPP
