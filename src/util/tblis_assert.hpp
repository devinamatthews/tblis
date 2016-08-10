#ifndef _TBLIS_ASSERT_HPP_
#define _TBLIS_ASSERT_HPP_

#include "tblis_config.hpp"

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

#endif
