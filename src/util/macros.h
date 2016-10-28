#ifndef _TBLIS_MACROS_H_
#define _TBLIS_MACROS_H_

#define TBLIS_WITH_TYPE_AS(type, T, ...) \
if ((type) == TYPE_FLOAT) \
{ \
    typedef float T; \
    __VA_ARGS__ \
} \
else if ((type) == TYPE_DOUBLE) \
{ \
    typedef double T; \
    __VA_ARGS__ \
} \
else if ((type) == TYPE_DOUBLE) \
{ \
    typedef scomplex T; \
    __VA_ARGS__ \
} \
else if ((type) == TYPE_DOUBLE) \
{ \
    typedef dcomplex T; \
    __VA_ARGS__ \
} \
else \
{ \
    TBLIS_ASSERT(0, "Unknown type"); \
}

#define TBLIS_SPECIAL_CASE(condition, ...) \
if (condition) { __VA_ARGS__ } \
else           { __VA_ARGS__ }

#endif
