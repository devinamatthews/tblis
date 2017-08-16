#ifndef _TBLIS_MACROS_H_
#define _TBLIS_MACROS_H_

#define TBLIS_PASTE_(x,y) x##y
#define TBLIS_PASTE(x,y) TBLIS_PASTE_(x,y)

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
else if ((type) == TYPE_SCOMPLEX) \
{ \
    typedef scomplex T; \
    __VA_ARGS__ \
} \
else if ((type) == TYPE_DCOMPLEX) \
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
