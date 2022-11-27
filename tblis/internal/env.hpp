#ifndef TBLIS_INTERNAL_ENV_HPP
#define TBLIS_INTERNAL_ENV_HPP

#include <tblis/base/env.h>

#include <cstdlib>

namespace tblis
{

inline long envtol(const char* name, long def)
{
    auto env = getenv(name);
    return env ? strtol(env, nullptr, 0) : def;
}

}

#endif //TBLIS_INTERNAL_ENV_HPP
