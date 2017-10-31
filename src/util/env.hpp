#ifndef _TBLIS_ENV_HPP_
#define _TBLIS_ENV_HPP_

#include <string>
#include <cstdlib>

namespace tblis
{

inline long envtol(const std::string& env, long fallback=0)
{
    char* str = getenv(env.c_str());
    if (str) return strtol(str, nullptr, 10);
    return fallback;
}

int get_verbose();

void set_verbose(int);

}

#endif
