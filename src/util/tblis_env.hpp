#ifndef _TBLIS_ENV_HPP_
#define _TBLIS_ENV_HPP_

namespace tblis
{

inline long envtol(const std::string& env, long fallback=0)
{
    char* str = getenv(env.c_str());
    if (str) return strtol(str, nullptr, 10);
    return fallback;
}

}

#endif
