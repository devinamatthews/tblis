#include "tblis_mutex.hpp"

#include <cstdarg>
#include <cstdio>
#include <mutex>

namespace tblis
{

mutex print_lock;

void printf_locked(const char* fmt, ...)
{
    std::lock_guard<mutex> guard(print_lock);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

}
