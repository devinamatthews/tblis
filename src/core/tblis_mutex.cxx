#include "tblis_mutex.hpp"

#include <cstdarg>

namespace tblis
{

Mutex print_lock;

void printf_locked(const char* fmt, ...)
{
    std::lock_guard<Mutex> guard(print_lock);
    va_list args;
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
}

}
