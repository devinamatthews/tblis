#include "env.hpp"

#include <cstdlib>
#include <strings.h>

namespace tblis
{

namespace
{

class verbosity
{
  private:
    int level_ = 0;

    verbosity()
    {
        const char* str = getenv("TBLIS_VERBOSE");
        if (!str)
            return;

        char* end;
        level_ = strtol(str, &end, 0);
        if (end != str)
            return;

        if (strcasecmp(str, "true")
            || strcasecmp(str, "yes")
            || strcasecmp(str, "on"))
        {
            level_ = 1;
        }
        else
        {
            level_ = 0;
        }
    }

  public:
    static int& level()
    {
        static verbosity v;
        return v.level_;
    }
};

} // namespace

int get_verbose()
{
    return verbosity::level();
}

void set_verbose(int level)
{
    verbosity::level() = level;
}

} // namespace tblis
