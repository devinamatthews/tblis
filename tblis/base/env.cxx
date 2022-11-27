#include "env.h"

#include <tblis/base/macros.h>
#include <tblis/config.h>

#include <strings.h>
#include <cstdlib>
#include <cstdio>
#include <string>

#if TBLIS_HAVE_SYSCTL
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

#if TBLIS_HAVE_SYSCONF
#include <unistd.h>
#endif

#if TBLIS_HAVE_HWLOC_H
#include <hwloc.h>
#endif

namespace
{

struct env
{
    int verbosity = 0;
    int num_threads = 0;

    void get_verbosity()
    {
        const char* str = getenv("TBLIS_VERBOSE");
        if (!str) return;

        char* end;
        verbosity = strtol(str, &end, 0);
        if (end != str) return;

        verbosity = strcasecmp(str, "true") ||
                    strcasecmp(str, "yes") ||
                    strcasecmp(str, "on");
    }

    void get_num_threads()
    {
        const char* str = nullptr;

        str = getenv("TBLIS_NUM_THREADS");
        if (!str) str = getenv("OMP_NUM_THREADS");

        if (str)
        {
            num_threads = strtol(str, nullptr, 10);
            if (num_threads > 0 && verbosity > 0)
                printf("tblis: using environment to determine number of threads\n");
        }

        #if TBLIS_HAVE_HWLOC_H

        if (num_threads < 1)
        {
            hwloc_topology_t topo;
            hwloc_topology_init(&topo);
            hwloc_topology_load(topo);

            int depth = hwloc_get_cache_type_depth(topo, 1, HWLOC_OBJ_CACHE_DATA);
            if (depth != HWLOC_TYPE_DEPTH_UNKNOWN)
            {
                num_threads = hwloc_get_nbobjs_by_depth(topo, depth);
                if (num_threads > 0 && verbosity > 0)
                    printf("tblis: using hwloc to determine number of threads\n");
            }

            hwloc_topology_destroy(topo);
        }

        #endif //TBLIS_HAVE_HWLOC_H

        #if TBLIS_HAVE_LSCPU

        if (num_threads < 1)
        {
            FILE *fd = popen("lscpu --parse=core | grep '^[0-9]' | sort -rn | head -n 1", "r");

            std::string s;
            int c;
            while ((c = fgetc(fd)) != EOF) s.push_back(c+1);

            pclose(fd);

            num_threads = strtol(s.c_str(), nullptr, 10);
            if (num_threads > 0 && verbosity > 0)
                printf("tblis: using lscpu to determine number of threads\n");
        }

        #endif //TBLIS_HAVE_LSCPU

        #if TBLIS_HAVE_SYSCTLBYNAME

        if (num_threads < 1)
        {
            size_t len = sizeof(num_threads);
            sysctlbyname("hw.physicalcpu", &num_threads, &len, nullptr, 0);
            if (num_threads > 0 && verbosity > 0)
                printf("tblis: using sysctlbyname(\"hw.physicalcpu\") to determine the number of threads\n");
        }

        #endif //TBLIS_HAVE_SYSCTLBYNAME

        #if TBLIS_HAVE_SYSCONF && TBLIS_HAVE__SC_NPROCESSORS_ONLN

        if (num_threads < 1)
        {
            num_threads = sysconf(_SC_NPROCESSORS_ONLN);
            if (num_threads > 0 && verbosity > 0)
                printf("tblis: using _SC_NPROCESSORS_ONLN to determine the number of threads\n");
        }

        #endif //TBLIS_HAVE_SYSCONF && TBLIS_HAVE__SC_NPROCESSORS_CONF

        #if TBLIS_HAVE_SYSCONF && TBLIS_HAVE__SC_NPROCESSORS_CONF

        if (num_threads < 1)
        {
            num_threads = sysconf(_SC_NPROCESSORS_CONF);
            if (num_threads > 0 && verbosity > 0)
                printf("tblis: using _SC_NPROCESSORS_CONF to determine the number of threads\n");
        }

        #endif //TBLIS_HAVE_SYSCONF && TBLIS_HAVE__SC_NPROCESSORS_CONF

        if (num_threads < 1)
        {
            num_threads = 1;
            if (verbosity > 0)
                printf("tblis: unable to determine the number of threads\n");
        }

        if (verbosity > 0)
            printf("tblis: initial number of threads: %d\n", num_threads);
    }

    env()
    {
        get_verbosity();
        get_num_threads();
    }

    static env& instance()
    {
        static env inst;
        return inst;
    }
};

}

TBLIS_EXPORT int tblis_get_verbosity()
{
    return env::instance().verbosity;
}

TBLIS_EXPORT void tblis_set_verbosity(int level)
{
    env::instance().verbosity = level;
}

TBLIS_EXPORT int tblis_get_num_threads()
{
    return env::instance().num_threads;
}

TBLIS_EXPORT void tblis_set_num_threads(int num_threads)
{
    env::instance().num_threads = num_threads;
}

