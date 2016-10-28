#include "configs.hpp"
#include "include_configs.hpp"

namespace tblis
{

namespace
{

enum config_t
{
#define FOREACH_CONFIG(config) config##_value,
#include "foreach_config.h"
    num_configs,
    first_config = 0
};

const config* const configs[num_configs] =
{
#define FOREACH_CONFIG(config) &config::instance(),
#include "foreach_config.h"
};

struct default_config
{
    const config* value;

    default_config()
    {
        int priority = -1;

        for (int cfg = first_config;cfg < num_configs;cfg++)
        {
            TBLIS_ASSERT(configs[cfg]->check);
            int cur_prio = configs[cfg]->check();
            printf("%s: %d\n", configs[cfg]->name, cur_prio);
            if (cur_prio > priority)
            {
                priority = cur_prio;
                value = configs[cfg];
            }
        }
        printf("using %s\n", value->name);
    }
};

}

const config& get_default_config()
{
    static default_config def;
    return *def.value;
}

const config& get_config(const tblis_config* cfg)
{
    return (cfg ? *(const config*)cfg : get_default_config());
}

}
