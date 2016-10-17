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

config configs[num_configs] =
{
#define FOREACH_CONFIG(config) config(),
#include "foreach_config.h"
};

struct default_config
{
    const config* value;

    default_config()
    {
        int priority = -1;

        for (config_t config = first_config;config < num_configs;config++)
        {
            int cur_prio = configs[config].check();
            if (cur_prio > priority)
            {
                priority = cur_prio;
                value = &configs[config];
            }
        }
    }
};

}

const config* get_default_config()
{
    static default_config def;
    return def.value;
}

}
