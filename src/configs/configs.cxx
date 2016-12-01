#include "configs.hpp"
#include "configs/include_configs.hpp"

namespace tblis
{

namespace
{

enum config_t
{
#define FOREACH_CONFIG(config) config##_value,
#include "configs/foreach_config.h"
    num_configs
};

const config* const configs[num_configs] =
{
#define FOREACH_CONFIG(config) &config::instance(),
#include "configs/foreach_config.h"
};

struct default_config
{
    const config* value;

    default_config()
    {
        int priority = -1;

        for (int cfg = 0;cfg < num_configs;cfg++)
        {
            TBLIS_ASSERT(configs[cfg]->check);
            int cur_prio = configs[cfg]->check();
            if (cur_prio > priority)
            {
                priority = cur_prio;
                value = configs[cfg];
            }
        }
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
    return (cfg ? *reinterpret_cast<const config*>(cfg) : get_default_config());
}

}
