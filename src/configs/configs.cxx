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

using instance_fn_t = const config& (*)(void);

const char* names[num_configs] =
{
#define FOREACH_CONFIG(config) config::name,
#include "configs/foreach_config.h"
};

const check_fn_t check[num_configs] =
{
#define FOREACH_CONFIG(config) config::check,
#include "configs/foreach_config.h"
};

const instance_fn_t instance[num_configs] =
{
#define FOREACH_CONFIG(config) &config::instance,
#include "configs/foreach_config.h"
};

struct default_config
{
    instance_fn_t value_fn = nullptr;
    const config* value = nullptr;

    default_config()
    {
        int priority = -1;

        for (int cfg = 0;cfg < num_configs;cfg++)
        {
            TBLIS_ASSERT(check[cfg]);
            int cur_prio = check[cfg]();
            if (cur_prio > priority)
            {
                priority = cur_prio;
                value_fn = instance[cfg];
            }

            if (get_verbose() >= 1)
            {
                printf("tblis: Configuration %s assigned priority %d.\n",
                       names[cfg], cur_prio);
            }
        }

        if (!value_fn)
            tblis_abort_with_message(
                "tblis: No usable configuration enabled, aborting!");

        value = &value_fn();

        if (!value)
            tblis_abort_with_message(
                "tblis: Could not get config instance, aborting!");

        if (get_verbose() >= 1)
        {
            printf("tblis: Using configuration %s.\n", value->name);
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

const config& get_config(const std::string& name)
{
    for (int cfg = 0;cfg < num_configs;cfg++)
    {
        if (names[cfg] == name)
        {
            if (check[cfg]() == -1)
                tblis_abort_with_message(
                    "tblis: Configuration %s cannot be used!", name.c_str());
            return instance[cfg]();
        }
    }

    tblis_abort_with_message(
        "tblis: No configuration named %s!", name.c_str());
}

}
