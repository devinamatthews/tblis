#include "configs.h"

#include "configs/configs.hpp"

namespace tblis
{

extern "C"
{

const tblis_config* tblis_get_config(const char* name)
{
    return reinterpret_cast<const tblis_config*>(&get_config(name));
}

}
}
