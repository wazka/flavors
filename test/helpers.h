#include <algorithm>
#include <random>

#include "keys.h"

using namespace Flavors;

bool CheckSort(Keys& keys)
{
    auto h_keys = keys.ReshapeKeys(Configuration::Default32).ToHost();
    return std::is_sorted(h_keys.begin()->begin(), h_keys.begin()->end());
}

bool CheckAgainstConfig(Keys& keys, const Configuration& config)
{
    auto hostKeys = keys.ToHost();

    for (int level = 0; level < keys.Depth(); ++level)
    {
        unsigned mask = config.Mask(level);

        if (std::any_of(
            hostKeys[level].begin(),
            hostKeys[level].end(),
            [&mask](unsigned value) { return value > mask; })
            )
            return false;

        if (std::all_of(
            hostKeys[level].begin(),
            hostKeys[level].end(),
            [](unsigned value) { return value == 0; })
            )
            return false;
    }

    return true;
}