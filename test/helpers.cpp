#include "helpers.h"

#include <algorithm>
#include <random>

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

bool CmpKeys(std::vector<std::vector<unsigned>>& hostKeys, int lhs, int rhs)
{
    return std::all_of(
        hostKeys.begin(),
        hostKeys.end(),
        [&lhs, &rhs](std::vector<unsigned>& level){ return level[lhs] == level[rhs]; });
}

bool AllKeysInTree(Tree& tree, Keys& keys)
{
    auto hostChildren = tree.Children.ToHost();
    auto hostKeys = keys.ToHost();
    auto hostPermutation = keys.Permutation.ToHost();

    for (int key = 0; key < keys.Count; ++key)
    {
        int currentNode = 1;
        for (int level = 0; level < tree.Depth(); ++level)
        {
            currentNode = hostChildren[level][(currentNode - 1) * tree.ChildrenCountsHost[level] + hostKeys[level][key]];
            if (currentNode == 0)
                return false;
        }

        auto retrivedKey = std::find(hostPermutation.begin(), hostPermutation.end(), currentNode - 1) - hostPermutation.begin();
        if (!CmpKeys(hostKeys, key, retrivedKey))
            return false;
    }

    return true;
}

bool CheckKeysFindResult(CudaArray<unsigned>& result, Keys& keys)
{
    auto h_result = result.ToHost();
    auto h_keys = keys.ToHost();
    auto h_permutation = keys.Permutation.ToHost();

    for(int key = 0; key < keys.Count; ++key)
    {
        if (h_permutation[key] != h_result[key] - 1)
        {
            auto retrivedKey = std::find(h_permutation.begin(), h_permutation.end(), h_result[key] - 1) - h_permutation.begin();
            if (!CmpKeys(h_keys, key, retrivedKey))
                return false;
        }
    }

    return true;
}

bool CheckMasksAgainstSource(
    Masks& masks, 
    std::vector<unsigned> data,
    std::vector<unsigned> lengths)
{
    auto h_masks = masks.ToHost();
    auto h_lengths = masks.Lengths.ToHost();

    for(int mask = 0; mask < masks.Count; ++mask)
	{
        int globalBit = 0;
        for(int level = 0; level < masks.Depth(); ++level)
        {
            for (int bit = masks.Config[level] - 1; bit >= 0; --bit)
            {
                if(globalBit < h_lengths[mask])
                {
                    if(((h_masks[level][mask] >> bit) & 1u) != ((data[level * masks.Count + mask] >> bit) & 1u))
                        return false;
                }
                 else
                     if((h_masks[level][mask] >> bit) & 1u)
                        return false;

                 ++globalBit;
            }
        }

        if(h_lengths[mask] != lengths[mask])
            return false;
    }

    return true;
}