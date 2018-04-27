#include "helpers.h"

#include <algorithm>
#include <random>

using namespace Flavors;

bool CheckSort(Keys& keys)
{
    auto h_keys = keys.ReshapeKeys(Configuration::Default32).ToHost();
    return std::is_sorted(h_keys.begin()->begin(), h_keys.begin()->end());
}

bool CheckLengths(Masks& masks, unsigned min, unsigned max)
{
    auto hostLengths = masks.Lengths.ToHost();
    return std::all_of(
        hostLengths.begin(),
        hostLengths.end(),
        [&min, &max](unsigned length) { return length >= min && length <= max; });
}

bool CheckAgainstConfig(Masks& masks, const Configuration& config)
{
    Keys& keys = masks;
    return CheckAgainstConfig(keys, config) && CheckLengths(masks, 1, config.Length);
}

bool CheckSort(Masks& masks)
{
    auto h_masks = masks.ReshapeMasks(Configuration::Default32).ToHost();
    auto h_lenghts = masks.Lengths.ToHost();

    for (int mask = 1; mask < masks.Count; ++mask)
    {
        if (h_masks[0][mask - 1] > h_masks[0][mask])
            return false;
        if (h_masks[0][mask - 1] == h_masks[0][mask])
            if (h_lenghts[mask - 1] > h_lenghts[mask])
                return false;
    }

    return true;
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

bool AllKeysInCompressedTree(Tree& tree, Keys& keys)
{
    auto hostChildren = tree.Children.ToHost();
    auto hostKeys = keys.ToHost();



    for (int key = 0; key < keys.Count; ++key)
    {
        int currentNode = 1;
        for (int level = 0; level < tree.Depth(); ++level)
        {
            currentNode = hostChildren[level][(currentNode - 1) * tree.ChildrenCountsHost[level] + hostKeys[level][key]];
            if (currentNode == 0)
            {
                std::cout << "Wrong path on level " << level << " for key " << key << " \n\n";
                return false;
            }
        }

        if (!CmpKeys(hostKeys, key, currentNode - 1))
        {
            // std::cout << keys << std::endl;
            std::cout << "Wrong node " << key << " : " << currentNode - 1 << " \n\n";
            return false;
        }
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

bool CheckKeysFindResultInCompressedTree(CudaArray<unsigned>& result, Keys& keys)
{
    auto h_result = result.ToHost();
    auto h_keys = keys.ToHost();

    for(int key = 0; key < keys.Count; ++key)
    {
        if (key != h_result[key] - 1)
        {
            if (!CmpKeys(h_keys, key, h_result[key] - 1))
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

bool AllMasksInTree(Tree& tree, Masks& masks)
{
    auto h_children = tree.Children.ToHost();
    auto h_masks = masks.ToHost();
    auto h_permutation = masks.Permutation.ToHost();
    auto h_lengths = masks.Lengths.ToHost();

    auto h_contStarts = tree.containers.Starts.ToHost();
    auto h_contLengths = tree.containers.Lengths.ToHost();
    auto h_contItems = tree.containers.Items.ToHost();

    auto h_masksParts = tree.masksParts.ToHost();
    auto h_treeLengths = tree.lengths.ToHost();

    for (int mask = 0; mask < masks.Count; ++mask)
    {
        int currentNode = 1;
        int depth = tree.Config[0];
        int level = 0;

        while(h_lengths[mask] > depth)
        {
            currentNode = h_children[level][(currentNode - 1) * tree.ChildrenCountsHost[level] + h_masks[level][mask]];
            
            ++level;
            depth += tree.Config[level];

            if (currentNode == 0)
                return false;
        }

        int listItem = 0;
        while(listItem < h_contLengths[level][currentNode - 1])
        {
            auto itemValue = h_contItems[h_contStarts[level][currentNode - 1] + listItem];
            if (h_lengths[mask] == h_treeLengths[itemValue] && h_masksParts[itemValue] == h_masks[level][mask])
                break;

            ++listItem;
        }

        if (listItem == h_contLengths[level][currentNode - 1])
            return false;

        auto itemValue = h_contItems[h_contStarts[level][currentNode - 1] + listItem];
        if (!CmpKeys(h_masks, mask, itemValue) || h_lengths[mask] != h_lengths[itemValue])
            return false;
    }

    return true;
}

bool CheckMasksFindResult(CudaArray<unsigned>& result, Masks& masks)
{
    auto h_result = result.ToHost();
    auto h_masks = masks.ToHost();
    auto h_permutation = masks.Permutation.ToHost();
    auto h_lengths = masks.Lengths.ToHost();
    
    for (int mask = 0; mask < masks.Count; ++mask)
    {
        if (h_permutation[mask] != h_result[mask] - 1)
        {
            auto retrivedMask = std::find(h_permutation.begin(), h_permutation.end(), h_result[mask] - 1) - h_permutation.begin();
            if (!CmpKeys(h_masks, mask, retrivedMask) || h_lengths[mask] != h_lengths[retrivedMask])
                return false;
        }
    }

    return true;
}

bool CheckMatchResult(CudaArray<unsigned>& result, Masks& masks)
{
    //This works, since in tests, we match masks with themselves and empty values are filled with 0.
    //That is why, we can compare using CmpKeys.

    auto h_result = result.ToHost();
    auto h_masks = masks.ToHost();
    auto h_permutation = masks.Permutation.ToHost();
    auto h_lengths = masks.Lengths.ToHost();

    for (int mask = 0; mask < masks.Count; ++mask)
    {
        if (h_permutation[mask] != h_result[mask] - 1)
        {
            auto retrivedMask = std::find(h_permutation.begin(), h_permutation.end(), h_result[mask] - 1) - h_permutation.begin();
            if (!CmpKeys(h_masks, mask, retrivedMask) || h_lengths[mask] > h_lengths[retrivedMask])
                return false;
        }
    }

    return true;
}