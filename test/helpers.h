#pragma once

#include <vector>

#include "keys.h"
#include "masks.h"
#include "tree.h"

using namespace Flavors;

bool CheckSort(Keys& keys);

bool CheckAgainstConfig(Keys& keys, const Configuration& config);

bool CmpKeys(std::vector<std::vector<unsigned>>& hostKeys, int lhs, int rhs);

bool AllKeysInTree(Tree& tree, Keys& keys);

bool CheckKeysFindResult(CudaArray<unsigned>& result, Keys& keys);

bool CheckMasksAgainstSource(
    Masks& masks, 
    std::vector<unsigned> data,
    std::vector<unsigned> lengths);