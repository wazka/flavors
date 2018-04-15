#pragma once

#include <vector>

#include "keys.h"
#include "masks.h"
#include "tree.h"

using namespace Flavors;

bool CheckSort(Keys& keys);

bool CheckSort(Masks& masks);

bool CheckAgainstConfig(Keys& keys, const Configuration& config);

bool CmpKeys(std::vector<std::vector<unsigned>>& hostKeys, int lhs, int rhs);

bool AllKeysInTree(Tree& tree, Keys& keys);

bool CheckKeysFindResult(CudaArray<unsigned>& result, Keys& keys);

bool CheckMasksAgainstSource(
    Masks& masks, 
    std::vector<unsigned> data,
    std::vector<unsigned> lengths);

bool CheckLengths(Masks& masks, unsigned min, unsigned max);

bool CheckAgainstConfig(Masks& masks, const Configuration& config);

bool AllMasksInTree(Tree& tree, Masks& masks);

bool CheckMasksFindResult(CudaArray<unsigned>& result, Masks& masks);

bool CheckMatchResult(CudaArray<unsigned>& result, Masks& masks);