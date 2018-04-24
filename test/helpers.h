#pragma once

#include <vector>

#include "keys.h"
#include "masks.h"
#include "tree.h"

#include "catch.hpp"

using namespace Flavors;
using namespace std;

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

template<class TreeType>
void TreeFromKeysTest(std::vector<unsigned> trueLevelsSizes)
{
    //given
    int count = 4;
    auto data = vector<unsigned>{
        1, 99, 50, 50,
        5,  6,  9,  8,
        9, 10, 11, 12};

    Configuration config{vector<unsigned>{8, 8, 4}};

    Keys keys{config, count};
    keys.FillFromVector(data);

    //when
    TreeType tree{keys};

    //then
    REQUIRE(tree.Count == count);
    REQUIRE(tree.Depth() == config.Depth());
    REQUIRE(tree.Config == config);
    REQUIRE(tree.h_LevelsSizes == trueLevelsSizes);

    //given
    CudaArray<unsigned> result{count};

    //when
    tree.Find(keys, result.Get());

    //then
    //0 in result would mean not found
    //TODO: Tree constructor sorts keys under the hood
    auto h_result = result.ToHost();
    REQUIRE(h_result[0] == 1);
    REQUIRE(h_result[1] == 4);
    REQUIRE(h_result[2] == 3);
    REQUIRE(h_result[3] == 2);

    //given
    Keys otherKeys{config, count};
    otherKeys.FillFromVector(data);
    result.Clear();

    //when
    tree.Find(otherKeys, result.Get());

    //then
    h_result = result.ToHost();
    REQUIRE(h_result[0] == 1);
    REQUIRE(h_result[1] == 2);
    REQUIRE(h_result[2] == 3);
    REQUIRE(h_result[3] == 4);
}