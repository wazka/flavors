#include "catch.hpp"
#include "tree.h"

#include <vector>

using namespace std;
using namespace Flavors;

TEST_CASE("Tree from keys test", "[tree]")
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
    Tree tree{keys};

    //then
    REQUIRE(tree.Count == count);
    REQUIRE(tree.Depth() == config.Depth());
    REQUIRE(tree.Config == config);
    REQUIRE(tree.h_LevelsSizes[0] == 1);
    REQUIRE(tree.h_LevelsSizes[1] == 3);
    REQUIRE(tree.h_LevelsSizes[2] == 4);

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

    //when
    tree.Find(otherKeys, result.Get());

    //then
    h_result = result.ToHost();
    REQUIRE(h_result[0] == 1);
    REQUIRE(h_result[1] == 2);
    REQUIRE(h_result[2] == 3);
    REQUIRE(h_result[3] == 4);
}

TEST_CASE("Tree from masks test", "[tree]")
{
    //given
    int count = 4;
    auto data = vector<unsigned>{
        1, 99, 50, 50,
        5,  6,  9,  8,
        9, 10, 11, 12};
    auto lengths = vector<unsigned>{7, 10, 19, 14};

    Configuration config{vector<unsigned>{8, 8, 4}};

    //when
    Masks masks{config, count};
    masks.FillFromVector(data, lengths);

    //when
    Tree tree{masks};

    //then
    REQUIRE(tree.Count == count);
    REQUIRE(tree.Depth() == config.Depth());
    REQUIRE(tree.Config == config);
    REQUIRE(tree.h_LevelsSizes[0] == 1);
    REQUIRE(tree.h_LevelsSizes[1] == 2);
    REQUIRE(tree.h_LevelsSizes[2] == 1);

    //given
    CudaArray<unsigned> result{count};

    //when
    tree.Find(masks, result.Get());

    //then
    //0 in result would mean not found
    //TODO: Tree constructor sorts keys under the hood
    auto h_result = result.ToHost();
    REQUIRE(h_result[0] == 1);
    REQUIRE(h_result[1] == 4);
    REQUIRE(h_result[2] == 3);
    REQUIRE(h_result[3] == 2);

    //given
    Masks otherMasks{config, count};
    otherMasks.FillFromVector(data, lengths);

    //when
    tree.Find(otherMasks, result.Get());

    //then
    h_result = result.ToHost();
    REQUIRE(h_result[0] == 1);
    REQUIRE(h_result[1] == 2);
    REQUIRE(h_result[2] == 3);
    REQUIRE(h_result[3] == 4);
}