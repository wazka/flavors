#include "catch.hpp"
#include "tree.h"
#include "compressedTree.h"
#include "helpers.h"

#include <vector>

using namespace std;
using namespace Flavors;

TEST_CASE("Tree from keys test", "[tree][keys][unit]")
{
    auto data = vector<unsigned>{
        1, 99, 50, 50,
        5,  6,  9,  8,
        9, 10, 11, 12};

    TreeFromKeysTest<Tree>(
        data,
        std::vector<unsigned>{1, 3, 4},
        std::vector<unsigned>{1, 4, 3, 2}
    );
}

TEST_CASE("Compressed tree from keys test", "[compressed-tree][keys][unit]")
{
    auto data = vector<unsigned>{
        1, 99, 50, 50,
        5,  6,  9,  8,
        9, 10, 11, 12};

    TreeFromKeysTest<CompressedTree>(
        data,
        std::vector<unsigned>{1, 1, 1},
        std::vector<unsigned>{1, 2, 3, 4}
    );
}

TEST_CASE("Compressed tree from keys with replicated key test", "[compressed-tree][keys][unit]")
{
    auto data = vector<unsigned>{
        1, 99, 50, 50,
        5,  6,  9,  9,
        9, 10, 11, 11};

    TreeFromKeysTest<CompressedTree>(
        data,
        std::vector<unsigned>{1, 1, 1},
        std::vector<unsigned>{1, 2, 3, 3}
    );
}

TEST_CASE("Compressed tree from keys with split test", "[compressed-tree][keys][unit]")
{
    auto data = vector<unsigned>{
        1, 99, 50, 50,
        5,  6,  8,  9,
        9, 10, 11, 11};

    TreeFromKeysTest<CompressedTree>(
        data,
        std::vector<unsigned>{1, 1, 2},
        std::vector<unsigned>{1, 2, 3, 4}
    );
}

TEST_CASE("Compressed tree from keys with circle test", "[compressed-tree][keys][unit]")
{
    auto data = vector<unsigned>{
        1, 99, 50, 51,
        5,  6,  9,  9,
        9, 10, 11, 12};

    TreeFromKeysTest<CompressedTree>(
        data,
        std::vector<unsigned>{1, 2, 1},
        std::vector<unsigned>{1, 2, 3, 4}
    );
}

TEST_CASE("Tree from masks test", "[tree][masks][unit]")
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
    result.Clear();

    //when
    tree.Find(otherMasks, result.Get());

    //then
    h_result = result.ToHost();
    REQUIRE(h_result[0] == 1);
    REQUIRE(h_result[1] == 2);
    REQUIRE(h_result[2] == 3);
    REQUIRE(h_result[3] == 4);

    //given
    result.Clear();

    //when
    REQUIRE_NOTHROW(tree.Match(otherMasks, result.Get()));
}