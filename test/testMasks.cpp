#include "catch.hpp"
#include "masks.h"

#include "helpers.h"

#include <vector>

using namespace std;
using namespace Flavors;

TEST_CASE("Masks test", "[masks]")
{
    //given
    //Data is 4 vectors, 3 parts each
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

    //then
    REQUIRE(masks.Config == config);
    REQUIRE(masks.Depth() == config.Depth());
    REQUIRE(masks.Sorted() == false);
    REQUIRE(masks.Count == count);
    REQUIRE(CheckMasksAgainstSource(masks, data, lengths));
    // WARN(masks);

    //when
    masks.Sort();

    //then
    REQUIRE(masks.Sorted() == true);
    auto h_permutation = masks.Permutation.ToHost();
    REQUIRE(h_permutation[0] == 0);
    REQUIRE(h_permutation[1] == 3);
    REQUIRE(h_permutation[2] == 2);
    REQUIRE(h_permutation[3] == 1);

    //given
    Configuration otherConfig{vector<unsigned>{9, 6, 2, 3}};

    //when
    auto otherMasks = masks.ReshapeMasks(otherConfig);

    //then
    REQUIRE(otherMasks.Count == masks.Count);
    REQUIRE(otherMasks.Sorted() == true);
    auto h_otherPermutation = otherMasks.Permutation.ToHost();
    REQUIRE_THAT(h_permutation, Catch::Equals(h_permutation));

    //when
    Masks yetOtherMasks{config, count};
    yetOtherMasks.FillFromVector(data, lengths);
    yetOtherMasks = yetOtherMasks.ReshapeMasks(otherConfig);
    yetOtherMasks.Sort();

    //then
    REQUIRE(otherMasks == yetOtherMasks);
}