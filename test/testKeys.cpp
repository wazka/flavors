#include "catch.hpp"
#include "keys.h"

#include <vector>

using namespace std;
using namespace Flavors;

TEST_CASE("Keys test", "[keys]")
{
    //given
    //Data is 4 vectors, 3 parts each
    int count = 4;
    auto data = vector<unsigned>{
        1, 99, 50, 50,
        5,  6,  9,  8,
        9, 10, 11, 12};

    Configuration config{vector<unsigned>{8, 8, 4}};

    //when
    Keys keys{config, count};
    keys.FillFromVector(data);

    //then
    REQUIRE(keys.Config == config);
    REQUIRE(keys.Depth() == config.Depth());
    REQUIRE(keys.Sorted() == false);
    REQUIRE(keys.Count == count);
    //WARN(keys);

    auto h_keys = keys.ToHost();
    for(int level = 0; level < keys.Depth(); ++level)
        for(int key = 0; key < keys.Count; ++key)
            REQUIRE(h_keys[level][key] == data[level * count + key]);

    //when
    keys.Sort();

    //then
    REQUIRE(keys.Sorted() == true);
    auto h_permutation = keys.Permutation.ToHost();
    REQUIRE(h_permutation[0] == 0);
    REQUIRE(h_permutation[1] == 3);
    REQUIRE(h_permutation[2] == 2);
    REQUIRE(h_permutation[3] == 1);

    //given
    Configuration otherConfig{vector<unsigned>{9, 6, 2, 3}};

    //when
    auto otherKeys = keys.ReshapeKeys(otherConfig);

    //then
    REQUIRE(otherKeys.Count == keys.Count);
    REQUIRE(otherKeys.Sorted() == true);
    auto h_otherPermutation = otherKeys.Permutation.ToHost();
    REQUIRE_THAT(h_permutation, Catch::Equals(h_permutation));

    //when
    Keys yetOtherKeys{config, count};
    yetOtherKeys.FillFromVector(data);
    yetOtherKeys = yetOtherKeys.ReshapeKeys(otherConfig);
    yetOtherKeys.Sort();

    //then
    REQUIRE(otherKeys == yetOtherKeys);
}