#include "catch.hpp"
#include "keys.h"
#include "helpers.h"

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

const std::vector<int> Counts = { 1000, 2000, 3000, 4000, 5000 };
const std::vector<int> Seeds = { 1234, 5765, 8304, 2365, 4968 };
const std::vector<Configuration> Configs =
{
    Flavors::Configuration{ std::vector<unsigned>{8, 8, 8, 8} },
    Flavors::Configuration{ std::vector<unsigned>{4, 4, 4, 4, 4, 4, 4, 4} },
    Flavors::Configuration{ std::vector<unsigned>{8, 8, 4, 4, 4, 4} },
    Flavors::Configuration{ std::vector<unsigned>{16, 4, 4, 4, 4} },
    Flavors::Configuration{ std::vector<unsigned>{7, 5, 3, 2, 3, 6, 6} }
};
Configuration UniqueConfig32{std::vector<unsigned>{5, 5, 3, 7, 2, 3, 7}};

TEST_CASE("Keys stress test", "[keys][stress]")
{
    for(auto count : Counts)
        for(auto seed : Seeds)
            for(auto config : Configs)
            {
                //when
                Keys keys{ config, count };
                keys.FillRandom(seed);

                //then
                REQUIRE(keys.Config == config);
                REQUIRE(keys.Depth() == config.Depth());
                REQUIRE(keys.Sorted() == false);
                REQUIRE(keys.Count == count);
                REQUIRE(CheckAgainstConfig(keys, config));

                //when
                auto newKeys = keys.ReshapeKeys(UniqueConfig32);

                //then
                REQUIRE(count == newKeys.Count);
                REQUIRE(newKeys.Config == UniqueConfig32);
                REQUIRE(UniqueConfig32.Depth() == newKeys.Depth());
                REQUIRE(CheckAgainstConfig(newKeys, UniqueConfig32));
                REQUIRE(keys.ReshapeKeys(Configuration::Default32) == newKeys.ReshapeKeys(Configuration::Default32));

                //when
                keys.Sort();

                //then
                REQUIRE(keys.Sorted() == true);
                REQUIRE(CheckSort(keys));

                //when
                newKeys.Sort();
                
                //then
                REQUIRE(newKeys.Sorted() == true);
                REQUIRE(CheckSort(newKeys));
                REQUIRE(keys.ReshapeKeys(Configuration::Default32) == newKeys.ReshapeKeys(Configuration::Default32));

                

            }
}