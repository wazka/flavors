#include "catch.hpp"
#include "keys.h"
#include "masks.h"
#include "tree.h"
#include "helpers.h"

using namespace Flavors;
using namespace std;

const vector<int> Counts = { 10000, 20000, 30000, 40000, 50000 };
const vector<int> Seeds = { 1234, 5765, 8304, 2365, 4968 };
const vector<Configuration> Configs =
{
    Flavors::Configuration{ vector<unsigned>{8, 8, 8, 8} },
    Flavors::Configuration{ vector<unsigned>{4, 4, 4, 4, 4, 4, 4, 4} },
    Flavors::Configuration{ vector<unsigned>{8, 8, 4, 4, 4, 4} },
    Flavors::Configuration{ vector<unsigned>{16, 4, 4, 4, 4} },
    Flavors::Configuration{ vector<unsigned>{7, 5, 3, 2, 3, 6, 6} }
};
Configuration UniqueConfig32{vector<unsigned>{5, 5, 3, 7, 2, 3, 7}};

TEST_CASE("Keys load test", "[load][keys]")
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

                //when
                Tree tree{ keys };

                //then
                REQUIRE(tree.Count == count);
                REQUIRE(tree.Depth() == config.Depth());
                REQUIRE(tree.Config == config);
                REQUIRE(AllKeysInTree(tree, keys));

                //when
                CudaArray<unsigned> result{ keys.Count };
                tree.FindKeys(keys, result.Get());

                //then
                REQUIRE(CheckKeysFindResult(result, keys));

                //given
                Keys randomKeys{ config, count };
                randomKeys.FillRandom(seed + 1);

                //when
                result.Clear();
                REQUIRE_NOTHROW(tree.FindKeys(randomKeys, result.Get()));
            }
}

TEST_CASE("Only long masks load test", "[load][masks][long]")
{
    for(auto count : Counts)
        for(auto seed : Seeds)
            for(auto config : Configs)
            {
                //given
                unsigned minLen = 8;

                //when
                Masks masks{ config, count };
                masks.FillRandom(seed);

                //then
                REQUIRE(masks.Config == config);
                REQUIRE(masks.Depth() == config.Depth());
                REQUIRE(masks.Sorted() == false);
                REQUIRE(masks.Count == count);
                REQUIRE(CheckAgainstConfig(masks, config));

                //when
                auto newMasks = masks.ReshapeMasks(UniqueConfig32);

                //then
                REQUIRE(count == newMasks.Count);
                REQUIRE(newMasks.Config == UniqueConfig32);
                REQUIRE(UniqueConfig32.Depth() == newMasks.Depth());
                REQUIRE(CheckAgainstConfig(newMasks, UniqueConfig32));
                REQUIRE(masks.ReshapeMasks(Configuration::Default32) == newMasks.ReshapeMasks(Configuration::Default32));

                //when
                masks.Sort();

                //then
                REQUIRE(masks.Sorted() == true);
                REQUIRE(CheckSort(masks));

                //when
                newMasks.Sort();
                
                //then
                REQUIRE(newMasks.Sorted() == true);
                REQUIRE(CheckSort(newMasks));
                REQUIRE(masks.ReshapeMasks(Configuration::Default32) == newMasks.ReshapeMasks(Configuration::Default32));

                //when
                Tree tree{ masks };

                //then
                REQUIRE(tree.Count == count);
                REQUIRE(tree.Depth() == config.Depth());
                REQUIRE(tree.Config == config);
                REQUIRE(AllMasksInTree(tree, masks));

                //when
                CudaArray<unsigned> result{ masks.Count };
                tree.FindMasks(masks, result.Get());

                //then
                REQUIRE(CheckMasksFindResult(result, masks));

                //given
                result.Clear();

                //when
                tree.Match(masks, result.Get());

                //then
                REQUIRE(CheckMatchResult(result, masks));

                //given
                Masks randomMasks{ config, count };
                randomMasks.FillRandom(seed + 1);

                //when
                result.Clear();
                REQUIRE_NOTHROW(tree.FindMasks(randomMasks, result.Get()));

                //when
                result.Clear();
                REQUIRE_NOTHROW(tree.Match(randomMasks, result.Get()));
            }
}