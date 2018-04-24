#include "catch.hpp"
#include "configuration.h"

#include <vector>
#include <string>

using namespace std;
using namespace Flavors;

TEST_CASE("Configuration test", "[configuration]")
{
    //given
    vector<unsigned> levels {8, 4, 4, 16, 32, 10, 5};
    int length = 0;
    for(auto level : levels)
        length += level;

    //when
    Configuration config{levels};

    //then
    REQUIRE(config.Depth() == levels.size());
    REQUIRE(config.Length == length);

    for(int level = 0; level < levels.size(); ++level)
        REQUIRE(config[level] == levels[level]);

    string configStr;
    REQUIRE_NOTHROW(configStr = config.ToString());
    //WARN(configStr);

    //when
    config.PopLastLevel();

    //then
    REQUIRE(config.Depth() == levels.size() - 1);

    //when
    REQUIRE_NOTHROW(config.Get());  
}

TEST_CASE("Default and binary configuration based on key lenght", "[configuration]")
{
    //given
    unsigned length = 37;

    //when
    auto config = Configuration::Default(length);

    //then
    REQUIRE(config.Depth() == 2);
    REQUIRE(config[0] == 32);
    REQUIRE(config[1] == length - 32);
    REQUIRE(config.Length == length);

    //when
    config = Configuration::Binary(length);

    //then
    REQUIRE(config.Depth() == length);
    REQUIRE(config.Length == length);
    for(int level = 0; level < length; ++level)
        REQUIRE(config[level] == 1);

    //when
    REQUIRE_NOTHROW(config.Get());
}

TEST_CASE("Default 32 bit configuration", "[configuration]")
{
    Configuration config = Configuration::Default32;

    REQUIRE(config.Depth() == 1);
    REQUIRE(config[0] == 32);
    REQUIRE(config.Length == 32);
    REQUIRE_NOTHROW(config.Get());
}

TEST_CASE("Binary 32 bit configuration", "[configuration]")
{
    Configuration config = Configuration::Binary32;

    REQUIRE(config.Depth() == 32);
    REQUIRE(config.Length == 32);

    for(int i = 0; i < 32; ++i)
        REQUIRE(config[i] == 1);

    REQUIRE_NOTHROW(config.Get());
}