#pragma once
#include <vector>
#include "configuration.h"

namespace FlavorsTests
{
	namespace TestData
	{
		const std::vector<int> SmallCounts = { 1000, 2000, 3000, 4000, 5000 };
		const std::vector<int> BigCounts = { 10000, 20000, 30000, 40000, 50000, 100000, 200000, 300000, 400000, 500000, 1000000 };
		const std::vector<int> Seeds = { 1234, 5765, 8304, 2365, 4968 };

		const std::vector<unsigned> Depths = { 32, 48, 64, 80, 96};
		const std::vector<unsigned> FirstLevelStrides = {16, 8, 4};
		const std::vector<unsigned> LevelStrides = {8, 4};

		const std::vector<Flavors::Configuration> Configs =
		{
			Flavors::Configuration{ std::vector<unsigned>{8, 8, 8, 8} },
			Flavors::Configuration{ std::vector<unsigned>{4, 4, 4, 4, 4, 4, 4, 4} },
			Flavors::Configuration{ std::vector<unsigned>{8, 8, 4, 4, 4, 4} },
			Flavors::Configuration{ std::vector<unsigned>{16, 4, 4, 4, 4} },
			Flavors::Configuration{ std::vector<unsigned>{7, 5, 3, 2, 3, 6, 6} }
		};

		const std::string BenchmarkResultFile = "";
	}
}
