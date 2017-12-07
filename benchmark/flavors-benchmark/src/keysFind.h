#pragma once
#include <configuration.h>
#include <string>

#include "benchmark.h"
#include "tree.h"

namespace FlavorsBenchmarks
{
	class KeysFindBenchmark : public RandomBenchmark
	{
	public:
		KeysFindBenchmark(
				int count,
				int seed,
				const Flavors::Configuration& config,
				std::vector<int> countsToFind,
				const std::string& resultPath,
				const std::string& resultName = "keysFindResult") :
			RandomBenchmark(count, seed, resultPath, resultName),
			config(config),
			countsToFind(countsToFind)
		{}

		void Run() override;

		virtual ~KeysFindBenchmark() = default;
	protected:

		Flavors::Configuration config;
		std::vector<int> countsToFind;
	};

}
