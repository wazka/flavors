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
		static std::string Label;

		KeysFindBenchmark(
				int count,
				int seed,
				const Flavors::Configuration& config,
				const std::string& resultPath,
				const std::string& resultName = "keysFindResult") :
			RandomBenchmark(count, seed, resultPath, resultName),
			config(config)
		{}

		void Run() override;

		virtual ~KeysFindBenchmark() = default;
	protected:

		Flavors::Configuration config;
	};

}
