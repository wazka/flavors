#pragma once

#include "benchmark.h"

namespace FlavorsBenchmarks
{
	class KeysLenBenchmark : public RandomBenchmark
	{
	public:
		static std::string Label;

		KeysLenBenchmark(
				int count,
				int seed,
				unsigned depth,
				unsigned firstLevelStride,
				unsigned levelStride,
				const std::string& resultPath,
				const std::string& resultName = "keysLenResult") :
			RandomBenchmark(count, seed, resultPath, resultName),
			depth(depth),
			firstLevelStride(firstLevelStride),
			levelStride(levelStride)

	{}

		virtual void Run() override;

	protected:
		unsigned depth;
		unsigned firstLevelStride;
		unsigned levelStride;

		Flavors::Configuration prepareConfig();
		virtual void recordParameters(Flavors::Configuration& config) override;
	};

}
