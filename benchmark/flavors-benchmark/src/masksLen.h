#pragma once

#include "keysLen.h"

namespace FlavorsBenchmarks
{
	class MasksLenBenchmark : public KeysLenBenchmark
	{
	public:
		static std::string Label;

		MasksLenBenchmark(
				int count,
				int seed,
				unsigned depth,
				unsigned firstLevelStride,
				unsigned levelStride,
				int max,
				int min,
				const std::string& resultPath,
				const std::string& resultName = "masksLenResult") :
			KeysLenBenchmark(count, seed, depth, firstLevelStride, levelStride, resultPath, resultName),
			max(max),
			min(min)
		{}

		void Run() override;

	protected:
		int max;
		int min;

		void recordParameters(Flavors::Configuration& config) override;
	};

}
