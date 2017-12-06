#pragma once

#include "keysFind.h"

namespace FlavorsBenchmarks
{
	class MasksFindBenchmark : public KeysFindBenchmark
	{
	public:
		static std::string Label;

		MasksFindBenchmark(
				int count,
				int seed,
				const Flavors::Configuration& config,
				std::vector<int> countsToFind,
				const std::string& resultPath,
				int minLen,
				int maxLen,
				const std::string& resultName = "masksFindResult") :
			KeysFindBenchmark(count, seed, config, countsToFind,resultPath, resultName),
			minLen(minLen),
			maxLen(maxLen)
		{}

		void Run() override;

	private:
		int minLen;
		int maxLen;

		virtual void recordParameters(Flavors::Configuration& config) override;
	};
}
