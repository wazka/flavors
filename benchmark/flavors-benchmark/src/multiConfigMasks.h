#pragma once
#include "multiConfigKeys.h"

namespace FlavorsBenchmarks
{
	class MultiConfigMasksBenchmark : public MultiConfigKeysBenchmark
	{
	public:
		static std::string Label;

		MultiConfigMasksBenchmark(int count, int seed, const std::vector<Flavors::Configuration>& configs, const std::string& resultPath, int minLen, int maxLen) :
			MultiConfigKeysBenchmark(count, seed, configs, resultPath),
			minLen(minLen),
			maxLen(maxLen)
		{
		}

		virtual void Run() override;

	protected:
		int minLen;
		int maxLen;
		Flavors::Masks rawMasks;

		void generateRawMasks();
		virtual void runForConfig(Flavors::Configuration& config) override;
		virtual void recordParameters(Flavors::Configuration& config) override;
	};
}
