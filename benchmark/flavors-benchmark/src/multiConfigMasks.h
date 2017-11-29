#pragma once
#include "multiConfigKeys.h"

namespace FlavorsBenchmarks
{
	class MultiConfigMasksBenchmark : public MultiConfigKeysBenchmark
	{
	public:
		static std::string Label;

		MultiConfigMasksBenchmark(
				int count,
				int seed,
				const std::vector<Flavors::Configuration>& configs,
				const std::string& resultPath,
				int minLen,
				int maxLen,
				const std::string& resultName = "multiConfigMasksFindResult") :
			MultiConfigKeysBenchmark(count, seed, configs, resultPath, resultName),
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
		virtual void getDataInfo(nlohmann::json& j) override;
		virtual void runForConfig(Flavors::Configuration& config) override;
		virtual void recordParameters(Flavors::Configuration& config) override;
	};
}
