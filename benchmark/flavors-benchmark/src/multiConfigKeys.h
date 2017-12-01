#pragma once

#include "../json/json.hpp"
#include "configuration.h"
#include "benchmark.h"
#include "keys.h"
#include "tree.h"

//TODO: Could this be done more elegantly?
namespace Flavors
{
	void to_json(nlohmann::json& j, const DataInfo& info);
}

namespace FlavorsBenchmarks
{
	class MultiConfigKeysBenchmark : public Benchmark
	{
	public:
		static std::string Label;

		MultiConfigKeysBenchmark(
				int count,
				int seed,
				const std::vector<Flavors::Configuration>& configs,
				const std::string& resultPath,
				const std::string& resultName = "multiConfigKeysFindResult") :
			Benchmark(count, seed, resultPath, resultName),
			configs(configs)
		{
		}

		void Run() override;

		virtual ~MultiConfigKeysBenchmark() = default;
	protected:
		std::vector<Flavors::Configuration> configs;
		Flavors::Keys rawKeys;

		void generateRawKeys();

		std::vector<nlohmann::json> dataInfo;
		virtual void getDataInfo(nlohmann::json& j);
		void saveDataInfo();

		virtual void runForConfig(Flavors::Configuration& config);
	};
}
