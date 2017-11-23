#pragma once

#include "configuration.h"
#include "benchmark.h"
#include "keys.h"
#include "tree.h"

namespace FlavorsBenchmarks
{
	class MultiConfigKeysBenchmark : public Benchmark
	{
	public:
		static std::string Label;

		MultiConfigKeysBenchmark(int count, int seed, const std::vector<Flavors::Configuration>& configs, const std::string& resultPath) :
			Benchmark(count, seed, resultPath),
			configs(configs)
		{
		}

		virtual void Run() override;

		virtual ~MultiConfigKeysBenchmark() = default;
	protected:
		std::vector<Flavors::Configuration> configs;
		Flavors::Keys rawKeys;

		void generateRawKeys();
		virtual void runForConfig(Flavors::Configuration& config);
	};
}
