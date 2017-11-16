#pragma once
#include <configuration.h>
#include <string>

#include "tree.h"
#include "timer.h"

namespace FlavorsBenchmarks
{
	class KeysFindBenchmark
	{
	public:
		KeysFindBenchmark(int count, int seed, const Flavors::Configuration& config, const std::string& resultPath) :
			count(count),
			seed(seed),
			config(config),
			resultPath(resultPath),
			result(count)
		{}

		virtual void Run();

		static std::string Label;

		virtual ~KeysFindBenchmark() = default;
	protected:
		struct Measured
		{
			float Generation;
			float Sort;
			float Reshape;
			float Build;
			float Find;
			float FindRandom;
			float FindRandomSorted;

			void appendToFile(std::string& path);
		};

		int count;
		int seed;
		std::string resultPath;
		Flavors::Configuration config;

		virtual void recordParams();

		Measured measured;
		Flavors::CudaArray<unsigned> result;

		Timer timer;

		Flavors::Tree tree;
		void buildTreeFromKeys(Flavors::Keys& keys);
		Flavors::Keys prepareKeys();

		void runForKeys();
		void runForRandKeys();
	};

}
