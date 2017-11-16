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

		void Run();

		const std::string Label = "Count;Seed;Config;Generation;Sort;Reshape;Build;Find;FindRandom;FindRandomSorted";

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

		void recordParams();

		Measured measured;
		Flavors::CudaArray<unsigned> result;

		Timer timer;

		Flavors::Tree tree;
		void buildTree(Flavors::Keys& keys);
		Flavors::Keys prepareKeys();

		void runForKeys();
		void runForRandKeys();
	};

}
