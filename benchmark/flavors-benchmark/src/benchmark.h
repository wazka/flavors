#pragma once

#include "timer.h"
#include "utils.h"
#include "configuration.h"
#include "tree.h"

namespace FlavorsBenchmarks
{
	class Benchmark
	{
public:
		Benchmark(int count, int seed, const std::string& resultPath) :
			count(count),
			seed(seed),
			resultPath(resultPath),
			result(count)
		{}

		virtual void Run() = 0;

		virtual ~Benchmark() = default;
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

			void appendToFileFull(std::string& path);
			void appendToFile(std::string& path);
		};

		int count;
		int seed;
		std::string resultPath;
		Timer timer;
		Measured measured;
		Flavors::CudaArray<unsigned> result;

		virtual void recordParameters(Flavors::Configuration& config);
		void recordStatistics(Flavors::Tree& tree);
	};

}
