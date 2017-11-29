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
		Benchmark(int count, int seed, const std::string& resultPath, const std::string& resultName) :
			count(count),
			seed(seed),
			resultPath(resultPath),
			resultName(resultName),
			result(count)
		{}

		virtual void Run() = 0;

		std::string ResultFullPath();

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

			void appendToFileFull(const std::string& path);
			void appendToFile(const std::string& path);
		};

		int count;
		int seed;
		std::string resultPath;
		std::string resultName;
		Timer timer;
		Measured measured;
		Flavors::CudaArray<unsigned> result;

		virtual void recordParameters(Flavors::Configuration& config);
		void recordStatistics(Flavors::Tree& tree);
	};

}
