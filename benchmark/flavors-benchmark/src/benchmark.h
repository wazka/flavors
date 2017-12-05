#pragma once

#include "timer.h"
#include "utils.h"
#include "configuration.h"
#include "tree.h"

#include <map>

namespace FlavorsBenchmarks
{
	class Measured
	{
	public:
		float& operator[](std::string&& measuredValue);
		void AppendToFile(const std::string& path);

	private:
		std::map<std::string, float> measuredValues;
	};

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
