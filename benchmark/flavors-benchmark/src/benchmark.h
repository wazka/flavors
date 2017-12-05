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
		Benchmark(const std::string& resultPath, const std::string& resultName) :
			resultPath(resultPath),
			resultName(resultName)
		{}

		virtual void Run() = 0;
		virtual ~Benchmark() = default;

	protected:
		std::string resultPath;
		std::string resultName;

		Timer timer;
		Measured measured;

		std::string resultFullPath();
		void recordStatistics(Flavors::Tree& tree, Flavors::CudaArray<unsigned>& result);
	};

	class RandomBenchmark : public Benchmark
	{
	public:
		RandomBenchmark(int count, int seed, const std::string& resultPath, const std::string& resultName) :
			Benchmark(resultPath, resultName),
			count(count),
			seed(seed),
			result(count)
		{}

protected:
		int count;
		int seed;
		Flavors::CudaArray<unsigned> result;

		virtual void recordParameters(Flavors::Configuration& config);
		void recordStatistics(Flavors::Tree& tree);
	};

}
