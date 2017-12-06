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
		std::vector<std::string> labels;
	};

	class Benchmark
	{
	public:
		Benchmark(const std::string& resultPath, const std::string& resultName) :
			resultPath(resultPath),
			resultName(resultName)
		{}

		std::string ResultFullPath();
		virtual void Run() = 0;
		virtual ~Benchmark() = default;

	protected:
		std::string resultPath;
		std::string resultName;

		Timer timer;
		Measured measured;

		void recordStatistics(Flavors::Tree& tree, Flavors::CudaArray<unsigned>& result);
		Flavors::Configuration prepareConfig(unsigned firstLevelStride, unsigned levelStride, unsigned depth);
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
