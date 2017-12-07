#pragma once

#include "timer.h"
#include "utils.h"
#include "configuration.h"
#include "tree.h"

#include <map>
#include <fstream>
#include <type_traits>

namespace FlavorsBenchmarks
{
	class Measured
	{
	public:
		float& operator[](std::string&& measuredValue);

		template<
			typename T, 
			typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type>
		inline void Add(std::string && label, T value)
		{
			if (values.count(label) == 0)
				labels.push_back(label);

			values[label] = std::to_string(value);
		}

		void Add(std::string&& label, Flavors::Configuration& config);
		void Add(std::string&& label, Flavors::Tree& tree);
		void Add(std::string&& label, std::string& value);
		void AddHitCount(Flavors::CudaArray<unsigned>& result);

		void AppendToFile(const std::string& path);

	private:
		std::map<std::string, float> measuredValues;

		std::map<std::string, std::string> values;
		std::vector<std::string> labels;

		inline bool exists(const std::string& name)
		{
			std::ifstream f(name.c_str());
			return f.good();
		}

		std::string fileLabel();
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
