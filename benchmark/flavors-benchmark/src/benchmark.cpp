#include "benchmark.h"

#include <fstream>
#include <iostream>
#include <algorithm>

using namespace Flavors;

namespace FlavorsBenchmarks
{

	float& Measured::operator [](std::string&& measuredValue)
	{
		if(measuredValues.count(measuredValue) == 0)
			labels.push_back(measuredValue);

		return measuredValues[measuredValue];
	}

	void Measured::AppendToFile(const std::string& path)
	{
		std::ofstream file{path.c_str(), std::ios_base::app | std::ios_base::out};

		if(!file)
			file.open(path.c_str(), std::ios_base::app | std::ios_base::out);

		for(auto l : labels)
			file << measuredValues[l] << ";";

		file.close();
	}

	std::string Benchmark::ResultFullPath()
	{
		return resultPath + resultName + ".csv";
	}

	void Benchmark::recordStatistics(Flavors::Tree& tree, Flavors::CudaArray<unsigned>& result)
	{
		std::ofstream file{ResultFullPath().c_str(), std::ios_base::app | std::ios_base::out};

		file << "{";
		for(auto levelSize : tree.h_LevelsSizes)
			file << levelSize << ",";
		file << "}" << ";";

		auto h_result = result.ToHost();
		auto hitCount = std::count_if(h_result.begin(), h_result.end(), [](int r){ return r != 0;});

		file << hitCount / static_cast<float>(h_result.size()) << std::endl;
		file.close();
	}

	Configuration Benchmark::prepareConfig(unsigned firstLevelStride, unsigned levelStride, unsigned depth)
	{
		std::vector<unsigned> levels {firstLevelStride};

		auto currentDepth = firstLevelStride;
		while(currentDepth + levelStride <= depth)
		{
			levels.push_back(levelStride);
			currentDepth += levelStride;
		}

		if(currentDepth < depth)
			levels.push_back(depth - currentDepth);

		return Configuration{levels};
	}

	void RandomBenchmark::recordParameters(Configuration& config)
	{
		std::ofstream file{ResultFullPath().c_str(), std::ios_base::app | std::ios_base::out};
		file << count << ";" << seed << ";" << config << ";";
		file.close();
	}

	void RandomBenchmark::recordStatistics(Tree& tree)
	{
		Benchmark::recordStatistics(tree, result);
	}
}
