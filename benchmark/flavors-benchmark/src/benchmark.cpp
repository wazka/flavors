#include "benchmark.h"

#include <fstream>
#include <iostream>
#include <algorithm>

using namespace Flavors;

namespace FlavorsBenchmarks
{

	float& Measured::operator [](std::string&& measuredValue)
	{
		return measuredValues[measuredValue];
	}

	void Measured::AppendToFile(const std::string& path)
	{
		std::ofstream file{path.c_str(), std::ios_base::app | std::ios_base::out};

		if(!file)
			file.open(path.c_str(), std::ios_base::app | std::ios_base::out);

		for(auto m : measuredValues)
			file << m.second << ";";

		file.close();
	}

	std::string Benchmark::resultFullPath()
	{
		return resultPath + resultName + ".csv";
	}

	void Benchmark::recordStatistics(Flavors::Tree& tree, Flavors::CudaArray<unsigned>& result)
	{
		std::ofstream file{resultFullPath().c_str(), std::ios_base::app | std::ios_base::out};

		file << "{";
		for(auto levelSize : tree.h_LevelsSizes)
			file << levelSize << ",";
		file << "}" << ";";

		auto h_result = result.ToHost();
		auto hitCount = std::count_if(h_result.begin(), h_result.end(), [](int r){ return r != 0;});

		file << hitCount / static_cast<float>(h_result.size()) << std::endl;
		file.close();
	}

	void RandomBenchmark::recordParameters(Configuration& config)
	{
		std::ofstream file{resultFullPath().c_str(), std::ios_base::app | std::ios_base::out};
		file << count << ";" << seed << ";" << config << ";";
		file.close();
	}

	void RandomBenchmark::recordStatistics(Tree& tree)
	{
		Benchmark::recordStatistics(tree, result);
	}
}
