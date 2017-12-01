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

	void Benchmark::recordParameters(Configuration& config)
	{
		std::ofstream file{ResultFullPath().c_str(), std::ios_base::app | std::ios_base::out};
		file << count << ";" << seed << ";" << config << ";";
		file.close();
	}

	void Benchmark::recordStatistics(Tree& tree)
	{
		std::ofstream file{ResultFullPath().c_str(), std::ios_base::app | std::ios_base::out};

		file << "{";
		for(auto levelSize : tree.h_LevelsSizes)
			file << levelSize << ",";
		file << "}" << ";";

		auto h_result = result.ToHost();
		auto hitCount = std::count_if(h_result.begin(), h_result.end(), [](int r){ return r != 0;});

		file << hitCount / static_cast<float>(count) << std::endl;
		file.close();
	}

	std::string Benchmark::ResultFullPath()
	{
		return resultPath + resultName + ".csv";
	}
}
