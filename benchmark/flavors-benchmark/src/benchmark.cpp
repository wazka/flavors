#include "benchmark.h"

#include <fstream>
#include <iostream>
#include <algorithm>

using namespace Flavors;

namespace FlavorsBenchmarks
{
	void Benchmark::Measured::appendToFileFull(const std::string& path)
	{
		std::ofstream file{path.c_str(), std::ios_base::app | std::ios_base::out};

		if(!file)
			file.open(path.c_str(), std::ios_base::app | std::ios_base::out);

		file << Generation << ";" << Sort << ";" << Reshape << ";" << Build
			<< ";" << Find << ";" << FindRandom << ";" << FindRandomSorted << ";";
		file.close();

	}

	void Benchmark::Measured::appendToFile(const std::string& path)
	{
		std::ofstream file{path.c_str(), std::ios_base::app | std::ios_base::out};

		if(!file)
			file.open(path.c_str(), std::ios_base::app | std::ios_base::out);

		file << Generation << ";" << Sort << ";" << Reshape << ";" << Build
				<< ";" << Find << ";";
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
