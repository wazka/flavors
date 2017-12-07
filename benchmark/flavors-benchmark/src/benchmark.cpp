#include "benchmark.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>

using namespace Flavors;

namespace FlavorsBenchmarks
{
	float& Measured::operator [](std::string&& measuredValue)
	{
		if(measuredValues.count(measuredValue) == 0)
			labels.push_back(measuredValue);

		return measuredValues[measuredValue];
	}

	void Measured::Add(std::string&& label, Flavors::Configuration& config)
	{
		if (values.count(label) == 0)
			labels.push_back(label);

		values[label] = config.ToString();
	}

	void Measured::Add(std::string && label, Flavors::Tree & tree)
	{
		std::stringstream ss;

		ss << "{";
		for (auto levelSize : tree.h_LevelsSizes)
			ss << levelSize << ",";
		ss << "}";

		if (values.count(label) == 0)
			labels.push_back(label);

		values[label] = ss.str();
	}

	void Measured::Add(std::string && label, std::string& value)
	{
		if (values.count(label) == 0)
			labels.push_back(label);

		values[label] = value;
	}

	void Measured::AddHitCount(Flavors::CudaArray<unsigned>& result)
	{
		auto h_result = result.ToHost();
		auto hitCount = std::count_if(h_result.begin(), h_result.end(), [](int r) { return r != 0; });

		Add("HitRate", hitCount / static_cast<float>(h_result.size()));
	}

	void Measured::AppendToFile(const std::string& path)
	{
		bool addLabel = !exists(path);

		std::ofstream file{path.c_str(), std::ios_base::app | std::ios_base::out};
		if(!file)
			file.open(path.c_str(), std::ios_base::app | std::ios_base::out);

		if (addLabel)
			file << fileLabel();

		for(auto l : labels)
			file << values[l] << ";";

		file << std::endl;
		file.close();
	}

	std::string Measured::fileLabel()
	{
		std::stringstream ss;

		for (auto label : labels)
			ss << label << ";";

		ss << std::endl;

		return ss.str();
	}

	std::string Benchmark::ResultFullPath()
	{
		return resultPath + resultName + ".csv";
	}

	void Benchmark::recordStatistics(Flavors::Tree& tree, Flavors::CudaArray<unsigned>& result)
	{

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
