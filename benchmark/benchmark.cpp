#include "benchmark.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>

using namespace Flavors;

namespace Flavors
{
	void to_json(nlohmann::json& j, const DataInfo& info)
	{
		j["n"] = info.N;
		j["min"] = info.Min;
		j["max"] = info.Max;
		j["mean"] = info.Mean;
		j["variance"] = info.Variance();
		j["std"] = std::sqrt(info.VarianceN());
		j["skewness"] = info.Skewness();
		j["kurtosis"] = info.Kurtosis();
	}

	void from_json(const nlohmann::json& j, Configuration& config)
	{
		std::vector<unsigned> levels;

		for (auto level : j)
			levels.push_back(level);

		config.Create(levels);
	}
}

namespace FlavorsBenchmarks
{
	int tryReadIntFromJson(nlohmann::json& j, std::string&& field)
	{
		int val;
		try
		{
			val = j.at(field).get<int>();
			return val;
		}
		catch(...)
		{
			std::cout << field << " missing from configuration file." << std::endl;
			return 0;
		}

		return val;
	}

	float tryReadFloatFromJson(nlohmann::json& j, std::string&& field)
	{
		float val;
		try
		{
			val = j.at(field).get<float>();
			return val;
		}
		catch(...)
		{
			std::cout << field << " missing from configuration file." << std::endl;
			return 0.0;
		}

		return val;
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
}
