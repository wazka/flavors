#pragma once

#include "utils.h"
#include "configuration.h"
#include "tree.h"

#include <map>
#include <fstream>
#include <type_traits>
#include <chrono>

#include "json.hpp"

namespace Flavors
{
	void to_json(nlohmann::json& j, const DataInfo& info);
	void from_json(const nlohmann::json& j, Configuration& config);
}

namespace FlavorsBenchmarks
{
	int tryReadIntFromJson(nlohmann::json& j, std::string&& field);
	float tryReadFloatFromJson(nlohmann::json& j, std::string&& field);

	template<typename T>
	T tryReadFromJson(nlohmann::json& j, std::string&& field)
	{
		T val;
		try
		{
			val = j.at(field).get<T>();
			return val;
		}
		catch(...)
		{
			std::cout << field << " missing from configuration file." << std::endl;
		}

		return val;
	}

	template<typename T>
	void saveDataInfo(
			T& rawKeys,
			int count,
			int seed,
			int dataItemLength,
			int deviceId,
			std::string& deviceName,
			std::string& dataInfoDirectory,
			int maxMaskLength = 0,
			int minMaskLength = 0)
	{
		nlohmann::json j;

		Flavors::Configuration binaryConfig = Flavors::Configuration::Binary(dataItemLength);
		j["dataInfo"] = rawKeys.ReshapeKeys(binaryConfig).GetInfo();
		j["seed"] = seed;
		j["count"] = count;
		j["dataItemLength"] = dataItemLength;
		j["deviceId"] = deviceId;
		j["deviceName"] = deviceName;

		std::stringstream fileName;
		fileName << std::to_string(count) << "_" << std::to_string(seed) << "_" << std::to_string(dataItemLength);
		fileName << "_" << deviceId << "_" << deviceName;

		if(maxMaskLength != 0)
		{
			j["maxMaskLength"] = maxMaskLength;
			j["minMaskLength"] = minMaskLength;

			fileName << "_" << maxMaskLength << "_" << minMaskLength;
		}

		fileName << ".json";

		std::ofstream dataInfoFile;
		dataInfoFile.open(dataInfoDirectory + fileName.str());

		if(!dataInfoFile.good())
		{
			std::cout << "\t WARNING: Unable to open data info file. File will be written to current directory" << std::endl;
			dataInfoFile.open(fileName.str());
		}

		if(dataInfoFile.good())
		{
			dataInfoFile << j.dump(3);
			dataInfoFile.close();
		}
		else
			std::cout << "\t ERROR: Unable to open data info file." << std::endl;
	}

	class Measured
	{
	public:
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

	class Timer
	{
	public:
		void Start()
		{
			start = std::chrono::high_resolution_clock::now();
		}

		template<typename Unit>
		unsigned long long Stop()
		{
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<Unit>( end - start ).count();

			return duration;
		}

		unsigned long long Stop()
		{
			return Stop<std::chrono::nanoseconds>();
		}

	private:
		std::chrono::high_resolution_clock::time_point start;

	};
}
