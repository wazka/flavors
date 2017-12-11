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

		float Stop()
		{
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start ).count();

			return duration;
		}

	private:
		std::chrono::high_resolution_clock::time_point start;

	};
}
