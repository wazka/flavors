#include "../json/json.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "configuration.h"
#include "keysFind.h"
#include "masksFind.h"
#include "multiConfigKeys.h"
#include "multiConfigMasks.h"

void runKeysFind(nlohmann::json& j);
void runMasksFind(nlohmann::json& j);
void runMultiConfigKeysFind(nlohmann::json& j);
void runMultiConfigMasksFind(nlohmann::json& j);

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		std::cerr << "Wrong arguments. Provide only a path to a json config file." << std::endl;
		return EXIT_FAILURE;
	}

	std::ifstream file(argv[1]);

	if (!file)
	{
		std::cerr << "Unable to open a file" << std::endl;
		return EXIT_FAILURE;
	}

	nlohmann::json j;
	file >> j;

	file.close();

	if(j["benchmark"] == "keysFind")
		runKeysFind(j);
	else if(j["benchmark"] == "masksFind")
		runMasksFind(j);
	else if (j["benchmark"] == "multiConfigKeysFind")
		runMultiConfigKeysFind(j);
	else if (j["benchmark"] == "multiConfigMasksFind")
		runMultiConfigMasksFind(j);
	else
	{
		std::cerr << "Unknown benchmark type." << std::endl;
		return EXIT_FAILURE;
	}
}

inline bool exists (const std::string& name)
{
    std::ifstream f(name.c_str());
    return f.good();
}

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

void runKeysFind(nlohmann::json& j)
{
	auto counts = tryReadFromJson<std::vector<int>>(j, "counts");
	auto seeds = tryReadFromJson<std::vector<int>>(j, "seeds");
	auto configs = tryReadFromJson<std::vector<std::vector<unsigned>>>(j, "configs");
	auto path = tryReadFromJson<std::string>(j, "resultFilePath");

	for(auto count : counts)
		for(auto seed : seeds)
			for(auto levels : configs)
			{
				try
				{
					std::cout << "Starting benchmark for count = " << count << ", seed = " << seed << ", config = {";
					for(auto level : levels)
						std::cout << level << ",";
					std::cout << "} ... ";

					Flavors::Configuration config {levels};
					FlavorsBenchmarks::KeysFindBenchmark bench{count, seed, config, path };

					if(!exists(bench.ResultFullPath()))
					{
						std::ofstream file{bench.ResultFullPath().c_str(), std::ios_base::app | std::ios_base::out};
						file << FlavorsBenchmarks::KeysFindBenchmark::Label << std::endl;
						file.close();
					}

					bench.Run();

					std::cout << "success" << std::endl;
				}
				catch(...)
				{
					std::cout << "failed due to exception" << std::endl;
				}
			}
}

void runMasksFind(nlohmann::json& j)
{
	auto counts = tryReadFromJson<std::vector<int>>(j, "counts");
	auto seeds = tryReadFromJson<std::vector<int>>(j, "seeds");
	auto configs = tryReadFromJson<std::vector<std::vector<unsigned>>>(j, "configs");
	auto path = tryReadFromJson<std::string>(j, "resultFilePath");
	auto minLen = tryReadFromJson<int>(j, "minLen");
	auto maxLen = tryReadFromJson<int>(j, "maxLen");

	for(auto count : counts)
		for(auto seed : seeds)
			for(auto levels : configs)
			{
				try
				{
					std::cout << "Starting benchmark for count = " << count << ", seed = " << seed << ", config = {";
					for(auto level : levels)
						std::cout << level << ",";
					std::cout << "} ... ";

					Flavors::Configuration config {levels};
					FlavorsBenchmarks::MasksFindBenchmark bench{count, seed, config, path, minLen, maxLen };

					if(!exists(bench.ResultFullPath()))
					{
						std::ofstream file{bench.ResultFullPath().c_str(), std::ios_base::app | std::ios_base::out};
						file << FlavorsBenchmarks::MasksFindBenchmark::Label << std::endl;
						file.close();
					}

					bench.Run();

					std::cout << "success" << std::endl;
				}
				catch(...)
				{
					std::cout << "failed due to exception" << std::endl;
				}
			}
}

void runMultiConfigKeysFind(nlohmann::json& j)
{
	auto counts = tryReadFromJson<std::vector<int>>(j, "counts");
	auto seeds = tryReadFromJson<std::vector<int>>(j, "seeds");
	auto h_configs = tryReadFromJson<std::vector<std::vector<unsigned>>>(j, "configs");
	auto path = tryReadFromJson<std::string>(j, "resultFilePath");

	std::vector<Flavors::Configuration> configs;
	for(auto config : h_configs)
		configs.push_back(Flavors::Configuration{config});

	for(auto count : counts)
		for(auto seed : seeds)
		{
			try
			{
				std::cout << "Starting benchmark for count = " << count << ", seed = " << seed << " ... ";
				FlavorsBenchmarks::MultiConfigKeysBenchmark bench{count, seed, configs, path};

				if(!exists(bench.ResultFullPath()))
				{
					std::ofstream file{bench.ResultFullPath().c_str(), std::ios_base::app | std::ios_base::out};
					file << FlavorsBenchmarks::MultiConfigKeysBenchmark::Label << std::endl;
					file.close();
				}

				bench.Run();

				std::cout << "success" << std::endl;
			}
			catch(...)
			{
				std::cout << "failed due to exception" << std::endl;
			}
		}
}

void runMultiConfigMasksFind(nlohmann::json& j)
{
	auto counts = tryReadFromJson<std::vector<int>>(j, "counts");
	auto seeds = tryReadFromJson<std::vector<int>>(j, "seeds");
	auto h_configs = tryReadFromJson<std::vector<std::vector<unsigned>>>(j, "configs");
	auto path = tryReadFromJson<std::string>(j, "resultFilePath");
	auto minLen = tryReadFromJson<int>(j, "minLen");
	auto maxLen = tryReadFromJson<int>(j, "maxLen");

	std::vector<Flavors::Configuration> configs;
	for(auto config : h_configs)
		configs.push_back(Flavors::Configuration{config});

	for(auto count : counts)
		for(auto seed : seeds)
		{
			try
			{
				std::cout << "Starting benchmark for count = " << count << ", seed = " << seed << " ... ";
				FlavorsBenchmarks::MultiConfigMasksBenchmark bench{count, seed, configs, path, minLen, maxLen};

				if(!exists(bench.ResultFullPath()))
				{
					std::ofstream file{bench.ResultFullPath().c_str(), std::ios_base::app | std::ios_base::out};
					file << FlavorsBenchmarks::MultiConfigMasksBenchmark::Label << std::endl;
					file.close();
				}

				bench.Run();

				std::cout << "success" << std::endl;
			}
			catch(...)
			{
				std::cout << "failed due to exception" << std::endl;
			}
		}
}
