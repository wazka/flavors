#pragma once

#include "benchmark.h"

namespace FlavorsBenchmarks
{
	class WordsBenchmark
	{
	public:
		const int BitsPerLetter = 8;

		WordsBenchmark(
			std::string dictionaries,
			std::string resultFile,
			std::vector<Flavors::Configuration>&& configs, 
			int deviceId) :
				dictionaries(dictionaries),
				resultFile(resultFile),
				configs(configs),
				deviceId(deviceId)
		{
		}

		WordsBenchmark(nlohmann::json& j):
			WordsBenchmark(
				tryReadFromJson<std::string>(j, "dictionaries"),
				tryReadFromJson<std::string>(j, "resultFile"),
				tryReadFromJson<std::vector<Flavors::Configuration>>(j, "configs"), 
				tryReadIntFromJson(j, "deviceId"))
		{
		}

		void Run();

		std::vector<Flavors::Configuration> configs;
		int deviceId;
		std::string dictionaries;
		std::string resultFile;

		Timer timer;
		Measured measured;
		std::string deviceName;

	private:

		void runForDictionary(std::string& path);
		std::vector<std::string> loadWordsFromFile(std::string path);
		Flavors::Keys readWords(std::string path);
	};
}
