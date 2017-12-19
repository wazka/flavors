#pragma once

#include "benchmark.h"

namespace FlavorsBenchmarks
{
	class DictionaryBenchmark
	{
	public:
		const int BitsPerLetter = 8;

		DictionaryBenchmark(
				std::string dictionaryPath,
				std::vector<std::string> bookPaths,
				const std::string& resultFile,
				int deviceId) :
			dictionaryPath(dictionaryPath),
			bookPaths(bookPaths),
			resultFile(resultFile),
			deviceId(deviceId),
			maxWordLen(0)
		{}

		DictionaryBenchmark(nlohmann::json& j) :
			DictionaryBenchmark(
					tryReadFromJson<std::string>(j, "dictionaryFile"),
					tryReadFromJson<std::vector<std::string>>(j, "bookFiles"),
					tryReadFromJson<std::string>(j, "resultFile"),
					tryReadIntFromJson(j, "deviceId"))
		{
		}


		void Run();
	protected:

		std::string resultFile;
		std::string dictionaryPath;
		std::vector<std::string> bookPaths;

		unsigned maxWordLen;

		Timer timer;
		Measured measured;

		int deviceId;
		std::string deviceName;

	private:
		std::vector<std::string> readWords(std::string path, unsigned& maxWordLen);
		Flavors::Masks wordsToMasks(std::vector<std::string>& words, Flavors::Configuration& config);

		Flavors::Masks loadDictionary();
		Flavors::Masks loadBook(std::string& bookPath, Flavors::Configuration& config);

		Flavors::Configuration prepareConfig(unsigned bitsPerLetter, unsigned maxWordLen);

	};

}
