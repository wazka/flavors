#pragma once

#include "benchmark.h"

namespace FlavorsBenchmarks
{
	class DictionaryBenchmark : public Benchmark
	{
	public:
		static std::string Label;

		const int BitsPerLetter = 8;

		DictionaryBenchmark(
				std::string dictionaryPath,
				std::vector<std::string> bookPaths,
				const std::string& resultPath,
				const std::string& resultName = "dictionaryResult") :
			Benchmark(resultPath, resultName),
			dictionaryPath(dictionaryPath),
			bookPaths(bookPaths),
			maxWordLen(0)
		{}


		void Run();
	protected:

		std::string dictionaryPath;
		std::vector<std::string> bookPaths;

		unsigned maxWordLen;

	private:
		std::vector<std::string> readWords(std::string path, unsigned& maxWordLen);
		Flavors::Masks wordsToMasks(std::vector<std::string>& words, Flavors::Configuration& config);

		Flavors::Masks loadDictionary();
		Flavors::Masks loadBook(std::string& bookPath, Flavors::Configuration& config);

		Flavors::Configuration prepareConfig(unsigned bitsPerLetter, unsigned maxWordLen);

	};

}
