#pragma once

#include "benchmark.h"

namespace FlavorsBenchmarks
{
	class DictionaryBenchmark
	{
	public:

		void Run();
	protected:

		std::string dictionaryPath;
		std::vector<std::string> bookPaths;

		std::string resultPath;
		std::string resultName;

	};

}
