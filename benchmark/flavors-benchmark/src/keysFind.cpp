#include "keysFind.h"
#include "keys.h"
#include "tree.h"
#include "timer.h"

#include <fstream>
#include <iostream>
#include <chrono>

using namespace Flavors;

namespace FlavorsBenchmarks
{
	void KeysFindBenchmark::Run()
	{
		recordParams();

		Timer t;

		t.Start();
		Keys rawKeys{Configuration::DefaultConfig32, count};
		rawKeys.FillRandom(seed);
		result.Generation = t.Stop();

		t.Start();
		rawKeys.Sort();
		result.Sort = t.Stop();

		t.Start();
		auto keys = rawKeys.ReshapeKeys(config);
		result.Reshape = t.Stop();

		t.Start();
		Tree tree{keys};
		result.Build = t.Stop();

		CudaArray<unsigned> findResult{keys.Count};

		t.Start();
		tree.FindKeys(keys, findResult.Get());
		result.Find = t.Stop();

		Keys randomKeys{config, count};
		randomKeys.FillRandom(seed + 1);

		t.Start();
		tree.FindKeys(randomKeys, findResult.Get());
		result.FindRandom = t.Stop();

		randomKeys.Sort();

		t.Start();
		tree.FindKeys(randomKeys, findResult.Get());
		result.FindRandomSorted = t.Stop();

		result.appendToFile(resultPath);
	}

	void KeysFindBenchmark::recordParams()
	{
		std::ofstream file{resultPath.c_str(), std::ios_base::app | std::ios_base::out};

		if(!file)
			file.open(resultPath.c_str(), std::ios_base::app | std::ios_base::out);

		file << count << ";" << seed << ";" << config << ";";
		file.close();
	}

	void KeysFindBenchmark::Result::appendToFile(std::string& path)
	{
		std::ofstream file{path.c_str(), std::ios_base::app | std::ios_base::out};

		if(!file)
			file.open(path.c_str(), std::ios_base::app | std::ios_base::out);

		file << Generation << ";" << Sort << ";" << Reshape << ";" << Build
				<< ";" << Find << ";" << FindRandom << ";" << FindRandomSorted << ";" << std::endl;
		file.close();

	}
}

