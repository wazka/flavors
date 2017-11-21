#include "keysFind.h"
#include "keys.h"
#include "tree.h"

#include <fstream>
#include <iostream>
#include <chrono>
#include <thread>

using namespace Flavors;

namespace FlavorsBenchmarks
{
	std::string KeysFindBenchmark::Label = "Count;Seed;Config;Generation;Sort;Reshape;Build;Find;FindRandom;FindRandomSorted";

	void KeysFindBenchmark::Run()
	{
		recordParams();

		runForKeys();
		runForRandKeys();

		measured.appendToFile(resultPath);
	}

	Flavors::Keys KeysFindBenchmark::prepareKeys()
	{
		timer.Start();
		Keys rawKeys{Configuration::DefaultConfig32, count};
		rawKeys.FillRandom(seed);
		measured.Generation = timer.Stop();

		timer.Start();
		rawKeys.Sort();
		measured.Sort = timer.Stop();

		timer.Start();
		auto keys = rawKeys.ReshapeKeys(config);
		measured.Reshape = timer.Stop();

		return keys;
	}

	void KeysFindBenchmark::buildTreeFromKeys(Flavors::Keys& keys)
	{
		timer.Start();
		Tree localTree{keys};
		measured.Build = timer.Stop();

		tree = std::move(localTree);
	}

	void KeysFindBenchmark::runForKeys()
	{
		auto keys = prepareKeys();
		buildTreeFromKeys(keys);

		timer.Start();
		tree.FindKeys(keys, result.Get());
		measured.Find = timer.Stop();
	}

	void KeysFindBenchmark::runForRandKeys()
	{
		Keys randomKeys{config, count};
		randomKeys.FillRandom(seed + 1);

		timer.Start();
		tree.FindKeys(randomKeys, result.Get());
		measured.FindRandom = timer.Stop();

		randomKeys.Sort();

		timer.Start();
		tree.FindKeys(randomKeys, result.Get());
		measured.FindRandomSorted = timer.Stop();
	}

	void KeysFindBenchmark::recordParams()
	{
		std::ofstream file{resultPath.c_str(), std::ios_base::app | std::ios_base::out};
		file << count << ";" << seed << ";" << config << ";";
		file.close();
	}

	void KeysFindBenchmark::Measured::appendToFile(std::string& path)
	{
		std::ofstream file{path.c_str(), std::ios_base::app | std::ios_base::out};

		if(!file)
			file.open(path.c_str(), std::ios_base::app | std::ios_base::out);

		file << Generation << ";" << Sort << ";" << Reshape << ";" << Build
				<< ";" << Find << ";" << FindRandom << ";" << FindRandomSorted << std::endl;
		file.close();

	}
}

