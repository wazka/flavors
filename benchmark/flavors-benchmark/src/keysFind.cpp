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
	std::string KeysFindBenchmark::Label = "Count;Seed;Config;Generation;Sort;Reshape;Build;Find;DataMemory;TreeMemory;RandomCount;FindRandom;RandomSort;FindRandomSorted;LevelsSizes;HitRate";

	void KeysFindBenchmark::Run()
	{
		timer.Start();
		Keys rawKeys{Configuration::Default32, count};
		rawKeys.FillRandom(seed);
		measured["Generation"] = timer.Stop();

		timer.Start();
		rawKeys.Sort();
		measured["Sort"] = timer.Stop();

		timer.Start();
		Keys keys = rawKeys.ReshapeKeys(config);
		measured["Reshape"] = timer.Stop();
		measured["DataMemory"] = keys.MemoryFootprint();

		timer.Start();
		Tree tree{keys};
		measured["Build"] = timer.Stop();
		measured["TreeMemory"] = tree.MemoryFootprint();

		timer.Start();
		tree.FindKeys(keys, result.Get());
		measured["Find"] = timer.Stop();

		std::cout << std::endl;
		for(auto countToFind : countsToFind)
		{
			std::cout << "\tRunning for count to find = " << countToFind << "... ";

			Keys randomKeys{config, countToFind};
			randomKeys.FillRandom(seed + 1);

			measured["RandomCount"] = countToFind;

			timer.Start();
			tree.FindKeys(randomKeys, result.Get());
			measured["FindRandom"] = timer.Stop();

			timer.Start();
			randomKeys.Sort();
			measured["RandomSort"] = timer.Stop();

			timer.Start();
			tree.FindKeys(randomKeys, result.Get());
			measured["FindRandomSorted"] = timer.Stop();

			recordParameters(config);
			measured.AppendToFile(ResultFullPath());
			recordStatistics(tree);

			std::cout << "finished" << std::endl;
		}
	}
}






















