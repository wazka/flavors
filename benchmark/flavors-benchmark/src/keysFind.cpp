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
	void KeysFindBenchmark::Run()
	{
		measured.Add("Count", count);
		measured.Add("Seed", seed);
		measured.Add("Config", config);

		timer.Start();
		Keys rawKeys{Configuration::Default32, count};
		rawKeys.FillRandom(seed);
		measured.Add("Generation", timer.Stop());

		timer.Start();
		rawKeys.Sort();
		measured.Add("Sort", timer.Stop());

		timer.Start();
		Keys keys = rawKeys.ReshapeKeys(config);
		measured.Add("Reshape", timer.Stop());
		measured.Add("DataMemory", keys.MemoryFootprint());

		timer.Start();
		Tree tree{keys};
		measured.Add("Build", timer.Stop());
		measured.Add("TreeMemory", tree.MemoryFootprint());
		measured.Add("TreeLevels", tree);

		timer.Start();
		tree.FindKeys(keys, result.Get());
		measured.Add("Find", timer.Stop());

		std::cout << std::endl;
		for(auto countToFind : countsToFind)
		{
			std::cout << "\tRunning for count to find = " << countToFind << "... ";

			Keys randomKeys{config, countToFind};
			randomKeys.FillRandom(seed + 1);
			measured.Add("RandomCount", countToFind);

			timer.Start();
			tree.FindKeys(randomKeys, result.Get());
			measured.Add("FindRandom", timer.Stop());

			timer.Start();
			randomKeys.Sort();
			measured.Add("RandomSort", timer.Stop());

			timer.Start();
			tree.FindKeys(randomKeys, result.Get());
			measured.Add("FindRandomSorted", timer.Stop());
			measured.AddHitCount(result);

			measured.AppendToFile(ResultFullPath());

			std::cout << "finished" << std::endl;
		}
	}
}






















