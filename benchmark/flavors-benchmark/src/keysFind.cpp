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
	std::string KeysFindBenchmark::Label = "Count;Seed;Config;Generation;Sort;Reshape;Build;Find;FindRandom;FindRandomSorted;LevelsSizes;HitRate";

	void KeysFindBenchmark::Run()
	{
		recordParameters(config);

		timer.Start();
		Keys rawKeys{Configuration::Default32, count};
		rawKeys.FillRandom(seed);
		measured.Generation = timer.Stop();

		timer.Start();
		rawKeys.Sort();
		measured.Sort = timer.Stop();

		timer.Start();
		Keys keys = rawKeys.ReshapeKeys(config);
		measured.Reshape = timer.Stop();

		timer.Start();
		Tree tree{keys};
		measured.Build = timer.Stop();

		timer.Start();
		tree.FindKeys(keys, result.Get());
		measured.Find = timer.Stop();

		Keys randomKeys{config, count};
		randomKeys.FillRandom(seed + 1);

		timer.Start();
		tree.FindKeys(randomKeys, result.Get());
		measured.FindRandom = timer.Stop();

		randomKeys.Sort();

		timer.Start();
		tree.FindKeys(randomKeys, result.Get());
		measured.FindRandomSorted = timer.Stop();

		measured.appendToFileFull(ResultFullPath());
		recordStatistics(tree);
	}
}






















