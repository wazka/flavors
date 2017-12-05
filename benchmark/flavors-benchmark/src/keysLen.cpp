#include "keysLen.h"

#include <fstream>

using namespace Flavors;

namespace FlavorsBenchmarks
{
	std::string KeysLenBenchmark::Label = "Count;Seed;Depth;Config;Generation;Sort;Reshape;Build;Find;DataMemory;TreeMemory;FindRandom;FindRandomSorted;LevelsSizes;HitRate";

	void KeysLenBenchmark::Run()
	{
		auto config = prepareConfig();
		recordParameters(config);

		timer.Start();
		Keys rawKeys{Configuration::Default(depth), count};
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

		Keys randomKeys{config, count};
		randomKeys.FillRandom(seed + 1);

		timer.Start();
		tree.FindKeys(randomKeys, result.Get());
		measured["FindRandom"] = timer.Stop();

		randomKeys.Sort();

		timer.Start();
		tree.FindKeys(randomKeys, result.Get());
		measured["FindRandomSorted"] = timer.Stop();

		measured.AppendToFile(resultFullPath());
		recordStatistics(tree);
	}

	Configuration KeysLenBenchmark::prepareConfig()
	{
		std::vector<unsigned> levels {firstLevelStride};

		auto currentDepth = firstLevelStride;
		while(currentDepth + levelStride <= depth)
		{
			levels.push_back(levelStride);
			currentDepth += levelStride;
		}

		if(currentDepth < depth)
			levels.push_back(depth - currentDepth);

		return Configuration{levels};
	}

	void KeysLenBenchmark::recordParameters(Flavors::Configuration& config)
	{
		std::ofstream file{resultFullPath().c_str(), std::ios_base::app | std::ios_base::out};
		file << count << ";" << seed << ";" << depth << ";" << config << ";";
		file.close();
	}
}
