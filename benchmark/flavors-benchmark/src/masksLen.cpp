#include "masksLen.h"

#include <fstream>

using namespace Flavors;

namespace FlavorsBenchmarks
{
	std::string MasksLenBenchmark::Label = "Count;Seed;Max;Min;Depth;Config;Generation;Sort;Reshape;Build;Find;DataMemory;TreeMemory;FindRandom;FindRandomSorted;LevelsSizes;HitRate";

	void MasksLenBenchmark::Run()
	{
		auto config = prepareConfig();
		recordParameters(config);

		timer.Start();
		Masks rawMasks{Configuration::Default(depth), count};
		rawMasks.FillRandom(seed, max, min);
		measured["Generation"] = timer.Stop();

		timer.Start();
		rawMasks.Sort();
		measured["Sort"] = timer.Stop();

		timer.Start();
		Masks masks = rawMasks.ReshapeMasks(config);
		measured["Reshape"] = timer.Stop();
		measured["DataMemory"] = masks.MemoryFootprint();

		timer.Start();
		Tree tree{masks};
		measured["Build"] = timer.Stop();
		measured["TreeMemory"] = tree.MemoryFootprint();

		timer.Start();
		tree.FindMasks(masks, result.Get());
		measured["Find"] = timer.Stop();

		Masks randomMasks{config, count};
		randomMasks.FillRandom(seed + 1);

		timer.Start();
		tree.FindMasks(randomMasks, result.Get());
		measured["FindRandom"] = timer.Stop();

		randomMasks.Sort();

		timer.Start();
		tree.FindMasks(randomMasks, result.Get());
		measured["FindRandomSorted"] = timer.Stop();

		measured.AppendToFile(ResultFullPath());
		recordStatistics(tree);
	}

	void MasksLenBenchmark::recordParameters(Configuration& config)
	{
		std::ofstream file{ResultFullPath().c_str(), std::ios_base::app | std::ios_base::out};
		file << count << ";" << seed << ";" << max << ";" << min << ";" << depth << ";" << config << ";";
		file.close();
	}

}
