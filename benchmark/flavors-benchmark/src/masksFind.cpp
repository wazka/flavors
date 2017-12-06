#include "masksFind.h"

#include "configuration.h"
#include <fstream>
#include <string>

using namespace Flavors;

namespace FlavorsBenchmarks
{
	std::string MasksFindBenchmark::Label = "Count;Seed;Config;MinLen;MaxLen;Generation;Sort;Reshape;Build;Find;DataMemory;TreeMemory;FindRandom;FindRandomSorted;LevelsSizes;HitRate";

	void MasksFindBenchmark::Run()
	{
		recordParameters(config);

		timer.Start();
		Masks rawMasks{Configuration::Default32, count};
		rawMasks.FillRandom(seed, maxLen, minLen);
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
		randomMasks.FillRandom(seed, maxLen, minLen);

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

	void MasksFindBenchmark::recordParameters(Flavors::Configuration& config)
	{
		std::ofstream file{ResultFullPath().c_str(), std::ios_base::app | std::ios_base::out};
		file << count << ";" << seed << ";" << config << ";" << minLen << ";" << maxLen << ";";
		file.close();
	}
}
