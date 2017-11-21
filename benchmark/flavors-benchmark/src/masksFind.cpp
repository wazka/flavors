#include "masksFind.h"

#include <configuration.h>
#include <fstream>
#include <string>

using namespace Flavors;

namespace FlavorsBenchmarks
{
	std::string MasksFindBenchmark::Label = "Count;Seed;Config;MinLen;MaxLen;Generation;Sort;Reshape;Build;Find;FindRandom;FindRandomSorted;LevelsSizes;HitRate";

	void MasksFindBenchmark::Run()
	{
		recordParams();

		runForMasks();
		runForRandMasks();

		measured.appendToFile(resultPath);
		recordStatistics();
	}

	Flavors::Masks MasksFindBenchmark::prepareMasks()
	{
		timer.Start();
		Masks rawMasks{Configuration::DefaultConfig32, count};
		rawMasks.FillRandom(seed, maxLen, minLen);

		measured.Generation = timer.Stop();

		timer.Start();
		rawMasks.Sort();
		measured.Sort = timer.Stop();

		timer.Start();
		auto masks = rawMasks.ReshapeMasks(config);
		measured.Reshape = timer.Stop();

		return masks;
	}

	void MasksFindBenchmark::buildTreeFromMasks(Masks& masks)
	{
		timer.Start();
		Tree localTree{masks};
		measured.Build = timer.Stop();

		tree = std::move(localTree);
	}

	void MasksFindBenchmark::runForMasks()
	{
		auto masks = prepareMasks();
		buildTreeFromMasks(masks);

		timer.Start();
		tree.FindMasks(masks, result.Get());
		measured.Find = timer.Stop();
	}

	void MasksFindBenchmark::runForRandMasks()
	{
		Masks randomMasks{config, count};
		randomMasks.FillRandom(seed, maxLen, minLen);

		timer.Start();
		tree.FindMasks(randomMasks, result.Get());
		measured.FindRandom = timer.Stop();

		randomMasks.Sort();

		timer.Start();
		tree.FindMasks(randomMasks, result.Get());
		measured.FindRandomSorted = timer.Stop();
	}

	void MasksFindBenchmark::recordParams()
	{
		std::ofstream file{resultPath.c_str(), std::ios_base::app | std::ios_base::out};
		file << count << ";" << seed << ";" << config << ";" << minLen << ";" << maxLen << ";";
		file.close();
	}

}
