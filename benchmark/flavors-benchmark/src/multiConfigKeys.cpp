#include "multiConfigKeys.h"

using namespace Flavors;

namespace FlavorsBenchmarks
{
	std::string MultiConfigKeysBenchmark::Label = "Count;Seed;Config;Generation;Sort;Reshape;Build;Find;LevelsSizes;HitRate";

	void MultiConfigKeysBenchmark::Run()
	{
		generateRawKeys();

		for(auto config : configs)
			runForConfig(config);
	}

	void MultiConfigKeysBenchmark::runForConfig(Configuration& config)
	{
		recordParameters(config);

		timer.Start();
		auto keys = rawKeys.ReshapeKeys(config);
		measured.Reshape = timer.Stop();

		timer.Start();
		Tree tree{keys};
		measured.Build = timer.Stop();

		timer.Start();
		tree.FindKeys(keys, result.Get());
		measured.Find = timer.Stop();

		measured.appendToFile(resultPath);
		recordStatistics(tree);
	}

	void MultiConfigKeysBenchmark::generateRawKeys()
	{
		timer.Start();
		rawKeys = Keys{Configuration::DefaultConfig32, count};
		rawKeys.FillRandom(seed);
		measured.Generation = timer.Stop();

		timer.Start();
		rawKeys.Sort();
		measured.Sort = timer.Stop();
	}
}
