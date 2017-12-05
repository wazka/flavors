#include "multiConfigKeys.h"

#include <fstream>
#include <sstream>

using namespace Flavors;

namespace Flavors
{
	void to_json(nlohmann::json& j, const DataInfo& info)
	{
		j["n"] = info.N;
		j["min"] = info.Min;
		j["max"] = info.Max;
		j["mean"] = info.Mean;
		j["variance"] = info.Variance();
		j["std"] = std::sqrt(info.VarianceN());
		j["skewness"] = info.Skewness();
		j["kurtosis"] = info.Kurtosis();
	}
}

namespace FlavorsBenchmarks
{
	std::string MultiConfigKeysBenchmark::Label = "Count;Seed;Config;Generation;Sort;Reshape;Build;Find;DataMemory;TreeMemory;LevelsSizes;HitRate";

	void MultiConfigKeysBenchmark::Run()
	{
		generateRawKeys();

		for(auto config : configs)
			runForConfig(config);

		saveDataInfo();
	}

	void MultiConfigKeysBenchmark::runForConfig(Configuration& config)
	{
		recordParameters(config);

		timer.Start();
		auto keys = rawKeys.ReshapeKeys(config);
		measured["Reshape"] = timer.Stop();
		measured["DataMemory"] = keys.MemoryFootprint();

		timer.Start();
		Tree tree{keys};
		measured["Build"] = timer.Stop();
		measured["TreeMemory"] = tree.MemoryFootprint();

		timer.Start();
		tree.FindKeys(keys, result.Get());
		measured["Find"] = timer.Stop();

		measured.AppendToFile(resultFullPath());
		recordStatistics(tree);
	}

	void MultiConfigKeysBenchmark::generateRawKeys()
	{
		timer.Start();
		rawKeys = Keys{Configuration::Default32, count};
		rawKeys.FillRandom(seed);
		measured["Generation"] = timer.Stop();

		timer.Start();
		rawKeys.Sort();
		measured["Sort"] = timer.Stop();
	}

	void MultiConfigKeysBenchmark::getDataInfo(nlohmann::json& j)
	{
		j["dataInfo"] = rawKeys.ReshapeKeys(Configuration::Binary32).GetInfo();
	}

	void MultiConfigKeysBenchmark::saveDataInfo()
	{
		std::ofstream recordDataFile;

		std::ostringstream fileName;
		fileName << resultPath << count << "_" << seed << ".json";

		recordDataFile.open(fileName.str());
		if (!recordDataFile.good())
			return;

		nlohmann::json j;

		getDataInfo(j);
		j["seed"] = seed;
		j["count"] = count;

		recordDataFile << j.dump(3);
		recordDataFile.close();
	}
}
