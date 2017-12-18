#include "benchmark.h"

#include <vector>
#include <string>
#include <map>
#include <random>
#include <algorithm>

namespace FlavorsBenchmarks
{
	class HostBenchmark
	{
	public:
		HostBenchmark(
			std::vector<int> counts,
			std::vector<int> randomCounts,
			std::vector<int> seeds,
			std::string resultFile,
			std::string deviceName) :
				counts(counts),
				randomCounts(randomCounts),
				seeds(seeds),
				resultFile(resultFile),
				deviceName(deviceName),
				caseIndex(1)
		{
			caseCount = counts.size() * seeds.size();
		}

		HostBenchmark(nlohmann::json& j) :
			HostBenchmark(
				tryReadFromJson<std::vector<int>>(j, "counts"),
				tryReadFromJson<std::vector<int>>(j, "randomCounts"),
				tryReadFromJson<std::vector<int>>(j, "seeds"),
				tryReadFromJson<std::string>(j, "resultFile"),
				tryReadFromJson<std::string>(j, "deviceName"))
		{
		}

		void Run()
		{
			std::cout << "Starting benchmark. Results will be saved to: \t" << resultFile << std::endl;

			deviceName.erase(remove_if(deviceName.begin(), deviceName.end(), isspace), deviceName.end());
			measured.Add("deviceName", deviceName);

			for(int count : counts)
				for(int seed : seeds)
					runCase(count, seed);
		}

	private:
		std::vector<int> counts;
		std::vector<int> randomCounts;
		std::vector<int> seeds;

		std::string resultFile;

		std::string deviceName;

		Measured measured;
		Timer timer;

		int caseIndex;
		int caseCount;

		void runCase(int count, int seed)
		{
			std::cout << "\t Starting case " << caseIndex << " / " << caseCount << " for count = " << count << ", seed = " << seed << std::endl;

			try
			{
				auto data = prepareData(count, seed);
				std::map<unsigned, int> dict;

				timer.Start();
				int index = 0;
				for(auto item : data)
					dict[item] = index++;
				measured.Add("Build", timer.Stop());

				timer.Start();
				for(auto item : data)
					index = dict[item];
				measured.Add("Find", timer.Stop());

				for(int randomCount : randomCounts)
				{
					std::cout << "\t\t\t Starting random find for random count = " << randomCount << std::endl;
					measured.Add("RandomCount", randomCount);

					auto randomData = prepareData(randomCount, seed);

					timer.Start();
					for(auto item : randomData)
						if(dict.find(item) != dict.end())
							index = dict[item];
					measured.Add("RandomFind", timer.Stop());

					measured.AppendToFile(resultFile);
				}
			}
			catch(...)
			{
				std::cout << "\t\t ERROR: Case failed due to exception" << std::endl;
				cuda::outstanding_error::clear();
			}

			++caseIndex;
		}

		std::vector<unsigned> prepareData(int count, int seed)
		{
			measured.Add("Count", count);
			measured.Add("Seed", seed);
			measured.Add("DataItemLenght", 32);

			std::mt19937 mt(seed);
			std::vector<unsigned> randomValues(count);
			std::generate(randomValues.begin(), randomValues.end(), mt);

			return randomValues;
		}
	};

}
