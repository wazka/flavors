#pragma once
#include <configuration.h>
#include <string>
#include <vector>

#include "benchmark.h"
#include "tree.h"

namespace FlavorsBenchmarks
{
	template<typename T>
	class RandomBenchmark
	{
	public:
		RandomBenchmark(
			std::vector<int> counts,
			std::vector<int> randomCounts,
			std::vector<int> seeds,
			std::vector<Flavors::Configuration>&& configs,
			std::string resultFile,
			std::string dataInfoDirectory,
			int deviceId,
			int maxMaskLength = 0,
			int minMaskLength = 0):
				counts(counts),
				randomCounts(randomCounts),
				seeds(seeds),
				configs(configs),
				resultFile(resultFile),
				dataInfoDirectory(dataInfoDirectory),
				caseIndex(1),
				dataItemLength(0),
				deviceId(deviceId),
				maxMaskLength(maxMaskLength),
				minMaskLength(minMaskLength)
		{
			caseCount = counts.size() * seeds.size();

			if(configs.size() > 0)
				dataItemLength = configs[0].Length;
		}

		RandomBenchmark(nlohmann::json& j, int maxMaskLength = 0, int minMaskLength = 0):
			RandomBenchmark::RandomBenchmark(
				tryReadFromJson<std::vector<int>>(j, "counts"),
				tryReadFromJson<std::vector<int>>(j, "randomCounts"),
				tryReadFromJson<std::vector<int>>(j, "seeds"),
				tryReadFromJson<std::vector<Flavors::Configuration>>(j, "configs"),
				tryReadFromJson<std::string>(j, "resultFile"),
				tryReadFromJson<std::string>(j, "dataInfoDirectory"),
				tryReadIntFromJson(j, "deviceId"),
				maxMaskLength,
				minMaskLength)
		{
		}

		void Run()
		{
			std::cout << "Starting benchmark. Results will be saved to: \t" << resultFile << std::endl;

			try
			{
				cuda::device::current::set(deviceId);
			}
			catch(...)
			{
				std::cout << "\t\t ERROR: Wrong device ID" << std::endl;
				cuda::outstanding_error::clear();
				return;
			}

			measured.Add("deviceId", deviceId);
			deviceName = cuda::device::current::get().name();
			deviceName.erase(remove_if(deviceName.begin(), deviceName.end(), isspace), deviceName.end());
			measured.Add("deviceName", deviceName);

			for(int count : counts)
				for(int seed : seeds)
					runCase(count, seed);
		}

	private:
		Measured measured;
		Timer timer;

		int deviceId;
		std::string deviceName;

		T prepareRawData(int count, int seed)
		{
			measured.Add("Count", count);
			measured.Add("Seed", seed);
			measured.Add("DataItemLenght", dataItemLength);

			if(maxMaskLength != 0)
			{
				measured.Add("MaxMaskLength", maxMaskLength);
				measured.Add("MinMaskLength", minMaskLength);
			}

			timer.Start();
			T rawData{Flavors::Configuration::Default(dataItemLength), count};
			fillRandom(rawData, seed);
			measured.Add("Generation", timer.Stop());

			timer.Start();
			rawData.Sort();
			measured.Add("Sort", timer.Stop());

			saveDataInfo(
					rawData,
					count,
					seed,
					dataItemLength,
					deviceId,
					deviceName,
					dataInfoDirectory,
					maxMaskLength,
					minMaskLength);

			return rawData;
		}

		void runCase(int count, int seed)
		{
			std::cout << "\t Starting case " << caseIndex << " / " << caseCount << " for count = " << count << ", seed = " << seed << std::endl;

			try
			{
				auto rawData = prepareRawData(count, seed);

				for(auto config : configs)
				{
					measured.Add("Config", config);
					runCaseForConfig(rawData, config, seed);
				}
			}
			catch(...)
			{
				std::cout << "\t\t ERROR: Setting up case failed due to exception" << std::endl;
				cuda::outstanding_error::clear();
			}

			++caseIndex;
		}

		Flavors::Keys reshapeData(Flavors::Keys& rawKeys, Flavors::Configuration& newConfig)
		{
			return rawKeys.ReshapeKeys(newConfig);
		}

		Flavors::Masks reshapeData(Flavors::Masks& rawMasks, Flavors::Configuration& newConfig)
		{
			return rawMasks.ReshapeMasks(newConfig);
		}

		void fillRandom(Flavors::Keys& keys, unsigned seed)
		{
			keys.FillRandom(seed + 1);
		}

		void fillRandom(Flavors::Masks& masks, unsigned seed)
		{
			masks.FillRandom(seed + 1, maxMaskLength, minMaskLength);
		}

		void runCaseForConfig(T& rawData, Flavors::Configuration& config, int seed)
		{
			std::cout << "\t\t Starting case for config = " << config << std::endl;

			try
			{
				Flavors::CudaArray<unsigned> result {rawData.Count};

				timer.Start();
				auto data = reshapeData(rawData, config);
				measured.Add("Reshape", timer.Stop());
				measured.Add("DataMemory", data.MemoryFootprint());

				timer.Start();
				Flavors::Tree tree{data};
				measured.Add("Build", timer.Stop());
				measured.Add("TreeMemory", tree.MemoryFootprint());
				measured.Add("TreeLevels", tree);
				measured.Add("Depth", tree.Depth());

				timer.Start();
				tree.Find(data, result.Get());
				measured.Add("Find", timer.Stop());

				for(int randomCount : randomCounts)
				{
					std::cout << "\t\t\t Starting random find for random count = " << randomCount << std::endl;
					Flavors::CudaArray<unsigned> randomResult {randomCount};

					try
					{
						T randomRawData{rawData.Config, randomCount};
						fillRandom(randomRawData, seed);

						measured.Add("RandomCount", randomCount);

						auto randomData = reshapeData(randomRawData, config);

						timer.Start();
						tree.Find(randomData, randomResult.Get());

						measured.Add("FindRandom", timer.Stop());

						timer.Start();
						randomRawData.Sort();
						measured.Add("RandomSort", timer.Stop());

						randomData = reshapeData(randomRawData, config);

						timer.Start();
						tree.Find(randomData, randomResult.Get());
						measured.Add("FindRandomSorted", timer.Stop());
						measured.AddHitCount(result);

						measured.AppendToFile(resultFile);
					}
					catch(...)
					{
						std::cout << "\t\t\t\t ERROR: Random find failed due to exception" << std::endl;
						cuda::outstanding_error::clear();
					}
				}

				//Special case of this benchmark, where there is no random finds
				if(randomCounts.size() == 0)
					measured.AppendToFile(resultFile);
			}
			catch(...)
			{
				std::cout << "\t\t\t ERROR: Running case failed due to exception" << std::endl;
				cuda::outstanding_error::clear();
			}
		}

		int caseIndex;
		int caseCount;

		std::vector<int> counts;
		std::vector<int> randomCounts;
		std::vector<int> seeds;
		std::vector<Flavors::Configuration> configs;
		int dataItemLength;

		int maxMaskLength;
		int minMaskLength;

		std::string resultFile;
		std::string dataInfoDirectory;
	};

	template<typename T>
	class LengthBenchmark
	{
	public:
		LengthBenchmark(
			std::vector<int>&& counts,
			std::vector<int>&& randomCounts,
			std::vector<int>&& seeds,
			std::vector<unsigned>&& keyLengths,
			std::vector<unsigned>&& firstLevelStrides,
			std::vector<unsigned>&& levelStrides,
			std::string&& resultFile,
			std::string&& dataInfoDirectory,
			float maxMaskLength = 0.0,
			float minMaskLength = 0.0) :
				counts(counts),
				randomCounts(randomCounts),
				seeds(seeds),
				keyLengths(keyLengths),
				firstLevelStrides(firstLevelStrides),
				levelStrides(levelStrides),
				resultFile(resultFile),
				dataInfoDirectory(dataInfoDirectory),
				maxMaskLength(maxMaskLength),
				minMaskLength(minMaskLength)
		{
		}

		LengthBenchmark(nlohmann::json& j, float maxMaskLength = 0.0, float minMaskLength = 0.0) :
			LengthBenchmark::LengthBenchmark(
				tryReadFromJson<std::vector<int>>(j, "counts"),
				tryReadFromJson<std::vector<int>>(j, "randomCounts"),
				tryReadFromJson<std::vector<int>>(j, "seeds"),
				tryReadFromJson<std::vector<unsigned>>(j, "keyLenghts"),
				tryReadFromJson<std::vector<unsigned>>(j, "firstLevelStrides"),
				tryReadFromJson<std::vector<unsigned>>(j, "levelStrides"),
				tryReadFromJson<std::string>(j, "resultFile"),
				tryReadFromJson<std::string>(j, "dataInfoDirectory"),
				maxMaskLength,
				minMaskLength)
		{
		}

		void Run()
		{
			for(auto keyLength : keyLengths)
				runForDepth(keyLength);
		}

	private:
		std::vector<int> counts;
		std::vector<int> randomCounts;
		std::vector<int> seeds;

		std::vector<unsigned> keyLengths;
		std::vector<unsigned> firstLevelStrides;
		std::vector<unsigned> levelStrides;

		float maxMaskLength;
		float minMaskLength;

		std::string resultFile;
		std::string dataInfoDirectory;

		std::vector<Flavors::Configuration> prepareConfigs(unsigned keyLength)
		{
			std::vector<Flavors::Configuration> configs;

			for(auto firstLevelStride : firstLevelStrides)
				for(auto levelStride : levelStrides)
				{
					std::vector<unsigned> config{firstLevelStride};

					auto currentDepth = firstLevelStride;
					while(currentDepth + levelStride <= keyLength)
					{
						config.push_back(levelStride);
						currentDepth += levelStride;
					}

					if(currentDepth < keyLength)
						config.push_back(keyLength - currentDepth);

					configs.push_back(Flavors::Configuration{config});
				}

			return configs;
		}

		void runForDepth(unsigned depth)
		{
			std::cout << "Starting keys benchmark for key length = " << depth << std::endl;

			int currentMaxLength = static_cast<float>(depth) * maxMaskLength;
			int currentMinLength = static_cast<float>(depth) * minMaskLength;

			RandomBenchmark<T> bench{
				counts,
				randomCounts,
				seeds,
				prepareConfigs(depth),
				resultFile,
				dataInfoDirectory,
				currentMaxLength,
				currentMinLength};

			bench.Run();
		}
	};

}
