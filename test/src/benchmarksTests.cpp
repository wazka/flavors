#include <gtest/gtest.h>
#include <string>
#include <fstream>

#include "testData.h"
#include "keysFind.h"
#include "masksFind.h"
#include "multiConfigKeys.h"
#include "multiConfigMasks.h"

using namespace Flavors;
using namespace FlavorsBenchmarks;

namespace FlavorsTests
{
	class BenchmarkTest
	{
	public:
		bool CheckFileExists(const std::string& name)
		{
			std::ifstream f(name.c_str());
			return f.good();
		}

		void RemoveFile(const std::string& name)
		{
			remove(name.c_str());
		}
	};

	class FindBenchmarkTest : public BenchmarkTest, public ::testing::TestWithParam<std::tuple<int, int, Configuration>>
	{
	};

	class MultiConfigBenchmarkTest : public BenchmarkTest, public ::testing::TestWithParam<std::tuple<int, int>>
	{
	};

	TEST_P(FindBenchmarkTest, KeysFind)
	{
		//given
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);
		Configuration config = std::get<2>(params);

		KeysFindBenchmark bench{ count, seed, config, TestData::BenchmarkResultFile};

		//when
		bench.Run();

		//then
		ASSERT_TRUE(CheckFileExists(bench.ResultFullPath()));

		//cleanup
		RemoveFile(bench.ResultFullPath());
	}

	TEST_P(FindBenchmarkTest, MasksFind)
	{
		//given
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);
		Configuration config = std::get<2>(params);

		int maxLen = config.Length;
		int minLen = maxLen / 2;

		MasksFindBenchmark bench{count, seed, config, TestData::BenchmarkResultFile, minLen, maxLen};

		//when
		bench.Run();

		//then
		ASSERT_TRUE(CheckFileExists(bench.ResultFullPath()));

		//cleanup
		RemoveFile(bench.ResultFullPath());
	}

	INSTANTIATE_TEST_CASE_P(
			SmallData,
			FindBenchmarkTest,
			::testing::Combine(
				::testing::ValuesIn(TestData::SmallCounts),
				::testing::ValuesIn(TestData::Seeds),
				::testing::ValuesIn(TestData::Configs))
		);

	INSTANTIATE_TEST_CASE_P(
		BigData,
		FindBenchmarkTest,
		::testing::Combine(
			::testing::ValuesIn(TestData::BigCounts),
			::testing::ValuesIn(TestData::Seeds),
			::testing::ValuesIn(TestData::Configs))
	);

	TEST_P(MultiConfigBenchmarkTest, KeysFind)
	{
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);

		MultiConfigKeysBenchmark bench{count, seed, TestData::Configs, TestData::BenchmarkResultFile};

		//when
		bench.Run();

		//then
		ASSERT_TRUE(CheckFileExists(bench.ResultFullPath()));

		//cleanup
		RemoveFile(bench.ResultFullPath());
		system("rm *.json");	//TODO: Doing this more elegantly
	}

	TEST_P(MultiConfigBenchmarkTest, MasksFind)
	{
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);

		int maxLen = TestData::Configs[0].Length;
		int minLen = maxLen / 2;

		MultiConfigMasksBenchmark bench{count, seed, TestData::Configs, TestData::BenchmarkResultFile, minLen, maxLen};

		//when
		bench.Run();

		//then
		ASSERT_TRUE(CheckFileExists(bench.ResultFullPath()));

		//cleanup
		RemoveFile(bench.ResultFullPath());
		system("rm *.json");	//TODO: Doing this more elegantly
	}

	INSTANTIATE_TEST_CASE_P(
			SmallData,
			MultiConfigBenchmarkTest,
			::testing::Combine(
				::testing::ValuesIn(TestData::SmallCounts),
				::testing::ValuesIn(TestData::Seeds))
		);

	INSTANTIATE_TEST_CASE_P(
			BigData,
			MultiConfigBenchmarkTest,
			::testing::Combine(
				::testing::ValuesIn(TestData::SmallCounts),
				::testing::ValuesIn(TestData::Seeds))
		);



}
