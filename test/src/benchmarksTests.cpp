#include <gtest/gtest.h>
#include <string>
#include <fstream>

#include "testData.h"
#include "keysFind.h"
#include "masksFind.h"
#include "multiConfigKeys.h"
#include "multiConfigMasks.h"
#include "keysLen.h"
#include "masksLen.h"

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

	class LenBenchmarkTest : public BenchmarkTest, public ::testing::TestWithParam<std::tuple<int, int, unsigned, unsigned, unsigned>>
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

		auto status = system("rm *.json");
		ASSERT_TRUE(WIFEXITED(status));
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

		auto status = system("rm *.json");
		ASSERT_TRUE(WIFEXITED(status));
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

	TEST_P(LenBenchmarkTest, KeysFind)
	{
		//given
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);
		unsigned depth = std::get<2>(params);
		unsigned firstLevelStride = std::get<3>(params);
		unsigned levelStride = std::get<4>(params);

		KeysLenBenchmark bench{
			count,
			seed,
			depth,
			firstLevelStride,
			levelStride,
			TestData::BenchmarkResultFile};

		//when
		bench.Run();

		//then
		ASSERT_TRUE(CheckFileExists(bench.ResultFullPath()));

		//cleanup
		RemoveFile(bench.ResultFullPath());
	}

	TEST_P(LenBenchmarkTest, MasksFind)
	{
		//given
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);
		unsigned depth = std::get<2>(params);
		unsigned firstLevelStride = std::get<3>(params);
		unsigned levelStride = std::get<4>(params);
		int max = depth;
		int min = 0.5 * max;

		MasksLenBenchmark bench{
			count,
			seed,
			depth,
			firstLevelStride,
			levelStride,
			max,
			min,
			TestData::BenchmarkResultFile};

		//when
		bench.Run();

		//then
		ASSERT_TRUE(CheckFileExists(bench.ResultFullPath()));

		//cleanup
		RemoveFile(bench.ResultFullPath());
	}

	INSTANTIATE_TEST_CASE_P(
		SmallData,
		LenBenchmarkTest,
		::testing::Combine(
			::testing::ValuesIn(TestData::SmallCounts),
			::testing::ValuesIn(TestData::Seeds),
			::testing::ValuesIn(TestData::Depths),
			::testing::ValuesIn(TestData::FirstLevelStrides),
			::testing::ValuesIn(TestData::LevelStrides))
	);

	INSTANTIATE_TEST_CASE_P(
		BigData,
		LenBenchmarkTest,
		::testing::Combine(
			::testing::ValuesIn(TestData::SmallCounts),
			::testing::ValuesIn(TestData::Seeds),
			::testing::ValuesIn(TestData::Depths),
			::testing::ValuesIn(TestData::FirstLevelStrides),
			::testing::ValuesIn(TestData::LevelStrides))
	);
}
