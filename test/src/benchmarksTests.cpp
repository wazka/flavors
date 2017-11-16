#include <gtest/gtest.h>
#include <string>
#include <fstream>

#include "testData.h"
#include "keysFind.h"
#include "masksFind.h"

using namespace Flavors;
using namespace FlavorsBenchmarks;



namespace FlavorsTests
{
	class BenchmarkTest : public ::testing::TestWithParam<std::tuple<int, int, Configuration>>
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

	TEST_P(BenchmarkTest, KeysFind)
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
		ASSERT_TRUE(CheckFileExists(TestData::BenchmarkResultFile));

		//cleanup
		RemoveFile(TestData::BenchmarkResultFile);
	}

	TEST_P(BenchmarkTest, MasksFind)
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
		ASSERT_TRUE(CheckFileExists(TestData::BenchmarkResultFile));

		//cleanup
		RemoveFile(TestData::BenchmarkResultFile);
	}

	INSTANTIATE_TEST_CASE_P(
			SmallData,
			BenchmarkTest,
			::testing::Combine(
				::testing::ValuesIn(TestData::SmallCounts),
				::testing::ValuesIn(TestData::Seeds),
				::testing::ValuesIn(TestData::Configs))
		);

	INSTANTIATE_TEST_CASE_P(
		BigData,
		BenchmarkTest,
		::testing::Combine(
			::testing::ValuesIn(TestData::BigCounts),
			::testing::ValuesIn(TestData::Seeds),
			::testing::ValuesIn(TestData::Configs))
	);



}
