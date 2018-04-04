#include <algorithm>
#include <random>
#include <gtest/gtest.h>

#include "testData.h"
#include "masks.h"
#include "keys.h"

using namespace Flavors;

namespace FlavorsTests
{
	class DataTest : public ::testing::TestWithParam<std::tuple<int, int, Configuration>>
	{
	public:
		static Configuration UniqueConfig32;

		static bool CheckAgainstConfig(Keys& keys, const Configuration& config)
		{
			auto hostKeys = keys.ToHost();

			for (int level = 0; level < keys.Depth(); ++level)
			{
				unsigned mask = config.Mask(level);

				if (std::any_of(
					hostKeys[level].begin(),
					hostKeys[level].end(),
					[&mask](unsigned value) { return value > mask; })
					)
					return false;

				if (std::all_of(
					hostKeys[level].begin(),
					hostKeys[level].end(),
					[](unsigned value) { return value == 0; })
					)
					return false;
			}

			return true;
		}

		static bool CheckLengths(Masks& masks, unsigned min, unsigned max)
		{
			auto hostLengths = masks.Lengths.ToHost();
			return std::all_of(
				hostLengths.begin(),
				hostLengths.end(),
				[&min, &max](unsigned length) { return length >= min && length <= max; });
		}

		static bool CheckAgainstConfig(Masks& masks, const Configuration& config)
		{
			Keys& keys = masks;
			return CheckAgainstConfig(keys, config) && CheckLengths(masks, 1, config.Length);
		}

		static bool CheckSort(Keys& keys)
		{
			auto h_keys = keys.ReshapeKeys(Configuration::Default32).ToHost();
			return std::is_sorted(h_keys.begin()->begin(), h_keys.begin()->end());
		}

		static bool CheckSort(Masks& masks)
		{
			auto h_masks = masks.ReshapeMasks(Configuration::Default32).ToHost();
			auto h_lenghts = masks.Lengths.ToHost();

			for (int mask = 1; mask < masks.Count; ++mask)
			{
				if (h_masks[0][mask - 1] > h_masks[0][mask])
					return false;
				if (h_masks[0][mask - 1] == h_masks[0][mask])
					if (h_lenghts[mask - 1] > h_lenghts[mask])
						return false;
			}

			return true;
		}
	};

	Configuration DataTest::UniqueConfig32{ std::vector <unsigned>{5, 5, 3, 7, 2, 3, 7} };

	TEST_P(DataTest, RandomKeys)
	{
		//given
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);
		Configuration config = std::get<2>(params);

		//when
		Keys keys{ config, count };
		keys.FillRandom(seed);

		//then
		ASSERT_EQ(count, keys.Count);
		ASSERT_EQ(config.Depth(), keys.Depth());
		ASSERT_FALSE(keys.Sorted());
		ASSERT_TRUE(CheckAgainstConfig(keys, config));
	}

	TEST_P(DataTest, RandomMasks)
	{
		//given
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);
		Configuration config = std::get<2>(params);

		//when
		Masks masks{ config, count };
		masks.FillRandom(seed);

		//then
		ASSERT_EQ(count, masks.Count);
		ASSERT_EQ(config.Depth(), masks.Depth());
		ASSERT_FALSE(masks.Sorted());
		ASSERT_TRUE(CheckAgainstConfig(masks, config));
	}

	TEST_P(DataTest, ReshapeKeys)
	{
		//given
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);
		Configuration config = std::get<2>(params);
		Keys keys{ config, count };
		keys.FillRandom(seed);

		//when
		Keys newKeys = keys.ReshapeKeys(UniqueConfig32);

		//then
		ASSERT_EQ(count, newKeys.Count);
		ASSERT_EQ(UniqueConfig32.Depth(), newKeys.Depth());
		ASSERT_TRUE(CheckAgainstConfig(newKeys, UniqueConfig32));
		ASSERT_EQ(keys.ReshapeKeys(Configuration::Default32), newKeys.ReshapeKeys(Configuration::Default32));
	}

	TEST_P(DataTest, ReshapeMasks)
	{
		//given
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);
		Configuration config = std::get<2>(params);
		Masks masks{ config, count };
		masks.FillRandom(seed);

		//when
		Masks newMasks = masks.ReshapeMasks(UniqueConfig32);

		//then
		ASSERT_EQ(count, newMasks.Count);
		ASSERT_EQ(UniqueConfig32.Depth(), newMasks.Depth());
		ASSERT_TRUE(CheckAgainstConfig(newMasks, UniqueConfig32));
		ASSERT_EQ(masks.ReshapeKeys(Configuration::Default32), newMasks.ReshapeKeys(Configuration::Default32));
	}

	TEST_P(DataTest, SortKeys)
	{
		//given
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);
		Configuration config = std::get<2>(params);
		Keys keys{ config, count };
		keys.FillRandom(seed);

		//when
		keys.Sort();

		//then
		ASSERT_EQ(count, keys.Count);
		ASSERT_EQ(config.Depth(), keys.Depth());
		ASSERT_TRUE(CheckAgainstConfig(keys, config));
		ASSERT_TRUE(keys.Sorted());
		ASSERT_TRUE(CheckSort(keys));
	}

	TEST_P(DataTest, SortMasks)
	{
		//given
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);
		Configuration config = std::get<2>(params);
		Masks masks{ config, count };
		masks.FillRandom(seed);

		//when
		masks.Sort();

		//then
		ASSERT_EQ(count, masks.Count);
		ASSERT_EQ(config.Depth(), masks.Depth());
		ASSERT_TRUE(CheckAgainstConfig(masks, config));
		ASSERT_TRUE(masks.Sorted());
		ASSERT_TRUE(CheckSort(masks));
	}

	TEST_P(DataTest, ReshapeSorted)
	{
		//given
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);
		Configuration config = std::get<2>(params);
		Keys keys{ config, count };
		keys.FillRandom(seed);

		//when
		keys.Sort();
		Keys newKeys = keys.ReshapeKeys(UniqueConfig32);

		//then
		ASSERT_TRUE(newKeys.Sorted());
		ASSERT_TRUE(CheckSort(newKeys));
	}

	INSTANTIATE_TEST_CASE_P(
		SmallData,
		DataTest,
		::testing::Combine(
			::testing::ValuesIn(TestData::SmallCounts),
			::testing::ValuesIn(TestData::Seeds),
			::testing::ValuesIn(TestData::Configs))
	);

	INSTANTIATE_TEST_CASE_P(
		BigData,
		DataTest,
		::testing::Combine(
			::testing::ValuesIn(TestData::BigCounts),
			::testing::ValuesIn(TestData::Seeds),
			::testing::ValuesIn(TestData::Configs))
	);
}
