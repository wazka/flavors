#include <gtest/gtest.h>
#include <algorithm>
#include "testData.h"

#define private public		//This hack should be removed - maybe replaced by HostTree?
#include "tree.h"

using namespace Flavors;

namespace FlavorsTests
{
	class TreeTest : public ::testing::TestWithParam<std::tuple<int, int, Configuration>>
	{
	public:
		static bool CmpKeys(std::vector<std::vector<unsigned>>& hostKeys, int lhs, int rhs)
		{
			return std::all_of(
				hostKeys.begin(),
				hostKeys.end(),
				[&lhs, &rhs](std::vector<unsigned>& level){ return level[lhs] == level[rhs]; });
		}

		static bool AreAllKeysInTree(Tree& tree, Keys& keys)
		{
			auto hostChildren = tree.Children.ToHost();
			auto hostKeys = keys.ToHost();
			auto hostPermutation = keys.Permutation.ToHost();

			for (int key = 0; key < keys.Count; ++key)
			{
				int currentNode = 1;
				for (int level = 0; level < tree.Depth(); ++level)
				{
					currentNode = hostChildren[level][(currentNode - 1) * tree.ChildrenCountsHost[level] + hostKeys[level][key]];
					if (currentNode == 0)
						return false;
				}

				auto retrivedKey = std::find(hostPermutation.begin(), hostPermutation.end(), currentNode - 1) - hostPermutation.begin();
				if (!CmpKeys(hostKeys, key, retrivedKey))
					return false;
			}

			return true;
		}

		static bool AreAllMasksInTree(Tree& tree, Masks& masks)
		{
			auto h_children = tree.Children.ToHost();
			auto h_masks = masks.ToHost();
			auto h_permutation = masks.Permutation.ToHost();
			auto h_lengths = masks.Lengths.ToHost();

			auto h_contStarts = tree.containers.Starts.ToHost();
			auto h_contLengths = tree.containers.Lengths.ToHost();
			auto h_contItems = tree.containers.Items.ToHost();

			auto h_masksParts = tree.masksParts.ToHost();
			auto h_treeLengths = tree.lengths.ToHost();

			for (int mask = 0; mask < masks.Count; ++mask)
			{
				int currentNode = 1;
				int depth = tree.Config[0];
				int level = 0;

				while(h_lengths[mask] > depth)
				{
					currentNode = h_children[level][(currentNode - 1) * tree.ChildrenCountsHost[level] + h_masks[level][mask]];
					
					++level;
					depth += tree.Config[level];

					if (currentNode == 0)
						return false;
				}

				int listItem = 0;
				while(listItem < h_contLengths[level][currentNode - 1])
				{
					auto itemValue = h_contItems[h_contStarts[level][currentNode - 1] + listItem];
					if (h_lengths[mask] == h_treeLengths[itemValue] && h_masksParts[itemValue] == h_masks[level][mask])
						break;

					++listItem;
				}

				if (listItem == h_contLengths[level][currentNode - 1])
					return false;

				auto itemValue = h_contItems[h_contStarts[level][currentNode - 1] + listItem];
				if (!CmpKeys(h_masks, mask, itemValue) || h_lengths[mask] != h_lengths[itemValue])
					return false;
			}

			return true;
		}

		static bool CheckKeysFindResult(CudaArray<unsigned>& result, Keys& keys)
		{
			auto h_result = result.ToHost();
			auto h_keys = keys.ToHost();
			auto h_permutation = keys.Permutation.ToHost();

			for(int key = 0; key < keys.Count; ++key)
			{
				if (h_permutation[key] != h_result[key] - 1)
				{
					auto retrivedKey = std::find(h_permutation.begin(), h_permutation.end(), h_result[key] - 1) - h_permutation.begin();
					if (!CmpKeys(h_keys, key, retrivedKey))
						return false;
				}
			}

			return true;
		}

		static bool CheckMasksFindResult(CudaArray<unsigned>& result, Masks& masks, Tree& tree)
		{
			auto h_result = result.ToHost();
			auto h_masks = masks.ToHost();
			auto h_permutation = masks.Permutation.ToHost();
			auto h_lengths = masks.Lengths.ToHost();
			
			for (int mask = 0; mask < masks.Count; ++mask)
			{
				if (h_permutation[mask] != h_result[mask] - 1)
				{
					auto retrivedMask = std::find(h_permutation.begin(), h_permutation.end(), h_result[mask] - 1) - h_permutation.begin();
					if (!CmpKeys(h_masks, mask, retrivedMask) || h_lengths[mask] != h_lengths[retrivedMask])
						return false;
				}
			}

			return true;
		}
	};

	TEST_P(TreeTest, BuildFromKeys)
	{
		//given
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);
		Configuration config = std::get<2>(params);

		Keys keys{ config, count };
		keys.FillRandom(seed);

		//when
		Tree tree{ keys };

		//then
		ASSERT_EQ(tree.Depth(), keys.Depth());
		ASSERT_EQ(tree.Config, keys.Config);
		ASSERT_EQ(tree.Count, keys.Count);
		ASSERT_TRUE(AreAllKeysInTree(tree, keys));
	}

	TEST_P(TreeTest, BuildFromMasks)
	{
		//given
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);
		Configuration config = std::get<2>(params);

		Masks masks{ config, count };
		masks.FillRandom(seed);

		//when
		Tree tree{ masks };

		//then
		ASSERT_EQ(tree.Depth(), masks.Depth());
		ASSERT_EQ(tree.Config, masks.Config);
		ASSERT_EQ(tree.Count, masks.Count);
		ASSERT_TRUE(AreAllMasksInTree(tree, masks));
	}

	TEST_P(TreeTest, BuildFromShortMasks)
	{
		//given
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);
		Configuration config = std::get<2>(params);

		Masks masks{ config, count };
		masks.FillRandom(seed, 24);

		//when
		Tree tree{ masks };

		//then
		ASSERT_LT(tree.Depth(), masks.Depth());
		ASSERT_EQ(tree.Count, masks.Count);
		ASSERT_TRUE(AreAllMasksInTree(tree, masks));
	}

	TEST_P(TreeTest, FindKeys)
	{
		//given
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);
		Configuration config = std::get<2>(params);

		Keys keys{ config, count };
		keys.FillRandom(seed);
		Tree tree{ keys };

		//when
		CudaArray<unsigned> result{ keys.Count };
		tree.FindKeys(keys, result.Get());

		//then
		ASSERT_TRUE(CheckKeysFindResult(result, keys));
	}

	TEST_P(TreeTest, FindMasks)
	{
		//given
		auto params = GetParam();
		int count = std::get<0>(params);
		int seed = std::get<1>(params);
		Configuration config = std::get<2>(params);

		Masks masks{ config, count };
		masks.FillRandom(seed);
		Tree tree{ masks };

		//when
		CudaArray<unsigned> result{ masks.Count };
		tree.FindMasks(masks, result.Get());

		//then
		ASSERT_TRUE(CheckMasksFindResult(result, masks, tree));
	}

	INSTANTIATE_TEST_CASE_P(
		SmallData,
		TreeTest,
		::testing::Combine(
			::testing::ValuesIn(TestData::SmallCounts),
			::testing::ValuesIn(TestData::Seeds),
			::testing::ValuesIn(TestData::Configs))
	);

	INSTANTIATE_TEST_CASE_P(
		BigData,
		TreeTest,
		::testing::Combine(
			::testing::ValuesIn(TestData::BigCounts),
			::testing::ValuesIn(TestData::Seeds),
			::testing::ValuesIn(TestData::Configs))
	);
}
