#include <algorithm>
#include <random>
#include <gtest/gtest.h>

#include "testData.h"
#include "masks.h"
#include "tree.h"


	TEST(SampleTest, KeysBasicSample)
	{
		int count = 1000;
		std::vector<unsigned> levels{ 8, 8, 8, 8 };

		//Creating some random data
		std::random_device rd{};
		std::mt19937 mt{ rd() };
		std::vector<unsigned> h_data(levels.size() * count);

		std::generate(h_data.begin(), h_data.end(), [&mt] { return mt() & 0xFF; });

		//Creating configuration
		Flavors::Configuration config{ levels };

		//Creating keys
		Flavors::Keys keys{ config, count, h_data.data() };

		//Building tree
		Flavors::Tree tree{ keys };

		//Finding keys
		Flavors::CudaArray<unsigned> result{ count };
		tree.FindKeys(keys, result.Get());

		auto h_results = result.ToHost();
	}

	TEST(SampleTest, MasksBasicSample)
	{
		int count = 1000;
		std::vector<unsigned> levels{ 8, 8, 8, 8 };

		//Creating some random data
		std::random_device rd{};
		std::mt19937 mt{ rd() };
		std::vector<unsigned> h_data(levels.size() * count);
		std::vector<unsigned> h_lengths(count);

		std::generate(h_data.begin(), h_data.end(), [&mt] { return mt() & 0xFF; });

		auto rand = std::bind(std::uniform_int_distribution<int>(10, 32), mt);
		std::generate(h_lengths.begin(), h_lengths.end(), rand);

		//Creating configuration
		Flavors::Configuration config{ levels };

		//Creating keys
		Flavors::Masks masks{ config, count, h_data.data(), h_lengths.data() };

		//Building tree
		Flavors::Tree tree{ masks };

		//Finding keys
		Flavors::CudaArray<unsigned> result{ count };
		tree.FindMasks(masks, result.Get());

		auto h_results = result.ToHost();
	}
