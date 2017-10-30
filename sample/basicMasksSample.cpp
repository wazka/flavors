#include <tree.h>
#include <iostream>
#include <algorithm>
#include <random>

int main()
{
	std::cout << "Masks sample starting..." << std::endl;

	int count = 10000;
	std::vector<unsigned> levels{ 8, 8, 8, 8 };

	std::cout << "Creating some random masks..." << std::endl;
	std::random_device rd{};
	std::mt19937 mt{ rd() };
	std::vector<unsigned> h_data(levels.size() * count);
	std::vector<unsigned> h_lengths(count);

	std::generate(h_data.begin(), h_data.end(), [&mt] { return mt() & 0xFF; });

	int maxLen = 32;
	int minLen = 10;
	
	auto rand = std::bind(std::uniform_int_distribution<int>(minLen, maxLen), mt);
	std::generate(h_lengths.begin(), h_lengths.end(), rand);

	std::cout << "Creating configuration..." << std::endl;
	Flavors::Configuration config{ levels };

	std::cout << "Creating Masks object to store data on the GPU..." << std::endl;
	Flavors::Masks masks{ config, count, h_data.data(), h_lengths.data() };

	std::cout << "Building the tree..." << std::endl;
	Flavors::Tree tree{ masks };

	std::cout << "Finding masks in the tree..." << std::endl;
	Flavors::CudaArray<unsigned> result{ count };
	tree.FindMasks(masks, result.Get());

	auto h_results = result.ToHost();
	
	std::cout << "Sample finished..." << std::endl;
}
