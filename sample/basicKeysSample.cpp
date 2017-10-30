#include <tree.h>
#include <iostream>
#include <algorithm>
#include <random>

int main()
{
	std::cout << "Keys sample starting..." << std::endl;

	int count = 10000;
	std::vector<unsigned> levels{ 8, 8, 8, 8 };

	std::cout << "Creating some random keys..." << std::endl;
	std::random_device rd{};
	std::mt19937 mt{ rd() };
	std::vector<unsigned> h_data(levels.size() * count);

	std::generate(h_data.begin(), h_data.end(), [&mt] { return mt() & 0xFF; });

	std::cout << "Creating configuration..." << std::endl;
	Flavors::Configuration config{ levels };

	std::cout << "Creating Keys object to store data on the GPU..." << std::endl;
	Flavors::Keys keys{ config, count, h_data.data() };

	std::cout << "Building the tree..." << std::endl;
	Flavors::Tree tree{ keys };

	std::cout << "Finding keys in the tree..." << std::endl;
	Flavors::CudaArray<unsigned> result{ count };
	tree.FindKeys(keys, result.Get());

	auto h_results = result.ToHost();

	std::cout << "Sample finished..." << std::endl;
}
