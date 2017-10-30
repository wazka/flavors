#include <tree.h>
#include <iostream>

int main()
{
	Flavors::Configuration config{ std::vector<unsigned>{8, 8, 8, 8}};
	int count = 10000;
	int seed = 1234;

	Flavors::Keys keys{ config, count};
	keys.FillRandom(seed);

	Flavors::Tree tree{keys};
	

	std::cout << tree.Depth() << std::endl;
}
