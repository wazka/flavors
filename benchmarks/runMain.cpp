#include "../../lib/json/json.hpp"

#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include "configuration.h"
#include "randomBenchmark.h"
#include "dictionary.h"
#include "hostBenchmark.h"
#include "words.h"
#include "ip.h"

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		std::cerr << "Wrong arguments. Provide only a path to a json config file." << std::endl;
		return EXIT_FAILURE;
	}

	std::ifstream file(argv[1]);

	if (!file)
	{
		std::cerr << "Unable to open a file" << std::endl;
		return EXIT_FAILURE;
	}

	nlohmann::json j;
	file >> j;

	file.close();

	if (j["benchmark"] == "keysBenchmark")
	{
		FlavorsBenchmarks::RandomBenchmark<Flavors::Keys, Flavors::Tree> bench{ j };
		bench.Run();
	}
	else if(j["benchmark"] == "keysCompressedTreeBenchmark")
	{
		FlavorsBenchmarks::RandomBenchmark<Flavors::Keys, Flavors::CompressedTree> bench{ j };
		bench.Run();
	}
	else if (j["benchmark"] == "masksBenchmark")
	{
		int maxMaskLength =
			FlavorsBenchmarks::tryReadIntFromJson(j, "maxMaskLength");
		int minMaskLength =
			FlavorsBenchmarks::tryReadIntFromJson(j, "minMaskLength");

		FlavorsBenchmarks::RandomBenchmark<Flavors::Masks, Flavors::Tree> bench{
			j,
			maxMaskLength,
			minMaskLength };

		bench.Run();
	}
	else if (j["benchmark"] == "keysLenBenchmark")
	{
		FlavorsBenchmarks::LengthBenchmark<Flavors::Keys> bench{ j };
		bench.Run();
	}
	else if (j["benchmark"] == "masksLenBenchmark")
	{
		float maxMaskLength =
			FlavorsBenchmarks::tryReadFloatFromJson(j, "maxMaskLength");
		float minMaskLength =
			FlavorsBenchmarks::tryReadFloatFromJson(j, "minMaskLength");

		FlavorsBenchmarks::LengthBenchmark<Flavors::Masks> bench{
			j,
			maxMaskLength,
			minMaskLength };

		bench.Run();
	}
	else if (j["benchmark"] == "dictionary")
	{
		FlavorsBenchmarks::DictionaryBenchmark bench{ j };
		bench.Run();
	}
	else if (j["benchmark"] == "host")
	{
		FlavorsBenchmarks::HostBenchmark bench{ j };
		bench.Run();
	}
	else if (j["benchmark"] == "words")
	{
		FlavorsBenchmarks::WordsBenchmark bench{ j };
		bench.Run();
	}
	else if(j["benchmark"] == "ip")
	{
		FlavorsBenchmarks::IpBenchmark bench{ j };
		bench.Run();
	}
	else
	{
		std::cerr << "Unknown benchmark type." << std::endl;
		return EXIT_FAILURE;
	}
}
