#include "ip.h"
#include <sstream>

using namespace Flavors;

namespace FlavorsBenchmarks
{
    void parseLine(const std::string& line, std::vector<std::string>& tmpMasks, std::vector<unsigned>& lenghts)
	{
		std::vector<std::string> tokens;
		std::string token;
		std::stringstream streamLine{ line };

		while (getline(streamLine, token, ';'))
			tokens.push_back(token);

		tmpMasks.push_back(tokens[2]);
		try 
        { 
            lenghts.push_back(stoul(tokens[3])); 
        }
		catch (...) 
        { 
            //TODO: Exception 
        }
	}

    unsigned parseTmpMask(std::string& tmpMask)
    {
        unsigned mask = 0;
        std::vector<std::string> tokens;
		std::string token;
		std::stringstream streamLine{ tmpMask };

		while (getline(streamLine, token, '.'))
			tokens.push_back(token);

		mask |= stoul(tokens[0]);
		mask |= stoul(tokens[1]) << 8;
		mask |= stoul(tokens[2]) << 16;
		mask |= stoul(tokens[3]) << 24;

        return mask;
    }

    Flavors::Masks IpBenchmark::loadIpSet(std::string& path)
    {
        std::vector<std::string> lines;
		std::ifstream file;
		file.open(path);

		std::string line;

        std::vector<std::string> tmpMasks;
        std::vector<unsigned> lenghts;

		if (file.good())
		{
			while (file >> line)
                parseLine(line, tmpMasks, lenghts);
		}

        std::vector<unsigned> reorderedMasks(tmpMasks.size());
        std::transform(tmpMasks.begin(), tmpMasks.end(), reorderedMasks.begin(), parseTmpMask);

        Flavors::Masks masks(Flavors::Configuration::Default32, tmpMasks.size());

        cuda::memory::copy(masks.Lengths.Get(), lenghts.data(), masks.Count * sizeof(unsigned));
        cuda::memory::copy(masks.Store.Get(), reorderedMasks.data(), masks.Count * sizeof(unsigned));

        return masks;
    }

    void IpBenchmark::runForDictionary(std::string& path)
	{
		std::cout << "Processing ip set: " << path << std::endl;
		measured.Add("Dictionary", path);

		try
		{
			timer.Start();
			auto ipSet = loadIpSet(path);
			measured.Add("Load", timer.Stop());
			measured.Add("Count", ipSet.Count);

			timer.Start();
			ipSet.Sort();
			measured.Add("Sort", timer.Stop());

			CudaArray<unsigned> result{ ipSet.Count };

			for(auto& config : configs)
			{
				measured.Add("Config", config);

				timer.Start();
				auto reshapedIpSet = ipSet.ReshapeMasks(config);
				measured.Add("Reshape", timer.Stop());

				timer.Start();
				Tree tree{ reshapedIpSet };
				measured.Add("Build", timer.Stop());
				measured.Add("TreeMemory", tree.MemoryFootprint());
				measured.Add("TreeLevels", tree);
				measured.Add("Depth", tree.Depth());

				timer.Start();
				tree.Find(reshapedIpSet, result.Get());
				measured.Add("Find", timer.Stop());

                timer.Start();
                tree.Match(reshapedIpSet, result.Get());
                measured.Add("Match", timer.Stop());

				int randomCount = 1000000;
                Keys randomIpSet(config, randomCount);
                randomIpSet.FillRandom(0);
				measured.Add("RandomCount", randomCount);

                timer.Start();
                tree.Match(randomIpSet, result.Get());
                measured.Add("RandomMatch", timer.Stop());

                timer.Start();
                randomIpSet.Sort();
                measured.Add("RandomSort", timer.Stop());

                result.Clear();
                timer.Start();
                tree.Match(randomIpSet, result.Get());
                measured.Add("RandomSortedMatch", timer.Stop());
                measured.AddHitCount(result);
            
			    measured.AppendToFile(resultFile);
            }
        }
		catch(...)
		{ 
			return; 
		}
	}

}