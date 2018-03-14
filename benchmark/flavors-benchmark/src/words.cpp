#include "words.h"

using namespace Flavors;

#ifdef __linux__
#include <sys/types.h>
#include <dirent.h>

void read_directory(const std::string& name, std::vector<std::string>& v)
{
   DIR* dirp = opendir(name.c_str());
   struct dirent * dp;
   while ((dp = readdir(dirp)) != NULL) {
       v.push_back(dp->d_name);
   }
   closedir(dirp);
}
#elif _WIN32
#include <filesystem>
#endif

namespace FlavorsBenchmarks
{
	void WordsBenchmark::Run()
	{
		try
		{
			cuda::device::current::set(deviceId);
		}
		catch (...)
		{
			std::cout << "\t\t ERROR: Wrong device ID" << std::endl;
			cuda::outstanding_error::clear();
			return;
		}

		measured.Add("deviceId", deviceId);
		deviceName = cuda::device::current::get().name();
		deviceName.erase(remove_if(deviceName.begin(), deviceName.end(), isspace), deviceName.end());
		measured.Add("deviceName", deviceName);

		#ifdef __linux__
		std::vector<std::string> paths;
		read_directory(dictionaries, paths);

		for (auto& path : paths)
		{
			std::string full_path = dictionaries + path;
			runForDictionary(full_path);
		}
		#elif _WIN32
		for (auto & p : std::experimental::filesystem::directory_iterator(dictionaries))
			runForDictionary(p.path().string());
		#endif		
	}

	void WordsBenchmark::runForDictionary(std::string& path)
	{
		std::cout << "Processing: " << path << std::endl;
		measured.Add("Dictionary", path);

		try
		{
			timer.Start();
			auto words = readWords(path);
			measured.Add("Load", timer.Stop());

			timer.Start();
			words.Sort();
			measured.Add("Sort", timer.Stop());

			std::vector<unsigned> outLevels{ 32, 32 };
			Configuration outConfig{ outLevels };
			auto outWords = words.ReshapeKeys(outConfig);

			auto outPath = "dat_" + path + ".dat";
			std::ofstream outFile(outPath);

			outFile << outWords;
			outFile.close();

			CudaArray<unsigned> result{ words.Count };

			for(auto& config : configs)
			{
				measured.Add("Config", config);

				timer.Start();
				auto reshapedWords = words.ReshapeKeys(config);
				measured.Add("Reshape", timer.Stop());

				timer.Start();
				Tree tree{ reshapedWords };
				measured.Add("Build", timer.Stop());
				measured.Add("TreeMemory", tree.MemoryFootprint());
				measured.Add("TreeLevels", tree);
				measured.Add("Depth", tree.Depth());

				timer.Start();
				tree.Find(reshapedWords, result.Get());
				measured.Add("Find", timer.Stop());
			
				measured.AppendToFile(resultFile);
			}
		}
		catch(...)
		{ 
			return; 
		}

	}

	std::vector<std::string> WordsBenchmark::loadWordsFromFile(std::string path)
	{
		std::vector<std::string> words;
		std::ifstream file;
		file.open(path);

		std::string word;

		if (file.good())
		{
			while (file >> word)
			{
				std::string result;

				std::transform(word.begin(), word.end(), word.begin(), ::tolower);
				std::remove_copy_if(word.begin(), word.end(), std::back_inserter(result), std::ptr_fun<int, int>(&std::ispunct));
				words.push_back(result);
			}
		}

		return words;
	}

	Flavors::Keys WordsBenchmark::readWords(std::string path)
	{
		auto rawWords = loadWordsFromFile(path);

		std::vector<unsigned> levels(rawWords[0].size(), BitsPerLetter);
		Configuration config{ levels };

		Keys keys{ config, static_cast<int>(rawWords.size()) };

		std::vector<unsigned> reorderedWords(keys.Count * keys.Depth());

		for(int i = 0; i < keys.Count; ++i)
		{
			auto currentWord = rawWords[i];
			for (int letter = 0; letter < keys.Depth(); ++letter)
				reorderedWords[letter * keys.Count + i] = currentWord[letter];
		}

		cuda::memory::copy(keys.Store.Get(), reorderedWords.data(), keys.Count * keys.Depth() * sizeof(unsigned));
		return keys;
	}
}
