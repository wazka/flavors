#include "dictionary.h"

#include <fstream>
#include <algorithm>
#include <string>
#include <cctype>

using namespace Flavors;

namespace FlavorsBenchmarks
{
	std::string DictionaryBenchmark::Label = "DictSourceRead;DictSourceWordCount;DictSourceMemory;DictSourceWordsSort;DictBuild;DictMemory;BookRead;BookWordCount;BookMemory;Find;BookSort;FindSorted;LevelsSizes;HitRate";


	std::vector<std::string> DictionaryBenchmark::readWords(std::string path, unsigned& maxWordLen)
	{
		std::vector<std::string> words;
		std::ifstream file;
		file.open(path);

		std::string word;
		maxWordLen = 0;

		if(file.good())
		{
			while(file >> word)
			{
				std::string result;

				std::transform(word.begin(), word.end(), word.begin(), ::tolower);
				std::remove_copy_if(word.begin(), word.end(), std::back_inserter(result), std::ptr_fun<int, int>(&std::ispunct));
				words.push_back(result);

				if(result.length() > maxWordLen)
					maxWordLen = result.length();
			}
		}

		return words;
	}

	Flavors::Configuration DictionaryBenchmark::prepareConfig(unsigned bitsPerLetter, unsigned maxWordLen)
	{
		return Benchmark::prepareConfig(bitsPerLetter, bitsPerLetter, bitsPerLetter * maxWordLen);

	}

	Flavors::Masks DictionaryBenchmark::wordsToMasks(std::vector<std::string>& words, Flavors::Configuration& config)
	{
		Masks masks{
			config,
			static_cast<int>(words.size())};

		std::vector<unsigned> rawWords(masks.Count * masks.Depth());
		std::vector<unsigned> rawLengths(masks.Count);

		for(int wordIndex = 0; wordIndex < masks.Count; ++wordIndex)
		{
			auto currentWord = words[wordIndex];

			rawLengths[wordIndex] = currentWord.length() * BitsPerLetter;

			for(int letterIndex = 0; letterIndex < masks.Depth(); ++letterIndex)
				rawWords[letterIndex * masks.Count + wordIndex] = currentWord[letterIndex];
		}

		cuda::memory::copy(masks.Lengths.Get(), rawLengths.data(), masks.Count * sizeof(unsigned));
		cuda::memory::copy(masks.Store.Get(), rawWords.data(), masks.Count * masks.Depth() * sizeof(unsigned));

		return masks;
	}

	Flavors::Masks DictionaryBenchmark::loadDictionary()
	{
		auto words = readWords(dictionaryPath, maxWordLen);
		auto config = prepareConfig(BitsPerLetter, maxWordLen);

		return wordsToMasks(words, config);
	}

	Flavors::Masks DictionaryBenchmark::loadBook(std::string& bookPath, Configuration& config)
	{
		unsigned maxBookWordLen = 0;
		auto bookWords = readWords(bookPath, maxBookWordLen);

		if(maxBookWordLen > maxWordLen)
			std::cerr << "WARNING: Longest book word is longer, then longest dictionary word. Longer words will be trimmed.";

		return wordsToMasks(bookWords, config);
	}

	void DictionaryBenchmark::Run()
	{
		timer.Start();
		auto dictSourceWords = loadDictionary();
		measured["DictSourceRead"] = timer.Stop();
		measured["DictSourceWordCount"] = dictSourceWords.Count;
		measured["DictSourceMemory"] = dictSourceWords.MemoryFootprint();

		timer.Start();
		dictSourceWords.Sort();
		measured["DictSourceWordsSort"] = timer.Stop();

		timer.Start();
		Tree dict{dictSourceWords};
		measured["DictBuild"] = timer.Stop();
		measured["DictMemory"] = dict.MemoryFootprint();

		for(auto bookPath : bookPaths)
		{
			std::cout << "\n\t Starting for book: " << bookPath << "... ";

			timer.Start();
			auto book = loadBook(bookPath, dictSourceWords.Config);
			measured["BookRead"] = timer.Stop();
			measured["BookWordCount"] = book.Count;
			measured["BookMemory"] = book.MemoryFootprint();

			CudaArray<unsigned> result{book.Count};

			timer.Start();
			dict.FindMasks(book, result.Get());
			measured["Find"] = timer.Stop();

			timer.Start();
			book.Sort();
			measured["BookSort"] = timer.Stop();

			timer.Start();
			dict.FindMasks(book, result.Get());
			measured["FindSorted"] = timer.Stop();

			measured.AppendToFile(ResultFullPath());
			Benchmark::recordStatistics(dict, result);

			std::cout << "finished";
		}

		std::cout << std::endl;
	}
}
