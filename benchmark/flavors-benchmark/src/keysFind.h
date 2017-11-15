#include <configuration.h>
#include <string>

namespace FlavorsBenchmarks
{
	class KeysFindBenchmark
	{
		struct Result
		{
			float Generation;
			float Sort;
			float Reshape;
			float Build;
			float Find;
			float FindRandom;
			float FindRandomSorted;

			void appendToFile(std::string& path);
		};

	public:
		KeysFindBenchmark(int count, int seed, const Flavors::Configuration& config, const std::string& resultPath) :
			count(count),
			seed(seed),
			config(config),
			resultPath(resultPath)
		{}

		void Run();

	private:
		int count;
		int seed;
		std::string resultPath;
		Flavors::Configuration config;

		void recordParams();

		Result result;
	};

}
