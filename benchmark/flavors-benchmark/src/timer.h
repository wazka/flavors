#include <chrono>

namespace FlavorsBenchmarks
{
	class Timer
	{
	public:
		void Start()
		{
			start = std::chrono::high_resolution_clock::now();
		}

		float Stop()
		{
			auto end = std::chrono::high_resolution_clock::now();
			auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>( end - start ).count();

			return duration;
		}

	private:
		std::chrono::high_resolution_clock::time_point start;

	};
}
