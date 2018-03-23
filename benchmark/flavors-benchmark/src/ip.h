#pragma once

#include "benchmark.h"
#include "words.h"

namespace FlavorsBenchmarks
{
    class IpBenchmark : public WordsBenchmark
    {
    public:

        IpBenchmark(nlohmann::json& j):
			WordsBenchmark(j)
		{
		}

    protected:
        void runForDictionary(std::string& path) override;

    private:
        Flavors::Masks loadIpSet(std::string& path);

    };
}