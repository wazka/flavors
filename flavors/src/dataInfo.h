#pragma once

#include <thrust/device_vector.h>
#include <thrust/extrema.h>

#include <cmath>

#include "../json/json.hpp"

namespace Flavors
{
	struct DataInfo
	{
		float N;
		float Min;
		float Max;
		float Mean;
		float M2;
		float M3;
		float M4;

		__host__ __device__ DataInfo();
		DataInfo(unsigned* nodesIndexes, int keysCount);

		float Variance() const { return M2 / (N - 1); }
		float VarianceN() const { return M2 / N; }
		float Skewness() const { return std::sqrt(N) * M3 / std::pow(M2, 1.5); }
		float Kurtosis() const { return N * M4 / (M2 * M2); }

		__host__ __device__ DataInfo operator()(const float& x) const;
		__host__ __device__ DataInfo operator()(const DataInfo& x, const DataInfo& y) const;
	};

	void to_json(nlohmann::json& j, const DataInfo& info);

}
