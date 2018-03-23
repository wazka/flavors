#include "dataInfo.h"
#include "utils.h"

namespace Flavors
{
	DataInfo::DataInfo() :
		N(0),
		Min(100000000.0f),
		Max(0.0f),
		Mean(0),
		M2(0),
		M3(0),
		M4(0)
	{
	}

	DataInfo::DataInfo(unsigned* nodesIndexes, int keysCount) :
		N(0),
		Min(std::numeric_limits<float>::max()),
		Max(std::numeric_limits<float>::min()),
		Mean(0),
		M2(0),
		M3(0),
		M4(0)
	{
		CudaArray<float> tmp{keysCount};
		CudaArray<unsigned> nodes{keysCount};
		CudaArray<unsigned> spans{keysCount};

		thrust::fill_n(thrust::device, tmp.Get(), keysCount, 1.0f);

		auto newEnd = thrust::reduce_by_key(
				thrust::device,
				nodesIndexes,
				nodesIndexes + keysCount,
				tmp.Get(),
				nodes.Get(),
				spans.Get());

		// compute summary statistics
		*this = thrust::transform_reduce(
				thrust::device,
				spans.Get(),
				newEnd.second,
				DataInfo{},
				DataInfo{},
				DataInfo{});
	}

	DataInfo DataInfo::operator()(const float& x) const
	{
		DataInfo result;
		result.N = 1;
		result.Min = x;
		result.Max = x;
		result.Mean = x;
		result.M2 = 0;
		result.M3 = 0;
		result.M4 = 0;

		return result;
	}

	DataInfo DataInfo::operator()(const DataInfo& x, const DataInfo& y) const
	{
		DataInfo result;

		// precompute some common subexpressions
		float n = x.N + y.N;
		float n2 = n * n;
		float n3 = n2 * n;

		float delta = y.Mean - x.Mean;
		float delta2 = delta * delta;
		float delta3 = delta2 * delta;
		float delta4 = delta3 * delta;

		//Basic number of samples (N), Min, and Max
		result.N = n;
		result.Min = x.Min < y.Min ? x.Min : y.Min;
		result.Max = x.Max > y.Max ? x.Max : y.Max;

		result.Mean = x.Mean + delta * y.N / n;

		result.M2 = x.M2 + y.M2;
		result.M2 += delta2 * x.N * y.N / n;

		result.M3 = x.M3 + y.M3;
		result.M3 += delta3 * x.N * y.N * (x.N - y.N) / n2;
		result.M3 += 3.0 * delta * (x.N * y.M2 - y.N * x.M2) / n;

		result.M4 = x.M4 + y.M4;
		result.M4 += delta4 * x.N * y.N * (x.N * x.N - x.N * y.N + y.N * y.N) / n3;
		result.M4 += 6.0 * delta2 * (x.N * x.N * y.M2 + y.N * y.N * x.M2) / n2;
		result.M4 += 4.0 * delta * (x.N * y.M3 - y.N * x.M3) / n;

		return result;
	}
}
