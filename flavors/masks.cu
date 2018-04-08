#include "masks.h"

#include <random>
#include <iostream>
#include <algorithm>
#include <device_launch_parameters.h>
#include <thrust/execution_policy.h>
#include <thrust/sort.h>
#include <thrust/gather.h>

namespace Flavors
{
	Masks::Masks(const Configuration& config, int count) :
		Keys(config, count),
		Lengths(Count)
	{
	}

	__global__ void clip(int count, int depth, unsigned* levels, unsigned* lengths, unsigned** masks)
	{
		int mask = blockIdx.x * blockDim.x + threadIdx.x;
		if (mask >= count)
			return;

		unsigned length = lengths[mask];
		unsigned maskBit = 0;

		for (int level = 0; level < depth; ++level)
		{
			for (int bit = levels[level] - 1; bit >= 0; --bit, ++maskBit)
			{
				if (maskBit >= length)
					masks[level][mask] = masks[level][mask] & ~(1 << bit);
			}
		}
	}

	Masks::Masks(const Configuration& config, int count, unsigned* data, unsigned* lengths) :
		Keys(config, count, data),
		Lengths(Count)
	{
		cuda::memory::copy(Lengths.Get(), lengths, count * sizeof(unsigned));

		auto kernelConfig = make_launch_config(Count);

		cuda::launch(
			clip,
			kernelConfig,
			Count,
			Depth(),
			Config.Get(),
			Lengths.Get(),
			Store.GetLevels());
	}

	void Masks::FillRandom(int seed)
	{
		FillRandom(seed, Config.Length);
	}

	void Masks::FillRandom(int seed, int max, int min)
	{
		Keys::FillRandom(seed);

		std::vector<unsigned> randomLengths(Count);

		auto rand = std::bind(
			std::uniform_int_distribution<int>(min, max),
			std::mt19937(seed));

		std::generate(
			randomLengths.begin(),
			randomLengths.end(),
			rand);

		cuda::memory::copy(
			Lengths.Get(),
			randomLengths.data(),
			Count * sizeof(unsigned));

		auto kernelConfig = make_launch_config(Count);

		cuda::launch(
			clip,
			kernelConfig,
			Count,
			Depth(),
			Config.Get(),
			Lengths.Get(),
			Store.GetLevels());
	}

	void Masks::FillFromVector(std::vector<unsigned> source, std::vector<unsigned> lengths)
	{
		Keys::FillFromVector(source);

		cuda::memory::copy(Lengths.Get(), lengths.data(), Count * sizeof(unsigned));

		auto kernelConfig = make_launch_config(Count);

		cuda::launch(
			clip,
			kernelConfig,
			Count,
			Depth(),
			Config.Get(),
			Lengths.Get(),
			Store.GetLevels());
	}

	void Masks::Sort()
	{
		CudaArray<unsigned> tmp{ Count };
		initPermutation();

		thrust::gather(thrust::device, Permutation.Get(), Permutation.Get() + Count, Lengths.Get(), tmp.Get());
		thrust::stable_sort_by_key(thrust::device, tmp.Get(), tmp.Get() + Count, Permutation.Get());

		for (int level = Depth() - 1; level >= 0; --level)
			updatePermutation(level, tmp);

		for (int level = 0; level < Depth(); ++level)
			applyPermutation(level, tmp);

		thrust::gather(thrust::device, Permutation.Get(), Permutation.Get() + Count, Lengths.Get(), tmp.Get());
		cuda::memory::copy(Lengths.Get(), tmp.Get(), Count * sizeof(unsigned));
	}

	Masks Masks::ReshapeMasks(Configuration& newConfig)
	{
		Masks newMasks{ newConfig, Count };
		launchReshape(newMasks);
		copyPermutation(newMasks);

		cuda::memory::copy(
			newMasks.Lengths.Get(), 
			Lengths.Get(), 
			Count * sizeof(unsigned));

		return newMasks;
	}

	std::ostream& operator<<(std::ostream& os, const Masks& obj)
	{
		auto h_store = obj.ToHost();
		auto h_lengths = obj.Lengths.ToHost();

		for (int item = 0; item < obj.Count; ++item)
		{
			int globalBit = 0;
			for (int level = 0; level < obj.Config.Depth(); ++level)
			{
				for (int bit = obj.Config[level] - 1; bit >= 0; --bit)
				{
					if(globalBit < h_lengths[item])
						std::cout << ((h_store[level][item] >> bit) & 1u);
					else
						std::cout << "X";

					++globalBit;
				}
				std::cout << "\t";
			}

			std::cout << "\t" << h_lengths[item] << std::endl;
		}

		return os;
	}

	bool operator==(const Masks& lhs, const Masks& rhs)
	{
		if(static_cast<const Keys&>(lhs) != static_cast<const Keys&>(rhs))
			return false;

		auto h_lhsLengths = lhs.Lengths.ToHost();
		auto h_rhsLenghts = rhs.Lengths.ToHost();

		auto cmpResult = std::mismatch(
			h_lhsLengths.begin(), 
			h_lhsLengths.end(), 
			h_rhsLenghts.begin());

		return cmpResult.first == h_lhsLengths.end();
	}

	bool operator!=(const Masks& lhs, const Masks& rhs)
	{
		return !(lhs == rhs);
	}

	size_t Masks::MemoryFootprint()
	{
		return Keys::MemoryFootprint() + Lengths.MemoryFootprint();
	}
}
