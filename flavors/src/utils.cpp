#include "utils.h"

namespace Flavors
{
	Cuda2DArray::Cuda2DArray() :
		Depth(0),
		count(0)
	{
	}

	Cuda2DArray::Cuda2DArray(int depth, int count):
		Depth(depth),
		count(count),
		store(depth * count),
		levels(depth)
	{
		for (int level = 0; level < depth; ++level)
			h_levels.push_back(store.Get() + level * count);

		cuda::memory::copy(levels.Get(), h_levels.data(), depth * sizeof(unsigned*));
	}

	std::vector<std::vector<unsigned>> Cuda2DArray::ToHost() const
	{
		std::vector<std::vector<unsigned>> h_store;

		for (int level = 0; level < Depth; ++level)
		{
			std::vector<unsigned> row(count);
			cuda::memory::copy(row.data(), operator[](level), count * sizeof(unsigned));
			h_store.push_back(row);
		}

		return h_store;
	}

	CudaJaggedArray::CudaJaggedArray(std::vector<unsigned> sizes, unsigned offset) :
		sizes(sizes),
		levels(sizes.size())
	{
		auto currentDevice = cuda::device::current::get();
		std::vector<unsigned*> h_levelsPtrs;

		for (int level = 0; level < Depth(); ++level)
		{
			CudaArray<unsigned> levelArray{ static_cast<int>(sizes[level] + offset) };	//TODO: Remove cast
			h_levelsPtrs.push_back(levelArray.Get());
			h_levels.push_back(std::move(levelArray));
		}

		cuda::memory::copy(levels.Get(), h_levelsPtrs.data(), Depth() * sizeof(unsigned*));
	}

	int CudaJaggedArray::Depth() const
	{
		return sizes.size();
	}

	unsigned* CudaJaggedArray::operator[](int level)
	{
		return h_levels[level].Get();
	}

	std::vector<std::vector<unsigned>> CudaJaggedArray::ToHost()
	{
		std::vector<std::vector<unsigned>> h_store;

		for (int level = 0; level < Depth(); ++level)
		{
			std::vector<unsigned> row(sizes[level]);
			cuda::memory::copy(row.data(), operator[](level), sizes[level] * sizeof(unsigned));
			h_store.push_back(row);
		}

		return h_store;
	}
}
