#pragma once
#include "utils.h"

namespace Flavors
{
	class Containers
	{
	public:
		CudaJaggedArray Lengths;
		CudaJaggedArray Starts;
		CudaArray<unsigned> Items;
		std::vector<unsigned> ItemsPerLevel;

		size_t MemoryFootprint()
		{
			return Lengths.MemoryFootprint() + Starts.MemoryFootprint() + Items.MemoryFootprint();
		}
	};
}
