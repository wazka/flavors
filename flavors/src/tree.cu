#include "tree.h"

#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <device_launch_parameters.h>

namespace Flavors
{
	Tree::Tree(): Count(0)
	{
	}

	Tree::Tree(Keys& keys) : Tree()
	{
		Config = keys.Config;
		h_LevelsSizes = std::vector<unsigned>(keys.Depth());
		Count = keys.Count;

		Cuda2DArray borders = keys.Borders();
		Cuda2DArray indexes = keys.Indexes(borders);
		levelsSizesToHost(indexes);
		fillNodes(borders, indexes, keys);
	}

	Tree::Tree(Masks& masks) : Tree()
	{
		Config = masks.Config;
		h_LevelsSizes = std::vector<unsigned>(masks.Depth());
		Count = masks.Count;

		Cuda2DArray borders = masks.Borders();
		Cuda2DArray indexes{ Depth(), Count };

		Cuda2DArray paths{ Depth(), Count };
		Cuda2DArray pathsEnds{ Depth(), Count };

		scanLevels();
		markPaths(masks, paths, pathsEnds);
		countNodes(borders, indexes, paths);
		fillNodes(borders, indexes, masks, true);

		copyMasks(masks, pathsEnds);

		auto tmpArray = CudaArray<unsigned>{ Count };
		buildListsLenghts(indexes, pathsEnds, tmpArray);
		buildListsStarts();
		buildItems(indexes, pathsEnds, tmpArray);
	}

	void Tree::levelsSizesToHost(Cuda2DArray& indexes)
	{
		h_LevelsSizes[0] = 1;

		for(int level = 1; level < Depth(); ++level)
			cuda::memory::copy(h_LevelsSizes.data() + level, indexes[level] + Count - 1, sizeof(unsigned));
	}

	void Tree::removeEmptyLevels()
	{
		for (int level = Depth() - 1; level >= 0; --level)
			if (h_LevelsSizes[level] == 0)
			{
				Config.PopLastLevel();
				h_LevelsSizes.pop_back();
			}
			else
				break;
	}

	void Tree::countNodes(Cuda2DArray& borders, Cuda2DArray& indexes, Cuda2DArray& paths)
	{
		Keys::Indexes(borders, indexes, Count, Depth());
		levelsSizesToHost(indexes);

		auto indexesMap = mapNewIndexes(indexes, paths);
		thrust::transform(
			thrust::device,
			indexes.Get(),
			indexes.Get() + Depth() * Count,
			borders.Get(),
			indexes.Get(),
			thrust::multiplies<unsigned>());
		removeEmptyNodes(borders, indexes, indexesMap);

		Keys::Indexes(borders, indexes, Count, Depth());
		levelsSizesToHost(indexes);

		removeEmptyLevels();
	}

	__global__ void fillNodesKernel(int level, unsigned *children, unsigned *store, unsigned *indexes, unsigned* nextIndexes, unsigned* nextBorders, int count, unsigned* childrenCounts)
	{
		int key = blockIdx.x * blockDim.x + threadIdx.x;
		if (key >= count)
			return;

		if (nextBorders[key] == 0)
			return;

		children[(indexes[key] - 1) * childrenCounts[level] + store[key]] = nextIndexes[key];
	}

	__global__ void fillLeavesKernel(int level, unsigned *children, unsigned *store, unsigned *indexes, int count, unsigned* permutation, unsigned* childrenCounts)
	{
		int key = blockIdx.x * blockDim.x + threadIdx.x;
		if (key >= count)
			return;

		children[(indexes[key] - 1) * childrenCounts[level] + store[key]] = permutation[key] + 1;
	}

	void Tree::fillNodes(Cuda2DArray& borders, Cuda2DArray& indexes, Keys& keys, bool forMasks)
	{
		allocateNodes(forMasks);

		auto kernelConfig = make_launch_config(Count);
		for (int level = 0; level < Depth() - 1; ++level)		//TODO: This loop could possibly be moved to kernel. Would it be faster?
		{
			cuda::launch(
				fillNodesKernel,
				kernelConfig,
				level,
				Children[level],
				keys.Store[level],
				indexes[level],
				indexes[level + 1],
				borders[level + 1],
				Count,
				ChildrenCounts.Get());
		}

		if (forMasks)
			return;

		int lastLevel = Depth() - 1;
		auto leavesConfig = make_launch_config(Count);
		cuda::launch(
			fillLeavesKernel,
			leavesConfig,
			lastLevel,
			Children[lastLevel],
			keys.Store[lastLevel],
			indexes[lastLevel],
			Count,
			keys.Permutation.Get(),
			ChildrenCounts.Get());
	}

	void Tree::allocateNodes(bool forMasks)
	{
		std::vector<unsigned> levelsRawSizes;
		for (int level = 0; level < Depth(); ++level)
		{
			ChildrenCountsHost.push_back(1u << Config[level]);
			levelsRawSizes.push_back(h_LevelsSizes[level] * ChildrenCountsHost.back());
		}

		ChildrenCounts = CudaArray<unsigned>{ Depth() };
		cuda::memory::copy(ChildrenCounts.Get(), ChildrenCountsHost.data(), Depth() * sizeof(unsigned));

		if (forMasks)
			levelsRawSizes.pop_back();

		Children = CudaJaggedArray{ levelsRawSizes };
	}

	__global__ void copyMasksKernel(int count, unsigned* store, unsigned* pathsEnds, unsigned* masksParts)
	{
		int mask = blockIdx.x * blockDim.x + threadIdx.x;

		if (mask >= count)
			return;

		if (pathsEnds[mask] == 1)
			masksParts[mask] = store[mask];
	}

	void Tree::copyMasks(Masks& masks, Cuda2DArray& pathsEnds)
	{
		permutation = CudaArray<unsigned>{ Count };
		thrust::transform(thrust::device, masks.Permutation.Get(), masks.Permutation.Get() + Count, permutation.Get(), thrust::placeholders::_1 + 1);

		lengths = CudaArray<unsigned>{ Count };
		cuda::memory::copy(lengths.Get(), masks.Lengths.Get(), Count * sizeof(unsigned));

		masksParts = CudaArray<unsigned>{ Count };
		auto kernelConfig = make_launch_config(Count);
		for(int level = 0; level < Depth(); ++level)
		{
			cuda::launch(
				copyMasksKernel,
				kernelConfig,
				Count,
				masks.Store[level],
				pathsEnds[level],
				masksParts.Get()
			);
		}
	}

	__global__ void markPathsKernel (int count, int depth, unsigned** paths, unsigned** pathsEnds,  unsigned* lenghts, unsigned* rsPreSums, unsigned* rsSums)
	{
		int mask = blockIdx.x * blockDim.x + threadIdx.x;

		if (mask >= count)
			return;

		for (int level = 0; level < depth; ++level)
		{
			if (lenghts[mask] <= rsPreSums[level])
				return;

			paths[level][mask] = 1;

			if (lenghts[mask] <= rsSums[level])
				pathsEnds[level][mask] = 1;
		}
	}

	void Tree::markPaths(Masks& masks, Cuda2DArray& paths, Cuda2DArray& pathsEnds)
	{
		auto kernelConfig = make_launch_config(Count);
		cuda::launch(
			markPathsKernel,
			kernelConfig,
			Count,
			Depth(),
			paths.GetLevels(),
			pathsEnds.GetLevels(),
			masks.Lengths.Get(),
			preScan.Get(),
			scan.Get());
	}

	__global__ void mapIndexesKernel(int count, unsigned* indexes, unsigned* paths, unsigned* indexesMap)
	{
		int mask = blockIdx.x * blockDim.x + threadIdx.x;

		if (mask >= count)
			return;

		int nodeIndex = indexes[mask];
		if (nodeIndex * paths[mask] != 0)
			indexesMap[nodeIndex] = 1;
	}

	Cuda2DArray Tree::mapNewIndexes(Cuda2DArray& indexes, Cuda2DArray& paths)
	{
		Cuda2DArray indexesMap{ Depth(), Count + 1 };

		auto kernelConfig = make_launch_config(Count);
		for(int level = 0; level < Depth(); ++level)
		{
			cuda::launch(
				mapIndexesKernel,
				kernelConfig,
				Count,
				indexes[level],
				paths[level],
				indexesMap[level]);
		}

		return indexesMap;
	}

	__global__ void removeEmptyNodesKernel(int count, unsigned* borders, unsigned* indexes, unsigned* indexesMap)
	{
		int mask = blockIdx.x * blockDim.x + threadIdx.x;

		if (mask >= count)
			return;

		if (indexes[mask] == 0)
			return;

		borders[mask] = indexesMap[indexes[mask]];
	}

	void Tree::removeEmptyNodes(Cuda2DArray& borders, Cuda2DArray& indexes, Cuda2DArray& indexesMap)
	{
		auto kernelConfig = make_launch_config(Count);
		for(int level = 0; level < Depth(); ++level)
		{
			cuda::launch(
				removeEmptyNodesKernel,
				kernelConfig,
				Count,
				borders[level],
				indexes[level],
				indexesMap[level]);
		}
	}

	void Tree::scanLevels()
	{
		scan = CudaArray<unsigned>{ Depth() };
		preScan = CudaArray<unsigned>{ Depth() };
		
		thrust::inclusive_scan(thrust::device, Config.Get(), Config.Get() + Depth(), scan.Get());
		thrust::exclusive_scan(thrust::device, Config.Get(), Config.Get() + Depth(), preScan.Get());
	}

	void Tree::buildListsLenghts(Cuda2DArray& indexes, Cuda2DArray& pathsEnds, CudaArray<unsigned>& tmpArray)
	{
		containers.Lengths = CudaJaggedArray{ h_LevelsSizes };

		unsigned val = thrust::reduce(thrust::device, pathsEnds[0], pathsEnds[0] + Count);
		cuda::memory::copy(containers.Lengths[0], &val, sizeof(unsigned));

		for (int level = 1; level < Depth(); ++level)
		{
			cuda::memory::copy(&val, indexes[level], sizeof(unsigned));
			if (val == 0)
				thrust::replace(thrust::device, indexes[level], indexes[level] + Count, 0, 1);
			thrust::reduce_by_key(thrust::device, indexes[level], indexes[level] + Count, pathsEnds[level], tmpArray.Get(), containers.Lengths[level]);
		}
	}

	void Tree::buildListsStarts()
	{
		containers.Starts = CudaJaggedArray{ h_LevelsSizes, 1 };
		containers.ItemsPerLevel = std::vector<unsigned>(Depth());

		for (int level = 0; level < Depth(); ++level)
		{
			thrust::inclusive_scan(thrust::device, containers.Lengths[level], containers.Lengths[level] + h_LevelsSizes[level], containers.Starts[level] + 1);
			cuda::memory::copy(containers.ItemsPerLevel.data() + level, containers.Starts[level] + h_LevelsSizes[level], sizeof(unsigned));
		}

		//shifting lists
		int shift = 0;
		for (int level = 1; level < Depth(); ++level)
		{
			shift += containers.ItemsPerLevel[level - 1];
			thrust::transform(
				thrust::device, 
				containers.Starts[level], 
				containers.Starts[level] + h_LevelsSizes[level], 
				containers.Starts[level], 
				thrust::placeholders::_1 + shift);
		}
	}

	void Tree::buildItems(Cuda2DArray& indexes, Cuda2DArray& pathsEnds, CudaArray<unsigned>& tmpArray)
	{
		containers.Items = CudaArray<unsigned>{ Count };

		thrust::sequence(
			thrust::device, 
			containers.Items.Get(), 
			containers.Items.Get() + Count);

		CudaArray<unsigned> placementCodes{ Count };
		getLevelsCodes(placementCodes, pathsEnds);
		thrust::sort_by_key(thrust::device, placementCodes.Get(), placementCodes.Get() + Count, containers.Items.Get());

		getNodesCodes(placementCodes, indexes, pathsEnds);

		thrust::gather(
			thrust::device, 
			containers.Items.Get(), 
			containers.Items.Get() + Count, 
			placementCodes.Get(), 
			tmpArray.Get());

		int levelStart = 0;
		for (int level = 0; level < Depth(); ++level)
		{
			thrust::sort_by_key(
				thrust::device, 
				tmpArray.Get() + levelStart, 
				tmpArray.Get() + levelStart + containers.ItemsPerLevel[level],
				containers.Items.Get() + levelStart);

			levelStart += containers.ItemsPerLevel[level];
		}
	}

	void Tree::getLevelsCodes(CudaArray<unsigned>& placementCodes, Cuda2DArray& pathsEnds)
	{
		cuda::memory::copy(placementCodes.Get(), pathsEnds[0], Count * sizeof(unsigned));

		for (int level = 1; level < Depth(); ++level)
		{
			thrust::transform(
				thrust::device, 
				placementCodes.Get(),
				placementCodes.Get() + Count,
				pathsEnds[level], 
				placementCodes.Get(),
				thrust::placeholders::_1 + thrust::placeholders::_2 * (level + 1));
		}
	}

	void Tree::getNodesCodes(CudaArray<unsigned>& placementCodes, Cuda2DArray& indexes, Cuda2DArray& pathsEnds)
	{
		thrust::transform(
			thrust::device, 
			lengths.Get(), 
			lengths.Get() + Count, 
			placementCodes.Get(), 
			Config.Length - thrust::placeholders::_1);

		thrust::transform(
			thrust::device, 
			indexes.Get(), 
			indexes.Get() + Count * Depth(), 
			pathsEnds.Get(), 
			indexes.Get(), 
			thrust::multiplies<int>());

		for (int level = 0; level < Depth(); ++level)
			thrust::transform(
				thrust::device, placementCodes.Get(), 
				placementCodes.Get() + Count,
				indexes[level], 
				placementCodes.Get(), 
				thrust::placeholders::_1 + thrust::placeholders::_2 * (Config.Length + 1));
			
	}

	__global__ void findKeysKernel(
			unsigned** children,
			unsigned* childrenSizes,
			unsigned** keysToFind,
			unsigned keysToFindCount,
			unsigned* result,
			int depth)
	{
		int key = blockDim.x * blockIdx.x + threadIdx.x;
		if (key >= keysToFindCount)
			return;

		int currentNode = children[0][keysToFind[0][key]];

		if (currentNode == 0)
			return;

		for(int level = 1; level < depth; ++level)
		{
			currentNode = children[level][(currentNode - 1) * childrenSizes[level] + keysToFind[level][key]];

			if (currentNode == 0)
				return;
		}

		result[key] = currentNode;
	}

	void Tree::Find(Keys& keys, unsigned* result)
	{
		FindKeys(keys, result);
	}

	void Tree::FindKeys(Keys& keys, unsigned* result)
	{
		auto kernelConfig = make_launch_config(keys.Count);
		cuda::launch(
			findKeysKernel,
			kernelConfig,
			Children.GetLevels(),
			ChildrenCounts.Get(),
			keys.Store.GetLevels(),
			keys.Count,
			result,
			Depth());
	}

	__global__ void findMaskNodeKernel(
		unsigned* children,
		unsigned* childrenSizes,
		unsigned* keysToFind,
		int count,
		unsigned* currentNodes,
		unsigned* resultLevel,
		int level,
		unsigned* masksToFindLenghts,
		unsigned* rsSums,
		unsigned* result)
	{
		int mask = blockDim.x * blockIdx.x + threadIdx.x;

		if (mask >= count)
			return;

		if (masksToFindLenghts[mask] <= rsSums[level])
			return;

		unsigned currentNode = currentNodes[mask];
		if (currentNode == 0)
			return;

		result[mask] = children[(currentNode - 1) * childrenSizes[level] + keysToFind[mask]];
		++resultLevel[mask];
	}

	__global__ void findMaskItemKernel(
		int level, 
		unsigned* result, 
		unsigned* resultLevel, 
		int count, 
		unsigned* listsStarts, 
		unsigned* listsLenghts, 
		unsigned* listsItems,
		unsigned* masksParts, 
		unsigned* lenghts, 
		unsigned* masksToFind, 
		unsigned* masksToFindLenghts, 
		unsigned* permutation)
	{
		int mask = blockDim.x * blockIdx.x + threadIdx.x;

		if (mask >= count)
			return;

		unsigned currentNode = result[mask];
		if (currentNode == 0)	//path, that mask would be on ended before reaching proper length
			return;

		if (resultLevel[mask] != level)
			return;

		unsigned startIndex = listsStarts[currentNode - 1];
		unsigned item = 0;

		unsigned maskLenght = masksToFindLenghts[mask];
		int listLenght = listsLenghts[currentNode - 1];

		//moving to the part of the list, where masks with proper lenghts are
		while (item < listLenght && maskLenght != lenghts[listsItems[startIndex + item]])
			++item;

		unsigned itemValue = listsItems[startIndex + item];

		while (item < listLenght && maskLenght == lenghts[itemValue])
		{
			if (masksParts[itemValue] == masksToFind[mask])
			{
				result[mask] = permutation[itemValue];
				return;
			}

			++item;
			itemValue = listsItems[startIndex + item];
		}
	}

	void Tree::Find(Masks& masks, unsigned* result)
	{
		FindMasks(masks, result);
	}

	void Tree::FindMasks(Masks& masks, unsigned* result)
	{
		//TODO: This is very, very slow for long lists (when there is lot of short masks in the tree)

		CudaArray<unsigned> resultLevel{ masks.Count };
		thrust::fill_n(thrust::device, result, masks.Count, 1);

		auto kernelConfig = make_launch_config(masks.Count);
		for (int level = 0; level < Depth() - 1; ++level)
			cuda::launch(
				findMaskNodeKernel,
				kernelConfig,
				Children[level],
				ChildrenCounts.Get(),
				masks.Store[level],
				masks.Count,
				result,
				resultLevel.Get(),
				level,
				masks.Lengths.Get(),
				scan.Get(),
				result);

		auto kernelConfigFindItems = make_launch_config(masks.Count);
		for (int level = 0; level < Depth(); ++level)
			cuda::launch(
				findMaskItemKernel,
				kernelConfigFindItems,
				level,
				result,
				resultLevel.Get(),
				masks.Count,
				containers.Starts[level],
				containers.Lengths[level],
				containers.Items.Get(),
				masksParts.Get(),
				lengths.Get(),
				masks.Store[level],
				masks.Lengths.Get(),
				permutation.Get());
	}

	__global__ void findPathKernel(
			unsigned** children,
			unsigned* childrenSizes,
			unsigned** keysToMatch,
			unsigned keysCount,
			unsigned** path,
			int depth)
	{
		int key = blockDim.x * blockIdx.x + threadIdx.x;
		if (key >= keysCount)
			return;

		int currentNode = children[0][keysToMatch[0][key]];
		if (currentNode == 0)
			return;

		path[1][key] = currentNode;

		for(int level = 1; level < depth - 1; ++level)
		{
			currentNode = children[level][(currentNode - 1) * childrenSizes[level] + keysToMatch[level][key]];

			if (currentNode == 0)
				return;

			path[level + 1][key] = currentNode;
		}
	}

	__global__ void matchKeyKernel(
			int keysCount,
			int depth,
			unsigned** currentNodes,
			unsigned** listsStarts,
			unsigned** listsLenghts,
			unsigned* listsItems,
			unsigned* masksParts,
			unsigned* masksLengths,
			unsigned* stridesScan,
			unsigned* permutation,
			unsigned** keysToMatch,
			unsigned* result)
	{
		int key = blockDim.x * blockIdx.x + threadIdx.x;
		if (key >= keysCount)
			return;

		for(int level = depth - 1; level >= 0; --level)
		{
			int currentNode = currentNodes[level][key];

			if(currentNode == 0)
				continue;

			unsigned startIndex = listsStarts[level][currentNode - 1];
			unsigned listLength = listsLenghts[level][currentNode - 1];

			for(int itemIndex = 0; itemIndex < listLength; ++itemIndex)
			{
				unsigned itemValue = listsItems[startIndex + itemIndex];
				unsigned irrelevantBits = stridesScan[level] - masksLengths[itemValue];

				unsigned maskPart = masksParts[itemValue] >> irrelevantBits;
				unsigned keyPart = keysToMatch[level][key] >> irrelevantBits;

				if(maskPart == keyPart)
				{
					result[key] = permutation[itemValue];
					return;
				}
			}
		}
	}

	void Tree::Match(Keys& keys, unsigned* result)
	{
		Cuda2DArray currentNodes{ Depth(), keys.Count};
		thrust::fill_n(thrust::device, currentNodes.Get(), keys.Count, 1);

		auto kernelConfig = make_launch_config(keys.Count);
		cuda::launch(
			findPathKernel,
			kernelConfig,
			Children.GetLevels(),
			ChildrenCounts.Get(),
			keys.Store.GetLevels(),
			keys.Count,
			currentNodes.GetLevels(),
			Depth());

		cuda::launch(
			matchKeyKernel,
			kernelConfig,
			keys.Count,
			Depth(),
			currentNodes.GetLevels(),
			containers.Starts.GetLevels(),
			containers.Lengths.GetLevels(),
			containers.Items.Get(),
			masksParts.Get(),
			lengths.Get(),
			scan.Get(),
			permutation.Get(),
			keys.Store.GetLevels(),
			result);
	}

	size_t Tree::MemoryFootprint()
	{
		return
				Children.MemoryFootprint() +
				ChildrenCounts.MemoryFootprint() +
				scan.MemoryFootprint() +
				preScan.MemoryFootprint() +
				permutation.MemoryFootprint() +
				lengths.MemoryFootprint() +
				masksParts.MemoryFootprint() +
				containers.MemoryFootprint();

	}

}
