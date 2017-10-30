#pragma once

#include "keys.h"
#include "masks.h"
#include "containers.h"

namespace Flavors
{
	class Tree
	{
		Tree();

	public:
		explicit Tree(Keys& keys);

		explicit Tree(Masks& masks);

		void FindKeys(Keys& keys, unsigned* result);
		void FindMasks(Masks& masks, unsigned* result);

		Configuration Config;
		std::vector<unsigned> h_LevelsSizes;
		CudaJaggedArray Children;

		int Depth() const { return Config.Depth(); };
		int Count;

		CudaArray<unsigned> ChildrenCounts;
		std::vector<unsigned> ChildrenCountsHost;

		Tree(const Tree& other) = delete;
		Tree(Tree&& other) noexcept = delete;
		Tree& operator=(const Tree& other) = delete;
		Tree& operator=(Tree&& other) noexcept = delete;
	private:
		void countNodes(Cuda2DArray& borders, Cuda2DArray& indexes);
		void removeEmptyLevels();
		void countNodes(Cuda2DArray& borders, Cuda2DArray& indexes, Cuda2DArray& paths);
		void fillNodes(Cuda2DArray& borders, Cuda2DArray& indexes, Keys& keys, bool forMasks = false);
		void allocateNodes(bool forMasks = false);
		void copyMasks(Masks& masks, Cuda2DArray& pathsEnds);

		void markPaths(Masks& masks, Cuda2DArray& paths, Cuda2DArray& pathsEnds);
		Cuda2DArray mapNewIndexes(Cuda2DArray& indexes, Cuda2DArray& paths);
		void removeEmptyNodes(Cuda2DArray& borders, Cuda2DArray& indexes, Cuda2DArray& indexesMap);

		void scanLevels();
		CudaArray<unsigned> scan;
		CudaArray<unsigned> preScan;

		CudaArray<unsigned> permutation;
		CudaArray<unsigned> lengths;
		CudaArray<unsigned> masksParts;

		Containers containers;

		void buildListsLenghts(Cuda2DArray& indexes, Cuda2DArray& pathsEnds, CudaArray<unsigned>& tmpArray);
		void buildListsStarts();
		void buildItems(Cuda2DArray& indexes, Cuda2DArray& pathsEnds, CudaArray<unsigned>& tmpArray);

		void getLevelsCodes(CudaArray<unsigned>& placementCodes, Cuda2DArray& pathsEnds);
		void getNodesCodes(CudaArray<unsigned>& placementCodes, Cuda2DArray& indexes, Cuda2DArray& pathsEnds);
	};
}