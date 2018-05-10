#include "compressedTree.h"

#include <device_launch_parameters.h>

using namespace std;

namespace Flavors
{
    __global__ void mapIndexes(unsigned* level, unsigned* map, unsigned int count)
    {
        int index = blockIdx.x * blockDim.x + threadIdx.x;

        if(index < count)
            if(level[index] != 0)
                 level[index] = map[level[index] - 1] + 1;
    }

    CompressedTree::CompressedTree(Keys& keys) : Tree(keys)
    {
        auto h_originalChildren = Children.ToHost();

        vector<vector<unsigned>> h_children(Depth());
        vector<vector<unsigned>> h_originalIndex(Depth());
        vector<vector<unsigned>> h_parentIndex(Depth());

        for(int level = Depth() - 1; level > 0; --level)
        {
            auto currentLevel = Children.ToHost(level);
            vector<unsigned> newLastLevel;
            vector<unsigned> newParentIndex;
            vector<unsigned> newOriginalIndex;

            int newLastLevelSize = 0;

            vector<unsigned> h_indexesMap;

            for(int node = 0; node < h_LevelsSizes[level]; ++node)
            {
                int newNode = 0;

                while(newNode < newLastLevelSize)
                {
                    bool canMerge = true;
                    for(int child = 0; child < ChildrenCountsHost[level]; ++child)
                    {
                        if(currentLevel[node * ChildrenCountsHost[level] + child] != 0 &&
                            newLastLevel[newNode * ChildrenCountsHost[level] + child] != 0)
                        {
                            canMerge = false;
                            break;
                        }
                    }

                    if(canMerge)
                        break;

                    ++newNode;
                }

                if(newNode == newLastLevelSize)
                {
                    newLastLevelSize++;

                    for(int i = 0; i < ChildrenCountsHost[level]; ++i)
                    {
                        newLastLevel.push_back(0);
                        newParentIndex.push_back(0);
                        newOriginalIndex.push_back(0);
                    }
                }

                for(int child = 0; child < ChildrenCountsHost[level]; ++child)
                {
                    if(currentLevel[node * ChildrenCountsHost[level] + child] != 0)
                    {
                        newLastLevel[newNode * ChildrenCountsHost[level] + child] = currentLevel[node * ChildrenCountsHost[level] + child];
                        newOriginalIndex[newNode * ChildrenCountsHost[level] + child] = node + 1;
                        newParentIndex[newNode * ChildrenCountsHost[level] + child] = h_originalChildren[level][node * ChildrenCountsHost[level] + child];
                    }

                }

                h_indexesMap.push_back(newNode);            
            }

            CudaArray<unsigned> indexMap{h_indexesMap};
            auto kernelConfig = make_launch_config(h_LevelsSizes[level] * ChildrenCountsHost[level]);

            cuda::launch(
                mapIndexes,
                kernelConfig,
                Children[level - 1],
                indexMap.Get(),
                h_LevelsSizes[level - 1] * ChildrenCountsHost[level - 1]
            );

            h_children[level] = newLastLevel;
            h_parentIndex[level] = newParentIndex;
            h_originalIndex[level] = newOriginalIndex;
            h_LevelsSizes[level] = newLastLevelSize;
        }

        h_children[0] = Children.ToHost(0);
        h_parentIndex[0] = h_originalChildren[0];

        std::vector<unsigned> levelsRawSizes;
		for (int level = 0; level < Depth(); ++level)
			levelsRawSizes.push_back(h_LevelsSizes[level] * ChildrenCountsHost[level]);

        Children = CudaJaggedArray{ levelsRawSizes };
        ParentIndex = CudaJaggedArray{ levelsRawSizes };
        OriginalIndex = CudaJaggedArray{ levelsRawSizes };

        for(int level = 0; level < Depth(); ++level)
        {
            cuda::memory::copy(Children[level], h_children[level].data(), h_children[level].size() * sizeof(unsigned));
            cuda::memory::copy(ParentIndex[level], h_parentIndex[level].data(), h_parentIndex[level].size() * sizeof(unsigned));
            cuda::memory::copy(OriginalIndex[level], h_originalIndex[level].data(), h_originalIndex[level].size() * sizeof(unsigned));
        }
    }

    __global__ void findKeysKernelCompressed(
        unsigned** children,
        unsigned** parentIndex,
        unsigned** originalIndex,
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
        int index = parentIndex[0][keysToFind[0][key]];

        if (currentNode == 0)
            return;

        for(int level = 1; level < depth; ++level)
        {
            if(index != originalIndex[level][(currentNode - 1) * childrenSizes[level] + keysToFind[level][key]])
                return;

            index = parentIndex[level][(currentNode - 1) * childrenSizes[level] + keysToFind[level][key]];
            currentNode = children[level][(currentNode - 1) * childrenSizes[level] + keysToFind[level][key]];

            if (currentNode == 0)
                return;
        }

        result[key] = currentNode;
    }

    void CompressedTree::FindKeys(Keys& keys, unsigned* result)
	{
		auto kernelConfig = make_launch_config(keys.Count);
		cuda::launch(
			findKeysKernelCompressed,
			kernelConfig,
			Children.GetLevels(),
            ParentIndex.GetLevels(),
            OriginalIndex.GetLevels(),
			ChildrenCounts.Get(),
			keys.Store.GetLevels(),
			keys.Count,
			result,
			Depth());
    }

    void CompressedTree::Find(Keys& keys, unsigned* result)
	{
		FindKeys(keys, result);
	}
    
    size_t CompressedTree::MemoryFootprint()
	{
		return
                Children.MemoryFootprint() +
                OriginalIndex.MemoryFootprint() + 
                ParentIndex.MemoryFootprint() + 
				ChildrenCounts.MemoryFootprint() +
				scan.MemoryFootprint() +
				preScan.MemoryFootprint() +
				permutation.MemoryFootprint() +
				lengths.MemoryFootprint() +
				masksParts.MemoryFootprint() +
				containers.MemoryFootprint();

	}
}