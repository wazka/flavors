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
        vector<vector<unsigned>> h_children(Depth());

        for(int level = Depth() - 1; level > 0; --level)
        {
            auto currentLevel = Children.ToHost(level);
            vector<unsigned> newLastLevel;

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
                        newLastLevel.push_back(0);
                }

                for(int child = 0; child < ChildrenCountsHost[level]; ++child)
                {
                    if(currentLevel[node * ChildrenCountsHost[level] + child] != 0)
                        newLastLevel[newNode * ChildrenCountsHost[level] + child] = currentLevel[node * ChildrenCountsHost[level] + child];

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
            h_LevelsSizes[level] = newLastLevelSize;
        }

        h_children[0] = Children.ToHost(0);

        allocateNodes();

        for(int level = 0; level < Depth(); ++level)
            cuda::memory::copy(Children[level], h_children[level].data(), h_children[level].size() * sizeof(unsigned));
    }
}