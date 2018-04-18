#include "compressedTree.h"

namespace Flavors
{
    CompressedTree::CompressedTree(Keys& keys)
    {
        Config = keys.Config;
		h_LevelsSizes = std::vector<unsigned>(keys.Depth());
		Count = keys.Count;

        auto h_keys = keys.ToHost();

        for(int level = 0; level < Depth(); ++level)
        {
            std::vector<int> counters(1u << Config[level], 0);
            int levelSize = 0;

            for(int key = 0; key < Count; ++key)
            {
                unsigned value = h_keys[level][key];
                int index = counters[value];
                ++counters[value];

                if(counters[value] > levelSize)
                    levelSize = counters[value];

            }


            h_LevelsSizes[level] = levelSize;
        }

        allocateNodes();


        // Cuda2DArray borders = keys.Borders();
		// Cuda2DArray indexes = keys.Indexes(borders);
		// levelsSizesToHost(indexes);
		// fillNodes(  borders, indexes, keys);
    }
}