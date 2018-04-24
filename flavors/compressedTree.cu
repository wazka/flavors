#include "compressedTree.h"

using namespace std;

namespace Flavors
{
    CompressedTree::CompressedTree(Keys& keys)
    {
        Config = keys.Config;
		h_LevelsSizes = std::vector<unsigned>(keys.Depth());
		Count = keys.Count;

        auto h_keys = keys.ToHost();

        vector<vector<unsigned>> h_indexes;
        h_LevelsSizes[0] = 1;

        for(int level = 0; level < Depth(); ++level)
            h_indexes.push_back(vector<unsigned>(Count, 0));

        for(int level = 1; level < Depth(); ++level)
        {
            vector<vector<int>> counters;
            for(int node = 0; node < h_LevelsSizes[level - 1]; ++node)
                counters.push_back(vector<int>(1u << Config[level], 0));
            
            int levelSize = 0;

            for(int key = 0; key < Count; ++key)
            {
                unsigned value = h_keys[level][key];
                h_indexes[level][key] = counters[h_indexes[level-1][key]][value];
                ++counters[h_indexes[level-1][key]][value];

                if(counters[h_indexes[level-1][key]][value] > levelSize)
                    levelSize = counters[h_indexes[level-1][key]][value];
            }

            h_LevelsSizes[level] = levelSize;
        }

        allocateNodes();

        for(int level = 0; level < Depth(); ++level)
            for(int key = 0; key < Count; ++key)
                ++h_indexes[level][key];

        Cuda2DArray indexes{Depth(), Count};

        for(int level = 0; level < Depth(); ++level)
            cuda::memory::copy(indexes[level], h_indexes[level].data(), Count * sizeof(unsigned));


        // Cuda2DArray borders = keys.Borders();
		// Cuda2DArray indexes = keys.Indexes(borders);
		// levelsSizesToHost(indexes);
		// fillNodes(  borders, indexes, keys);
    }
}