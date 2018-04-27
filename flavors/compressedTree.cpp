#include "compressedTree.h"

#include <iomanip>

using namespace std;

namespace Flavors
{
    CompressedTree::CompressedTree(Keys& keys)
    {
        Config = keys.Config;
		h_LevelsSizes = std::vector<unsigned>(keys.Depth(), 0u);
		Count = keys.Count;

        auto h_keys = keys.ToHost();
        h_LevelsSizes[0] = 1;

        vector<vector<unsigned>> h_children;
        for(int level = 0; level < Depth(); ++level)
            h_children.push_back(vector<unsigned>());

        for (int level = 0; level < Depth(); ++level)
            ChildrenCountsHost.push_back(1u << Config[level]);

        for(int i = 0; i < ChildrenCountsHost[0]; ++i)
            h_children[0].push_back(0);

        h_LevelsSizes[0] = 1;


        for(int key = 0; key < Count; ++key)
        {
            int currentNode = 1;

            for(int level = 0; level < Depth() - 1; ++level)
            {
                int nextNode = h_children[level][(currentNode - 1) * ChildrenCountsHost[level] + h_keys[level][key]];

                if(nextNode == 0)
                {
                    // Jeśli nie było wolnego miejsca to alokacja nowego węzła
                    if(nextNode == 0)
                    {
                        h_LevelsSizes[level + 1]++;
                        nextNode = h_LevelsSizes[level + 1];

                        for(int i = 0; i < ChildrenCountsHost[level + 1]; ++i)
                            h_children[level + 1].push_back(0);
                    }

                    h_children[level][(currentNode - 1) * ChildrenCountsHost[level] + h_keys[level][key]] = nextNode;
                }

                currentNode = nextNode;
            }

            if(h_children[Depth() - 1][(currentNode - 1) * ChildrenCountsHost[Depth() - 1] + h_keys[Depth() - 1][key]] == 0)
                h_children[Depth() - 1][(currentNode - 1) * ChildrenCountsHost[Depth() - 1] + h_keys[Depth() - 1][key]] = key + 1;
            else
            {
                cout << "Nadpisywanie " << h_children[Depth() - 1][(currentNode - 1) * ChildrenCountsHost[Depth() - 1] + h_keys[Depth() - 1][key]] << " : " << key + 1 << endl;
            }
        }

        for(int level = Depth() - 1; level > 0; --level)
        {
            vector<unsigned> newLastLevel;
            int newLastLevelSize = 0;

            vector<unsigned> indexesMap;

            // Compressing tree
            for(int node = 0; node < h_LevelsSizes[level]; ++node)
            {
                int newNode = 0;

                while(newNode < newLastLevelSize)
                {
                    bool canMerge = true;
                    for(int child = 0; child < ChildrenCountsHost[level]; ++child)
                    {
                        if(h_children[level][node * ChildrenCountsHost[level] + child] != 0 &&
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
                    // newNode = newLastLevelSize;

                    for(int i = 0; i < ChildrenCountsHost[level]; ++i)
                        newLastLevel.push_back(0);
                }

                for(int child = 0; child < ChildrenCountsHost[level]; ++child)
                {
                    if(h_children[level][node * ChildrenCountsHost[level] + child] != 0)
                    {
                        if(newLastLevel[newNode * ChildrenCountsHost[level] + child] != 0)
                            cout << "ERROR\n";
                        newLastLevel[newNode * ChildrenCountsHost[level] + child] = h_children[level][node * ChildrenCountsHost[level] + child];
                    }

                }

                // cout << "Merging: " << node << " into " << newNode << endl;
                indexesMap.push_back(newNode);            
            }

            for(int i = 0; i < h_LevelsSizes[level - 1] * ChildrenCountsHost[level - 1]; ++i)
                if(h_children[level - 1][i] != 0)
                    h_children[level - 1][i] = indexesMap[h_children[level - 1][i] - 1] + 1;

            // cout << newLastLevelSize << endl;

            h_children[level] = newLastLevel;
            h_LevelsSizes[level] = newLastLevelSize;
        }

        allocateNodes();

        for(int level = 0; level < Depth(); ++level)
            cuda::memory::copy(Children[level], h_children[level].data(), h_children[level].size() * sizeof(unsigned));
    }
}