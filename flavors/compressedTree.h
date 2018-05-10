#pragma once
#include "tree.h"

namespace Flavors
{
    class CompressedTree : public Tree
    {
    public:
        explicit CompressedTree(Keys& keys);

        CudaJaggedArray ParentIndex;
        CudaJaggedArray OriginalIndex;

        void FindKeys(Keys& keys, unsigned* result);
        size_t MemoryFootprint();
        void Find(Keys& data, unsigned* result);

        std::string Caption()
		{
			return std::string{"CompressedTree"};
		}
    };

}