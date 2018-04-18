#pragma once
#include "tree.h"

namespace Flavors
{
    class CompressedTree : public Tree
    {
    public:
        explicit CompressedTree(Keys& keys);

    };

}