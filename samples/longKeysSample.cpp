
#include "keys.h"
#include "tree.h"

#include <vector>
#include <iostream>

using namespace Flavors;
using namespace std;

int main()
{
    //This sample show how to handle longer keys
    //We have 5 keys, 64-bit each
    //Please be advised, that data is stored in level by level order
    vector<unsigned> data {
        0, 1, 2, 3, 4, 
        5, 6, 7, 8, 9
    };
    int keysCount = 5;

    //Here we prepare configuration
    vector<unsigned> levels{32, 32};
    Configuration config{levels};

    //And create keys
    Keys keys{config, keysCount};
    keys.FillFromVector(data);

    cout << "Keys are:\n";
    cout << keys << endl;
    // Keys are
    // 00000000000000000000000000000000        00000000000000000000000000000101
    // 00000000000000000000000000000001        00000000000000000000000000000110
    // 00000000000000000000000000000010        00000000000000000000000000000111
    // 00000000000000000000000000000011        00000000000000000000000000001000
    // 00000000000000000000000000000100        00000000000000000000000000001001

    //Now we can reshape them
    vector<unsigned> newLevels{8, 8, 8, 8, 8, 8, 8, 8};
    Configuration newConfig{newLevels};

    auto newKeys = keys.ReshapeKeys(newConfig);

    cout << "Now, keys are:\n";
    cout << newKeys << endl; 
    // Now, keys are:
    // 00000000        00000000        00000000        00000000        00000000        00000000        00000000        00000101
    // 00000000        00000000        00000000        00000001        00000000        00000000        00000000        00000110
    // 00000000        00000000        00000000        00000010        00000000        00000000        00000000        00000111
    // 00000000        00000000        00000000        00000011        00000000        00000000        00000000        00001000
    // 00000000        00000000        00000000        00000100        00000000        00000000        00000000        00001001

    //Building the tree
    Tree tree{newKeys};

    //Finally we can find keys in tree
    CudaArray<unsigned> result{keysCount};
    tree.Find(newKeys, result.Get());

    cout << "Result:\n";
    cout << result << endl;
    // Result:
    // 1, 2, 3, 4, 5,
}