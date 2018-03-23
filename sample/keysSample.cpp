
#include "keys.h"
#include "tree.h"

#include <vector>
#include <iostream>

using namespace Flavors;
using namespace std;

int main()
{
    //Prepare data
    vector<unsigned> data{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int keysCount = data.size();

    //Prepare configuration (bit strides)
    //For now we do one level of 32 bits, so it will match our data
    Configuration config = Configuration::Default32;

    //Build Keys class instance
    //Constructor allocates memory on GPU
    Flavors::Keys keys{config, keysCount};

    //This method is copying data from vector to GPU memory
    keys.FillFromVector(data);

    //Print will look like this
    cout << "Our keys are: \n"
         << keys << endl;
    //
    // Our keys are:
    // 00000000000000000000000000000000
    // 00000000000000000000000000000001
    // 00000000000000000000000000000010
    // 00000000000000000000000000000011
    // 00000000000000000000000000000100
    // 00000000000000000000000000000101
    // 00000000000000000000000000000110
    // 00000000000000000000000000000111
    // 00000000000000000000000000001000
    // 00000000000000000000000000001001

    //Now, we can reshape our keys in any way we desire
    vector<unsigned> levels{8, 8, 8, 4, 4};
    Configuration newConfig{levels};

    auto newKeys = keys.ReshapeKeys(newConfig);

    //Let's look at keys now
    cout << "New keys look like this:\n"
         << newKeys << endl;
    //
    // New keys look like this:
    // 00000000        00000000        00000000        0000    0000
    // 00000000        00000000        00000000        0000    0001
    // 00000000        00000000        00000000        0000    0010
    // 00000000        00000000        00000000        0000    0011
    // 00000000        00000000        00000000        0000    0100
    // 00000000        00000000        00000000        0000    0101
    // 00000000        00000000        00000000        0000    0110
    // 00000000        00000000        00000000        0000    0111
    // 00000000        00000000        00000000        0000    1000
    // 00000000        00000000        00000000        0000    1001

    // Finally, we are ready to build a tree
    Tree tree{newKeys};

    //And to find something in it

    //CudaArray is nothig more, than a wrapper around cudamalloc
    //Get method returns pointer to raw memory
    CudaArray<unsigned> result{newKeys.Count};

    tree.Find(newKeys, result.Get());

    //Now, in results array we have indexes of found keys
    cout << "Result is:\n" << result << endl;
    //
    //In result we index from 1, because 0 means no value was found, so it will be
    //
    // 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,


    //Now, we will try to find randomly generated keys
    Keys randomKeys{newConfig, keysCount};
    randomKeys.FillRandom(0);

    //As we are wery efficient programmers, we will reuse the same memory for result
    result.Clear();

    tree.Find(randomKeys, result.Get());

    cout << "New result is:\n" << result << endl;
    //
    //On my machine new result is
    //
    //0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    //
    //This means, that no keys from randomKeys were found in the tree
}