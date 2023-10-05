// example of a C/C++ equivalent data structure to the python dict
// there are two: std::map automatically sorts on key; std::unordered_map doesn't

#include "llama.h"

#include <iostream>
#include <map>

int main() {
    std::map<std::string, int> dict;

    dict["apple"] = 5;
    dict["banana"] = 2;
    dict["orange"] = 7;


    // Accessing elements in the map
    std::cout << "Value of apple: " << dict["apple"] << std::endl;

    for (const auto& pair : dict) {
        std::cout << "Key: " << pair.first << ", Value: " << pair.second << std::endl;
    }

    return 0;
}