// example of a C/C++ equivalent data structure to the python dict
// there are two: std::map automatically sorts on key; std::unordered_map does not

#include <map>
#include <string>

int main() {
    std::map<std::string, int> dict;

    dict[std::string("apple")] = 5;
    dict[std::string("banana")] = 2;
    dict[std::string("orange")] = 7;


    // Accessing elements in the map
    printf("Value of apple: %d\n", dict[std::string("apple")]);

    for (const auto& pair : dict) {
    printf("Key: %s, Value: $d\n", pair.first.c_str(), pair.second);
    }

    return 0;
}