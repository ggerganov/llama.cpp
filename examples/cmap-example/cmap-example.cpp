// example of a C/C++ equivalent data structure to the python dict
// there are two: std::map automatically sorts on key; std::unordered_map does not

#include <map>

int main() {
    std::map<std::string, int> dict;

    dict["apple"] = 5;
    dict["banana"] = 2;
    dict["orange"] = 7;


    // Accessing elements in the map
    printf("Value of apple: %d\n", dict["apple"]);

    for (const auto& pair : dict) {
    printf("Key: %s, Value: $s\n", pair.first, pair.second);
    }

    return 0;
}