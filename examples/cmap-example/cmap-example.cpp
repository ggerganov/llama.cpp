// example of a C/C++ equivalent data structure to the python dict
// there are two: std::map automatically sorts on key; std::unordered_map does not

#include <map>
#include <list>
#include <string>
#include <bitset>

int main() {
    std::map<std::string, int> dict;
    std::map<std::string, std::list<std::string>> helpdict;

    dict[std::string("apple")] = 5;
    dict[std::string("banana")] = 2;
    dict[std::string("orange")] = 7;


    // Accessing elements in the map
    printf("First kind of dictionary\n\nValue of apple: %d\n", dict[std::string("apple")]);

    for (const auto& pair : dict) {
    printf("Key: %10s, Value: %4d\n", pair.first.c_str(), pair.second);
    }

    // Now try the helpdict idea
    
    printf("Second kind of dictionary\n");

    // Create a list of strings
    std::list<std::string> stringList = {"apple", "banana", "orange"};

    // Add key-value pair to map
    helpdict["fruits"] = stringList;

    // Access and modify the list of strings
    std::list<std::string>& fruitsList = helpdict["fruits"];
    fruitsList.push_back("grape");
    fruitsList.push_back("pineapple");
   
    for (const auto& pair : helpdict) {
        printf("helpdict contains a list of %s\n", pair.first.c_str());
        for (const auto& element : pair.second) {
            printf(" %s", element.c_str());
        }
        printf("\n");
    }

    // Create a binary key for each value consisting of a list of strings

    std::map<std::string, std::list<std::string>> bitdict;

    // Example binary key
    int binaryKey1 = 0b0000001;
    int binaryKey2 = 0b0000010;
    int binaryKey3 = 0b0000100;
    int binaryKey4 = 0b0001000;
    int binaryKey5 = 0b0010000;

    // Convert binary key to string
    std::string keyString1 = std::bitset<8>(binaryKey1).to_string();
    std::string keyString2 = std::bitset<8>(binaryKey2).to_string();
    std::string keyString3 = std::bitset<8>(binaryKey3).to_string();
    std::string keyString4 = std::bitset<8>(binaryKey4).to_string();
    std::string keyString5 = std::bitset<8>(binaryKey5).to_string();

    // Add key-value pair to map
    bitdict[keyString1] = {"-h", "--help", "print this help list and exit"};
    bitdict[keyString2] = {"-f", "FNAME", "--file", "FNAME", "read the prompts from an external text file"};
    bitdict[keyString3] = {"-n", "N", "--n-predict", "N", "number of tokens to predict in generating a completion"};
    bitdict[keyString4] = {"-t", "N", "--threads", "N", "number of threads to use"};
    bitdict[keyString5] = {"-m", "MODELPATH", "--model", "MODELPATH", "path to llama model to use"};


    for (const auto& pair : bitdict) {
        printf("help dictionary contains a list of arguments specific to this app %s\n", pair.first.substr(pair.first.size() - 5).c_str());
        for (const auto& element : pair.second) {
            printf(" %5s", element.c_str());
        }
        printf("\n");
    }

    printf("\nThis is the complete help file in this mock-up illustrative example:\n\n");
    for (const auto& pair: bitdict) {
        printf("%s  ",pair.first.c_str());
        for (const auto& element : pair.second) {
            printf(" %5s", element.c_str());
    }
        printf("\n");
    }

    // Now we try to use the appcode to select from the help available
    // app1 has only -h and -f so 0b00011; app2 has only -h and -n so 0b00101

    int app1code = 0b01011;
    int app2code = 0b10101;

    printf("\nNow processing app with only -h, -t and -f and appcode %3d\n", app1code);
    if (app1code != 0) {
        for (const auto& kvp : bitdict) {
            if ((app1code & std::stoi(kvp.first)) != 0) {
                printf("%s  ",kvp.first.c_str());
                for (const auto& element : kvp.second) {
                    printf(" %5s", element.c_str());
                }
                printf("\n");
            }
        }
        printf("\n");
    }

    printf("\nNow processing app with only -h, -m, -n and appcode %3d\n", app2code);
    if (app2code != 0) {
        for (const auto& kvp : bitdict) {
            if ((app2code & std::stoi(kvp.first)) != 0) {
                printf("%s  ",kvp.first.c_str());
                for (const auto& element : kvp.second) {
                    printf(" %5s", element.c_str());
                }
                printf("\n");
            }
        }
        printf("\n");
    }

    return 0;
}