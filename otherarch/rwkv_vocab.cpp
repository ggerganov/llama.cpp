#include <vector>
#include <string>
#include <fstream>
#include <iostream>

#include "expose.h"

std::vector<std::string> rwkv_vocab;

void read_rwkv_vocab()
{
    std::string line;
    auto filepath = executable_path+ "rwkv_vocab.embd";
    printf("Reading vocab from %s",filepath.c_str());
    std::ifstream myfile(filepath);
    if (myfile.is_open())
    {
        while (myfile.good())
        {
            getline(myfile, line);
            rwkv_vocab.push_back(line);
        }
        myfile.close();
    }

    else
    {
        std::cout << "Unable to open RWKV vocab file";
    }
}