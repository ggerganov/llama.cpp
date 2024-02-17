/*
A utility to represent the kv-cache occupancy graphically
Takes as parameters
- total cache size (-c)
- number of simultaneous accesses/slots (-np)
- a parameter related to the display context (max window width - data display requirements)
It then uses a trick borrowed from tqdm to display occupancy
TODO: Show contiguous space and block availability
*/
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib> // for rand()

// my custom function to display graphics of the kvcache status
static void show_kvcache(std::vector<std::pair<int,struct llama_client_slot>> used_blocks, int cache_size) {

    int max_length = 128;
    int num_blocks = used_blocks.size();
    int slot_cache_size = cache_size / num_blocks;
    bool cls_flag = true;
    std::string slot_symbol1 = "";
    std::string slot_symbol2 = "";
    std::string slot_symbol3 = "";
    auto& p = used_blocks[0];
    llama_client_slot slot = p.second;

    return; // remove when not in debug mode

    if ((used_blocks.size() == 0) || (used_blocks[0].first == 0)) {
        return;
    }

    // Print visualization
    // Always start at the top left of the window (H means 'move cursor to this position'; 2J = cls)
    // Only clear the screen the first time round
    if (cls_flag) {
        printf("\033[2J");
        cls_flag = false;
    }
    printf("\033[1;0H\033[K**************************\n\033[KKVcache occupancy by slot:\n\033[K**************************\n");
    for(int i=0; i<num_blocks; i++) {
        printf("\033[K");  // clear the current line
        for(int j=0; j < max_length; j++) {
            int used = used_blocks[i].first * max_length / slot_cache_size;
            if((j < max_length / 2) && (j < used)) {
                printf("\033[90m█\033[0m");
            } else if (j < used) {
                printf("\033[94m█\033[0m");
            } else {
                printf("\033[91m█\033[0m");
            }
        }
        if(used_blocks[i].second.state == PROCESSING) {
            slot_symbol1 = "\u23F0"; // clock symbol = processing
        } else if(used_blocks[i].second.state == IDLE) {
            slot_symbol1 = "\u2705"; // red box white tick
        } else {
            slot_symbol1 = "\u2620"; // skull and crossbones symbol = dead?
        }
        if(used_blocks[i].second.command == LOAD_PROMPT) {
            slot_symbol2 = "\u24C1"; // dingbat L symbol = loading
        } else if(used_blocks[i].second.command == RELEASE) {
            slot_symbol2 = "\u24C7"; // dingbat R release
        } else if(used_blocks[i].second.command == NONE) {
            slot_symbol2 = "\u24C3"; // dingbat N none
        }
        if(used_blocks[i].first == slot_cache_size) {
            slot_symbol3 = "\u274E"; // red box white cross
        } else {
            slot_symbol3 = "";
        }
    printf(" %4d/%5d %2d %s %s %s\n", used_blocks[i].first, slot_cache_size, used_blocks[i].second.id, slot_symbol1.c_str(), slot_symbol2.c_str(), slot_symbol3.c_str());
    }
    printf("\n\033[%dJ", 0);
}
