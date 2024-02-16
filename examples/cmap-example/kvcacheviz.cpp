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

static void show_kvcache(
  std::vector<int> used_blocks,
  int cache_size,
  int max_length
) {
  int num_blocks = used_blocks.size();
  int slot_cache_size = cache_size / num_blocks;

  while(true) {

    // Print visualization after erasing the current line
    for(int i=0; i<num_blocks; i++) {
      for(int j=0; j<max_length; j++) {
        if(j<used_blocks[i] * max_length / slot_cache_size) {
          std::cout << "\033[94m█\033[0m";
        }
        //else if ((j == int(used_blocks[i] * max_length / slot_cache_size + 0.5)) && (j > 7 * max_length / slot_cache_size + 0.5)) {
        //  std::cout << "\033[D\033[D\033[D\033[D" << std::setw(3) << used_blocks[i] << "\033[C";
        //}
        else {
          std::cout << "\033[91m█\033[0m";
        }
      }
    std::cout << " " << std::setw(5) << used_blocks[i] << "/" << std::setw(5) << slot_cache_size << std::endl;
    }
  std::cout << "{";
  std::string upcursor = "\033[K\033[A\033[K";

  for(int i=0; i < num_blocks; i++){
    //std::cout << used_blocks[i] << " ";
    upcursor += "\033[A\033[K";
  }

  // Remove first element
  used_blocks.erase(used_blocks.begin());

  // Add new random block at the end
  u_int new_block = rand() % slot_cache_size;
  used_blocks.push_back(new_block);

// Adjust the cursor so that the display overwrites itself
  upcursor += "\033[A\033[K";
  std::cout << "}" << std::endl;
  std::cin.get();
  std::cout << upcursor;
  }
}

int main() {
  std::vector<int> used_blocks = {64, 64, 64, 64, 64, 64, 64, 64, 64, 46, 46, 46, 46, 46, 46, 46, 46, 46};
  int cache_size = 65536;
  int max_length = 128;
  show_kvcache(used_blocks, cache_size, max_length);
  }
