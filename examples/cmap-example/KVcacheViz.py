# A simple illustration of how to represent cache occupancy
# graphically using unicvode blocks
# which are generated using print("\u2588"), print("\u2591")

from time import sleep
import random

CACHE_SIZE = 50 
used_blocks = [5, 3, 2, 1, 10, 2, 6, 4, 7, 10]

def visualize_kv_cache(used_blocks, total_size):
    cache_viz = "["
    tot_used = 0
    for i in range(len(used_blocks)):
        # cache_viz += "█" * used_blocks[i]
        cache_viz += "\u2589" * used_blocks[i]
        cache_viz += "░" * (total_size - used_blocks[i])
        cache_viz += f"{used_blocks[i]:3.0f}/{total_size}]\r["
        tot_used += used_blocks[i]

        #print(f"\r[{cache_viz}] {used_blocks[i]:2.0f}/{total_size}", end="")

    print(f"\r{cache_viz}] {tot_used}/{len(used_blocks) * total_size}", end="")
    

while True:
    visualize_kv_cache(used_blocks, CACHE_SIZE)
    sleep(0.5)
    used_blocks = used_blocks[1:] + [random.randint(0,50)] # update used blocks
