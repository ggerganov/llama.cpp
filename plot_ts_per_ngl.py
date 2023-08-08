#!/usr/bin/env python3

import sqlite3
import numpy as np
import matplotlib.pyplot as plt

con = sqlite3.connect("llama.sqlite")
cur = con.cursor()

res = cur.execute("SELECT n_gpu_layers, 1000000.0*n_eval/t_eval_us FROM llama_runs ORDER BY n_gpu_layers;")
ts = np.array(res.fetchall())

plt.plot(ts[:, 0], ts[:, 1])
plt.xlim(0, 35)
plt.ylim(0, 130)
plt.title("7b q4_0, 3700X, 3200 MHz dual-channel RAM, RTX 3090")
plt.xlabel("-ngl")
plt.ylabel("Generated t/s")
plt.savefig("benchmark.png", dpi=240)
plt.show()
