#!/usr/bin/env python3

import sqlite3
import matplotlib.pyplot as plt

con = sqlite3.connect("llama.sqlite")
cur = con.cursor()

ts = []

for ngl in range(0, 36):
    res = cur.execute(f"SELECT t_eval_us,n_eval FROM llama_runs WHERE n_gpu_layers={ngl};")
    t_eval_us, n_eval = res.fetchone()
    ts.append(n_eval * 1000000/t_eval_us)

plt.plot(ts)
plt.xlim(0, 35)
plt.ylim(0, 130)
plt.title("7b q4_0, 3700X, 3200 MHz dual-channel RAM, RTX 3090")
plt.xlabel("-ngl")
plt.ylabel("Generated t/s")
plt.savefig("benchmark.png", dpi=240)
plt.show()
