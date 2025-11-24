
import sys
from pathlib import Path
import json
from matplotlib import pyplot as plt

target = "nsga/tests/simple/40_100_m_chance_dot01/results.json"
cwd = Path().cwd().resolve()

path = cwd / target


with open(path, "r") as f:
    data = json.load(f)

results = data[0] # just one run!!

conv = results["conv"]
deltas = results["deltas"]

_, axes = plt.subplots(nrows=2)
axes[0].plot(conv)
axes[0].set_xlabel("generations")
axes[0].set_ylabel("convergence")

axes[1].plot(deltas)
axes[1].set_xlabel("generations")
axes[1].set_ylabel("deltas")

plt.show()