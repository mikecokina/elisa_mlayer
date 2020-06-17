import json
import os.path as op
import numpy as np

from typing import List
from matplotlib import pyplot as plt


FILENAME = "conv1d_spots.json"
DATA = op.join(op.abspath(op.dirname(__file__)), "data")


def get_data(filename):
    fpath = op.join(DATA, filename)
    with open(fpath, "r") as f:
        return json.loads(f.read())


history = get_data(FILENAME)
loss_history: List = history["loss_history"]
lr_s: List = history["lr_s"]


# allowed = lr_s
allowed = [1e-6, 1e-5, 1e-4, 1e-3, 4e-3, 7e-3, 1e-2]
for hist, learning_rate in zip(loss_history, lr_s):
    if learning_rate in allowed:
        plt.plot(np.arange(0, len(hist)), hist, label=f"lr: {learning_rate}")

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
plt.show()

plt.xscale("log")
plt.plot(lr_s, np.array(loss_history)[:, -1])
plt.xlabel('Learning Rate')
plt.ylabel('Loss')
plt.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
plt.show()
