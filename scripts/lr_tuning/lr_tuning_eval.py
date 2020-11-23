import json
import os.path as op
import numpy as np
import matplotlib as mpl

from typing import List
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

FILENAME = "conv1d_morphology_spotty.json"
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


def main():
    params = {'legend.fontsize': 10, 'legend.handlelength': 0.5, "font.size": 12}
    mpl.rcParams.update(params)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13, 5), sharey="all")
    ax1, ax2 = axes

    # ax1
    for hist, learning_rate in zip(loss_history, lr_s):
        if learning_rate in allowed:
            learning_rate = "{:.1e}".format(learning_rate)
            ax1.plot(np.arange(0, len(hist)), hist, label=f"l-rate: {learning_rate}", linewidth=2)

    _handles, _labels = ax1.get_legend_handles_labels()
    ax1.legend(_handles, _labels, loc='upper right')
    ax1.set_xlabel(r"Epoch"
                   "\n"
                   "\n"
                   r"a)", fontsize=12)
    ax1.set_ylabel(r"Loss", fontsize=12)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlim([-1.0, 20.0])
    ax1.set_ylim([0.0, 0.5])
    ax1.grid()

    # ax2
    plt.plot(lr_s, np.array(loss_history)[:, -1], linewidth=2)
    _handles, _labels = ax1.get_legend_handles_labels()
    ax2.set_xlabel(r"Learning Rate"
                   "\n"
                   "\n"
                   r"b)", fontsize=12)
    # ax2.set_ylabel(r"Loss", fontsize=12)
    plt.xscale("log")
    ax2.set_xlim([0.0, 2e-2])
    ax2.grid()

    plt.tight_layout(pad=1)


    plt.show()



if __name__ == '__main__':
    main()

# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend(loc="upper right", bbox_to_anchor=(1.0, 1.0))
# plt.show()
#
# plt.xscale("log")
# plt.plot(lr_s, np.array(loss_history)[:, -1])
# plt.xlabel('Learning Rate')
# plt.ylabel('Loss')
# plt.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0))
# plt.show()
