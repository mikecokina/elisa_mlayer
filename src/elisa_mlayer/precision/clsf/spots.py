import argparse
import json
import numpy as np

from matplotlib import pyplot as plt
from elisa_mlayer import config
from elisa_mlayer import sio
from elisa_mlayer.logger import getLogger
from elisa_mlayer.nn.clsf.spots import Conv1DNet, MlpNet

logger = getLogger("precision.clsf.spots")


NETS = {
    "Conv1DNet": Conv1DNet,
    "MlpNet": MlpNet
}

SPOTTY = {
    0: "no-spot",
    1: "spot"
}

JUST_SPOTS = True


def main(net, model_path, table_name):
    """
    True means is spotty
    """

    observed_io = sio.ObservedMySqlIO(db_conf=config.DB_CONF, table_name=table_name)
    data_iter = observed_io.get_iter(batch_size=1000)
    rows = next(data_iter())

    xs = np.array([json.loads(row[4])["flux"] for row in rows])
    target = [row[7] for row in rows]

    if net in ["Conv1DNet"]:
        xs = np.expand_dims(xs, axis=2)

    net = NETS[net]

    _nn = net(test_size=0.0, reinitialize_feed=False)
    _nn.load_weights(model_path)

    predict = _nn.predict(xs)
    prediction = np.argmax(predict, axis=1)

    for _xs, _prediction, _target in zip(xs, prediction, target):
        # if _target not in ["CO And", "PV Cas", "AD Boo", "ASAS J045304-0700.4"]:
        #     continue

        if JUST_SPOTS and _prediction == 1:
            plt.plot(np.linspace(-0.6, 0.6, 100, endpoint=True), _xs)
        elif not JUST_SPOTS and _prediction == 0:
            plt.plot(np.linspace(-0.6, 0.6, 100, endpoint=True), _xs)

        if (JUST_SPOTS and _prediction == 1) or (not JUST_SPOTS and _prediction == 0):
            plt.xlabel('Phase', fontsize=18)
            plt.ylabel('Flux', fontsize=18)

            plt.title(_target, fontsize=20)
            plt.xticks(fontsize=18, rotation=0)
            plt.yticks(fontsize=18, rotation=0)
            plt.show()
            plt.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('net', nargs='?', choices=['MlpNet', 'Conv1DNet'], help=f'net choice')
    parser.add_argument('--table', type=str, nargs='?', help='table name to read data', default="observed_lc")
    parser.add_argument('--load-model', type=str, nargs='?', help='path to h5 file for model', default=None)

    args = parser.parse_args()

    if args.net is None:
        raise ValueError("Positional argument is required, choices: `MlpNet`, `Conv1DNet`")

    main(net=args.net, model_path=args.load_model, table_name=args.table)
