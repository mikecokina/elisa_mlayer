import argparse
import numpy as np

from elisa_mlayer import config
from elisa_mlayer import sio
from elisa_mlayer.nn.clsf.morphology import Conv1DNet, MlpNet
from elisa_mlayer.logger import getLogger
from keras import backend as K
from keras.utils import to_categorical

logger = getLogger("precision.clsf.morphology")

NETS = {
    "Conv1DNet": Conv1DNet,
    "MlpNet": MlpNet
}


def loss(target, predict, nn):
    target = K.constant(to_categorical(target, 2))
    output = K.constant(predict)
    return K.eval(K.mean(nn.model.loss(target, output)))


def main(net, model_path, table_name):
    """
    True means is detached
    """

    observed_io = sio.ObservedMySqlIO(db_conf=config.DB_CONF, table_name=table_name)
    xs, ys = next(observed_io.get_predictor_iter(batch_size=100)())

    if net in ["Conv1DNet"]:
        xs = np.expand_dims(xs, axis=2)

    net = NETS[net]

    _nn = net(test_size=0.0, reinitialize_feed=False)
    _nn.load_weights(model_path)

    predict = _nn.predict(xs)
    prediction = np.argmax(predict, axis=1)

    total_acc = sum(np.equal(prediction, ys)) / len(ys)
    total_loss = loss(ys, predict, _nn)

    detached = np.where(ys == 1)[0]
    detached_acc = sum(np.equal(prediction[detached], ys[detached])) / len(detached)

    over_contact = np.where(ys == 0)[0]
    over_contact_acc = sum(np.equal(prediction[over_contact], ys[over_contact])) / len(over_contact)

    logger.info("Precision")
    logger.info(f"Total Acc: {total_acc}")
    logger.info(f"Total Loss: {total_loss}")
    logger.info(f"Detached Acc: {detached_acc}")
    logger.info(f"Over-Contact Acc: {over_contact_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('net', nargs='?', choices=['MlpNet', 'Conv1DNet'], help=f'net choice')
    parser.add_argument('--table', type=str, nargs='?', help='table name to read data', default="observed_lc")
    parser.add_argument('--load-model', type=str, nargs='?', help='path to h5 file for model', default=None)

    args = parser.parse_args()

    if args.net is None:
        raise ValueError("Positional argument is required, choices: `MlpNet`, `Conv1DNet`")

    main(net=args.net, model_path=args.load_model, table_name=args.table)
