import os
import json
import numpy as np
from datetime import datetime

import tensorflow as tf
import keras.backend as K


def get_gpu_session(ratio=None, interactive=False):
    config = tf.ConfigProto(allow_soft_placement=True)
    if ratio is None:
        config.gpu_options.allow_growth = True
    else:
        config.gpu_options.per_process_gpu_memory_fraction = ratio
    if interactive:
        sess = tf.InteractiveSession(config=config)
    else:
        sess = tf.Session(config=config)
    return sess


def set_gpu_usage(ratio=None):
    sess = get_gpu_session(ratio)
    K.set_session(sess)


WEIGHTS_PATH = '/home/liushrui/code/dsb2018/weights/'


class PathConfig:

    BASE = 'UNet'

    def __init__(self):
        self.today = datetime.today()

        self.path = os.path.join(WEIGHTS_PATH, self.BASE,
                                 self.today.strftime("%y%m%d%H%M%S"))
        os.makedirs(self.path)
        print("Path setup.")
        with open(os.path.join(self.path, 'config.json'), 'w') as f:
            f.write(json.dumps(self.get_attrs(), indent=4))

    def now(self):
        return datetime.today()

    def get_attrs(self):
        attrs = dir(self)
        ret = {}
        for attr in attrs:
            if not attr.startswith("__"):
                ret[attr] = str(getattr(self, attr))
        return ret

    @property
    def weight_path(self):
        return os.path.join(self.path, "weights.{epoch:02d}.hdf5")

    @property
    def best_path(self):
        return os.path.join(self.path, "best.hdf5")

    @property
    def csv_path(self):
        return os.path.join(self.path, 'training.csv')

    @property
    def log_path(self):
        return os.path.join(self.path, 'logs/')

    @property
    def yaml_path(self):
        return os.path.join(self.path, 'model.yaml')

    @property
    def final_path(self):
        return os.path.join(self.path, "final.hdf5")

