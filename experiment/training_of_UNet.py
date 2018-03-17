import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

import _init_path
from UNet import UNet
from inference import inference
from helper import set_gpu_usage
from config import ClfResumeConfig
from metrics import mean_iou, dice_coef_loss

if __name__ == '__main__':
    conf = ClfResumeConfig()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(conf.gpu)
    set_gpu_usage()

    # Load Data
    with np.load(conf.train_load_path) as f:
        train_images = f['images']
        train_masks = f['masks']

    with np.load(conf.test_load_path) as f:
        test_images = f['images']
        test_image_shapes = f['shapes']

    train_data = train_images / 255
    train_labels = np.expand_dims(train_masks, axis=-1)

    # Model
    model = UNet()
    if conf.loss == 'dice_coef_loss':
        model.compile(optimizer=conf.optimizer, loss=dice_coef_loss, metrics=[mean_iou])
    elif conf.loss == 'binary_cross_entropy':
        model.compile(optimizer=conf.optimizer, loss='binary_cross_entropy', metrics=[mean_iou])
    else:
        raise()
    model.summary()

    checkpointer = ModelCheckpoint(filepath=conf.weight_path, verbose=1, period=5, save_weights_only=True)
    best_keeper = ModelCheckpoint(filepath=conf.best_path, verbose=1, save_weights_only=True,
                                  monitor='val_mean_iou', save_best_only=True, period=1, mode='max')

    csv_logger = CSVLogger(conf.csv_path)
    tensorboard = TensorBoard(log_dir=conf.log_path)

    early_stopping = EarlyStopping(monitor='val_mean_iou', min_delta=0, mode='max', patience=conf.val_patience, verbose=1)
    lr_reducer = ReduceLROnPlateau(monitor='val_mean_iou', factor=conf.lr_reduce_ratio, patience=conf.lr_patience,
                                   verbose=1, mode='max', epsilon=1.e-5, cooldown=conf.cooldown, min_lr=conf.min_lr)

    with open(conf.yaml_path, "w") as f:
        f.write(model.to_yaml())

    print("Started training @%s." % conf.now())

    model.fit(train_data, train_labels, validation_split=0.1,
            batch_size=conf.steps, epochs=conf.epochs, #max_queue_size=conf.steps * 50,
            callbacks=[checkpointer, best_keeper, csv_logger, early_stopping, lr_reducer, tensorboard])

    model.save_weights(conf.final_path)
    print("Finished @%s." % conf.now())

