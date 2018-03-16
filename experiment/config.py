from helper import PathConfig
from keras.optimizers import Adam, RMSprop, SGD
from metrics import dice_coef_loss

class ClfResumeConfig(PathConfig):

    train_load_path = '../data/stage1_train_data_compressed.npz'
    test_load_path = '../data/stage1_test_data_compressed.npz'

    gpu = 1    
    steps = 32
    epochs = 500

    # callbacks
    val_patience = 10
    lr_reduce_ratio = 0.333  # lr *= lr_reduce_ratio
    lr_patience = 5
    cooldown = 2
    min_lr = 1e-5




    loss = 'binary_crossentropy'
    # loss = dice_coef_loss
    optimizer = Adam(lr=3.e-3, decay=3.e-5)
    # optimizer = RMSprop(lr=3.33e-5, decay=1.e-5)
